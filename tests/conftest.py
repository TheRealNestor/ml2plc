"""Pytest configuration for efficient model caching and setup.

MODEL CACHING STRATEGY:
======================

1. STRUCTURAL TESTS (Untrained models)
   - setup_test_models() fixture (session scope, runs once per test session)
   - Exports untrained ONNX models to examples/models/onnx/
   - Used by: @pytest.mark.fast tests (test_lstm_structural, etc.)

2. FUNCTIONAL TESTS (Trained models)
   - train_minimal_model() fixture (called by individual tests)
   - Checks if <prefix>_trained.onnx exists in cache → use it (fast path)
   - If not cached → trains minimally (1 epoch) → cache it
   - Used by: @pytest.mark.slow tests (test_lstm_functional, etc.)

All models now implement the MLModel interface:
  - create_model() - Build architecture
  - train(epochs, **kwargs) - Train for N epochs
  - export_to_onnx(output_path) - Export to ONNX

This eliminates conditional logic (hasattr, try/except) and simplifies testing.
"""

import pytest
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================================
# Model Setup & Caching
# ============================================================================


def load_python_model_class(model_file: Path, class_name: str):
    """Dynamically load a model class from a Python file."""
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def export_or_load_model(
    model_file: Path,
    class_name: str,
    onnx_dir: Path,
    model_prefix: str,
    force_regenerate: bool = False,
) -> Path:
    """
    Load cached model, or quickly export untrained model.

    FAST PATH (cache hit): <100ms - just load from disk
    MEDIUM PATH (first run): ~10-20s - create & export untrained model (for structural tests)

    NOTE: Exported models are UNTRAINED. Use for compilation tests only,
    not for numerical validation. For validation, use conftest --regenerate-models.
    """
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # ── CACHE HIT: Fast path ──────────────────────────────────────────
    if not force_regenerate:
        candidates = list(onnx_dir.glob(f"{model_prefix}_model_*.onnx"))
        if candidates:
            print(f"  [CACHED] {model_prefix}: {candidates[0].name}", flush=True)
            return candidates[0]

    # ── CACHE MISS: Quick export (NO TRAINING) ────────────────────────
    print(f"  [EXPORT] {model_prefix}...", end="", flush=True)

    try:
        ModelClass = load_python_model_class(model_file, class_name)
        model = ModelClass()

        # Just create model architecture - NO TRAINING
        print(" [build]", end="", flush=True)
        model.create_model()

        # Export to ONNX
        print(" [onnx]", end="", flush=True)
        onnx_path = model.export_to_onnx()
        print(f" [OK]", flush=True)
        return Path(onnx_path)

    except Exception as e:
        print(f" [ERROR: {str(e)[:30]}]", flush=True)
        # Try to use any cached model as fallback
        candidates = list(onnx_dir.glob(f"{model_prefix}_model_*.onnx"))
        if candidates:
            print(f"  [FALLBACK] Using cached: {candidates[0].name}", flush=True)
            return candidates[0]
        raise


@pytest.fixture(scope="session", autouse=True)
def setup_test_models():
    """
    Session-scoped fixture: exports all models once at test session start.

    Uses smart caching:
    - First run: Exports all 6 models (~30-60 seconds total)
    - Subsequent runs: All models cached (~1 second total)
    """
    models_dir = Path(__file__).parent.parent / "examples" / "models"
    onnx_dir = models_dir / "onnx"
    python_models_dir = models_dir

    models_config = [
        ("lstm_model.py", "LSTMTemperatureModel", "lstm"),
        ("conv_model.py", "ConvTemperatureModel", "conv"),
        ("gru_model.py", "GRUTemperatureModel", "gru"),
        ("resnet_model.py", "ResNetTemperatureModel", "resnet"),
        ("local_model.py", "TemperaturePredictionModel", "local"),
        ("local_model2.py", "TemperaturePredictionModel", "local2"),
    ]

    print("\n" + "=" * 70)
    print("Setting up test models (session scope)")
    print("=" * 70)

    for model_file, class_name, prefix in models_config:
        try:
            export_or_load_model(
                python_models_dir / model_file,
                class_name,
                onnx_dir,
                prefix,
            )
        except Exception as e:
            print(f"  [SKIP] {prefix} (export failed, will use cache)", flush=True)

    print("=" * 70)


@pytest.fixture
def models_dir():
    """Return path to examples/models directory."""
    return Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Return temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def train_minimal_model(models_dir):
    """
    Factory fixture to train a model minimally (if not already cached).

    Strategy:
    - Check if trained model exists in cache → use it (fast path)
    - If not cached → train on MINIMAL data (default: 1 epoch, configurable) → cache it
    - Never retrain if cached

    All models now implement MLModel interface:
      - create_model()
      - train(epochs, **kwargs)
      - export_to_onnx(output_path)

    Usage:
        # Train with default epochs (1 for most, 5 for GRU by default)
        trained_onnx = train_minimal_model(
            model_file_path,
            "ClassName",
            "model_prefix"
        )

        # Or explicitly specify epochs
        trained_onnx = train_minimal_model(
            model_file_path,
            "ClassName",
            "model_prefix",
            epochs=10
        )
    """

    def _train_model(
        model_file: Path, class_name: str, prefix: str, epochs: int | None = None
    ) -> Path:
        onnx_dir = models_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # Determine epochs if not explicitly provided
        if epochs is None:
            # Default: more epochs for GRU (needs more training for convergence)
            # Other models use 1 epoch
            epochs = 5 if prefix == "gru" else 1

        # FAST PATH: Check if trained model already exists
        trained_model = onnx_dir / f"{prefix}_trained.onnx"
        if trained_model.exists():
            print(f"  [CACHED-TRAINED] {prefix}", flush=True)
            return trained_model

        # SLOW PATH: Train and cache
        print(f"  [TRAIN-MINIMAL] {prefix}...", end="", flush=True)
        try:
            ModelClass = load_python_model_class(model_file, class_name)
            model = ModelClass()

            # Build model
            print(" [build]", end="", flush=True)
            model.create_model()

            # Train with specified epochs
            print(f" [{epochs}-epoch]", end="", flush=True)
            model.train(epochs=epochs)

            # Export and cache
            print(" [export]", end="", flush=True)
            model.export_to_onnx(output_path=str(trained_model))
            print(" [OK]", flush=True)
            return trained_model

        except Exception as e:
            print(f" [ERROR: {str(e)[:50]}]", flush=True)
            raise

    return _train_model
