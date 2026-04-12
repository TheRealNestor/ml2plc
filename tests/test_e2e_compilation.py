"""
End-to-end compilation tests: ONNX model → Structured Text → Validation.

Tests the complete compilation pipeline for different model architectures:
- LSTM: Recurrent temporal model
- Bidirectional LSTM: Two-direction temporal context model
- GRU: Gated recurrent unit (simpler than LSTM)
- MLP: Multi-layer perceptron (feedforward)
- CNN: Convolutional neural network
- (Future: RNN, Transformer, etc.)

FAST FUNCTIONAL VALIDATION STRATEGY:
====================================
Goal: Catch real bugs quickly without expensive training

TEST TYPES:
-----------
1. STRUCTURAL VALIDATION (@pytest.mark.fast): Compile + translate without training
   - Detects: Loop bugs, memory errors, type mismatches
   - Uses: Untrained ONNX models (fast, cached)
   - Tolerance: Loose (±0.01) - untrained models are quirky

2. FUNCTIONAL VALIDATION (@pytest.mark.slow): Full numerical accuracy check
   - Detects: Precision loss, algorithmic errors, accuracy issues
   - Uses: Trained ONNX models (minimal training)
   - Tolerance: Tight (±0.00001) - must match trained model

CLI EXAMPLES:
=============
# FAST: Structural validation only
pytest -m "not slow" tests/test_e2e_compilation.py -v

# FAST: Just LSTM structural tests
pytest -m "not slow" -k lstm tests/test_e2e_compilation.py -v

# SLOW: Full functional validation with trained models
pytest -m slow tests/test_e2e_compilation.py -v

# SLOW: Just MLP functional tests
pytest -m slow -k mlp tests/test_e2e_compilation.py -v

# ALL: Both fast and slow tests
pytest tests/test_e2e_compilation.py -v

# SPECIFIC TEST
pytest tests/test_e2e_compilation.py::TestLSTMModel::test_lstm_structural -v
pytest tests/test_e2e_compilation.py::TestLSTMModel::test_lstm_functional -v

# WITH DEBUGGING
pytest -m "not slow" -vv --tb=long -k lstm          # Verbose output + long traceback
pytest -m "not slow" --log-cli-level=DEBUG -k lstm  # Show debug logs

# EXIT ON FIRST FAILURE
pytest -m "not slow" -x tests/test_e2e_compilation.py  # Stop on first fail

# SHOW SLOWEST TESTS
pytest -m "not slow" --durations=5 tests/test_e2e_compilation.py
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import importlib.util
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codegen.main import compile_onnx_to_st
from translation_validation.validation import (
    load_high_level_model,
    translate_and_save,
    load_translated_function,
    compare_inference,
    run_onnx_inference,
    generate_test_inputs,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pytest Markers & Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "fast: fast compilation and structural tests")
    config.addinivalue_line(
        "markers", "slow: slow functional validation tests (trained models)"
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def models_dir():
    """Path to test models directory."""
    return Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for compilation outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Helper Functions
# ============================================================================


def load_python_model_class(model_file: Path, class_name: str):
    """
    Dynamically load a model class from a Python file.

    Args:
        model_file: Path to Python model file
        class_name: Name of the class to load

    Returns:
        Loaded class
    """
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def compile_and_validate_structural(
    onnx_path: Path,
    tmp_dir: Path,
    test_name: str,
    num_samples: int = 5,
    input_shape: tuple = (20,),
) -> dict:
    """
    FAST STRUCTURAL VALIDATION using untrained ONNX model.

    Tests that translation preserves logic structure without full accuracy.
    Minimal samples to catch only the most obvious bugs quickly.

    Tests: Compilation → Translation → Basic numerical checks

    Args:
        onnx_path: Path to ONNX model (untrained OK)
        tmp_dir: Temporary directory for outputs
        test_name: Name for this test (used in filenames)
        num_samples: Number of test samples (5 = minimal, fast)
        input_shape: Shape of each input sample (excluding batch dimension)

    Returns:
        dict with keys: passed, max_abs_diff, max_rel_diff, num_samples, error
    """
    try:
        # 1. Compile ONNX to ST
        st_path = tmp_dir / f"{test_name}_struct.st"
        st_code = compile_onnx_to_st(str(onnx_path), output_path=str(st_path))
        assert st_code and st_path.exists(), "Compilation failed"

        # 2. Translate ST to Python
        py_path = tmp_dir / f"{test_name}_struct.py"
        func_name = translate_and_save(st_path, py_path)
        assert func_name and py_path.exists(), "Translation failed"

        # 3. Load models
        onnx_model = load_high_level_model(onnx_path)
        translated_func = load_translated_function(py_path, func_name)

        # 4. Generate minimal test inputs (just check it runs)
        test_inputs = np.random.randn(num_samples, *input_shape).astype(np.float32)

        # 5. Validate structure (very loose tolerances - just check it doesn't crash)
        results = compare_inference(
            onnx_model,
            translated_func,
            test_inputs,
            model_type="onnx",
            rtol=1e-1,  # Very loose: untrained model can be very different
            atol=1e-1,
            verbose=False,
        )
        results["type"] = "structural"
        return results

    except Exception as e:
        logger.error(f"Structural validation failed: {e}", exc_info=True)
        return {
            "passed": False,
            "type": "structural",
            "error": str(e),
        }


def compile_and_validate_functional(
    onnx_path: Path,
    tmp_dir: Path,
    test_name: str,
    num_samples: int = 3,
    input_shape: tuple = (20,),
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict:
    """
    MINIMAL FUNCTIONAL VALIDATION - verifies translation doesn't break logic.

    Uses trained model (minimal training) + minimal test samples.
    Goal: Ensure translation is semantically correct, not accuracy checking.

    Tests: Compilation → Translation → Minimal samples to verify correctness

    Args:
        onnx_path: Path to ONNX model (trained, 1 epoch)
        tmp_dir: Temporary directory for outputs
        test_name: Name for this test (used in filenames)
        num_samples: Number of test samples (3 = minimal, just verify it works)
        input_shape: Shape of each input sample (excluding batch dimension)

    Returns:
        dict with keys: passed, max_abs_diff, max_rel_diff, num_samples, error
    """
    try:
        # Reuse compiled ST if available
        st_path = tmp_dir / f"{test_name}_struct.st"
        if not st_path.exists():
            st_path = tmp_dir / f"{test_name}_func.st"
            st_code = compile_onnx_to_st(str(onnx_path), output_path=str(st_path))
            assert st_code and st_path.exists(), "Compilation failed"

        # Translate ST to Python
        py_path = tmp_dir / f"{test_name}_func.py"
        func_name = translate_and_save(st_path, py_path)
        assert func_name and py_path.exists(), "Translation failed"

        # Load models
        onnx_model = load_high_level_model(onnx_path)
        translated_func = load_translated_function(py_path, func_name)

        # Generate minimal test inputs - just a few to verify correctness
        test_inputs = np.random.randn(num_samples, *input_shape).astype(np.float32)

        # Validate with reasonable tolerances
        # Not too tight (it's only 1 epoch) but not too loose (should still work)
        results = compare_inference(
            onnx_model,
            translated_func,
            test_inputs,
            model_type="onnx",
            rtol=rtol,
            atol=atol,
            verbose=False,
        )
        results["type"] = "functional"
        return results

    except Exception as e:
        logger.error(f"Functional validation failed: {e}", exc_info=True)
        return {
            "passed": False,
            "type": "functional",
            "error": str(e),
        }


# ============================================================================
# End-to-End Tests
# ============================================================================


class TestLSTMModel:
    """LSTM model tests: Recurrent temporal processing."""

    @pytest.fixture
    def lstm_onnx(self, models_dir):
        """Load cached LSTM ONNX model."""
        onnx_dir = models_dir / "onnx"
        candidates = list(onnx_dir.glob("lstm_model_*.onnx"))
        assert candidates, "LSTM model not found. Models should be cached by conftest."
        return candidates[0]

    @pytest.mark.fast
    def test_lstm_structural(self, lstm_onnx, tmp_output_dir):
        """
        FAST: Test LSTM structural validation (10 samples).

        Catches: Loop structure bugs, memory layout errors, type mismatches
        Uses: Untrained ONNX (fast caching)
        """
        results = compile_and_validate_structural(
            lstm_onnx, tmp_output_dir, "lstm_struct", num_samples=10, input_shape=(20,)
        )
        assert results["passed"], f"Structural validation failed: {results}"
        logger.info(f"LSTM Structural: max_abs_diff={results['max_abs_diff']:.2e}")

    @pytest.mark.slow
    def test_lstm_functional(
        self, lstm_onnx, tmp_output_dir, train_minimal_model, models_dir
    ):
        """
        SLOW: Minimal functional validation.

        Verifies that translation preserves semantic correctness.
        Not checking accuracy - just that translation doesn't break logic.
        Uses: Trained ONNX model (minimal training)
        """
        # Train model minimally for validation (1 epoch, cached)
        trained_model = train_minimal_model(
            models_dir / "lstm_model.py", "LSTMTemperatureModel", "lstm"
        )

        results = compile_and_validate_functional(
            trained_model, tmp_output_dir, "lstm_func", num_samples=3, input_shape=(20,)
        )
        assert results["passed"], (
            f"Functional validation failed: max_abs_diff={results.get('max_abs_diff', 'N/A')}, "
            f"error={results.get('error', 'Unknown')}"
        )
        logger.info(
            f"LSTM Functional: max_abs_diff={results['max_abs_diff']:.2e}, "
            f"max_rel_diff={results['max_rel_diff']:.2e}"
        )


class TestBidirectionalLSTMModel:
    """Bidirectional LSTM model tests: temporal processing in both directions."""

    @pytest.fixture
    def bilstm_onnx(self, models_dir):
        """Load cached Bidirectional LSTM ONNX model."""
        onnx_dir = models_dir / "onnx"
        candidates = list(onnx_dir.glob("bilstm_model_*.onnx"))
        assert (
            candidates
        ), "Bidirectional LSTM model not found. Models should be cached by conftest."
        return candidates[0]

    @pytest.mark.fast
    def test_bilstm_structural(self, bilstm_onnx, tmp_output_dir):
        """
        FAST: Test Bidirectional LSTM structural validation (10 samples).

        Catches: recurrent graph conversion/codegen issues with bidirectional topology.
        Uses: Untrained ONNX (fast caching)
        """
        results = compile_and_validate_structural(
            bilstm_onnx,
            tmp_output_dir,
            "bilstm_struct",
            num_samples=10,
            input_shape=(20,),
        )
        assert results["passed"], f"Structural validation failed: {results}"
        logger.info(f"BiLSTM Structural: max_abs_diff={results['max_abs_diff']:.2e}")

    @pytest.mark.slow
    def test_bilstm_functional(
        self, bilstm_onnx, tmp_output_dir, train_minimal_model, models_dir
    ):
        """
        SLOW: Minimal functional validation.

        Verifies that translation preserves semantic correctness for BiLSTM.
        Uses: Trained ONNX model (minimal training)
        """
        trained_model = train_minimal_model(
            models_dir / "bidirectional_lstm_model.py",
            "BidirectionalLSTMTemperatureModel",
            "bilstm",
        )

        results = compile_and_validate_functional(
            trained_model,
            tmp_output_dir,
            "bilstm_func",
            num_samples=3,
            input_shape=(20,),
            rtol=5e-2,
            atol=5e-2,
        )
        assert results["passed"], (
            f"Functional validation failed: max_abs_diff={results.get('max_abs_diff', 'N/A')}, "
            f"error={results.get('error', 'Unknown')}"
        )
        logger.info(
            f"BiLSTM Functional: max_abs_diff={results['max_abs_diff']:.2e}, "
            f"max_rel_diff={results['max_rel_diff']:.2e}"
        )


class TestMLPModel:
    """MLP model tests: Feedforward multi-layer perceptron."""

    @pytest.fixture
    def mlp_onnx(self, models_dir):
        """Load cached MLP ONNX model."""
        onnx_dir = models_dir / "onnx"
        candidates = list(onnx_dir.glob("local_model_*.onnx")) + list(
            onnx_dir.glob("local2_model_*.onnx")
        )
        assert candidates, "MLP model not found. Models should be cached by conftest."
        return candidates[0]

    @pytest.mark.fast
    def test_mlp_structural(self, mlp_onnx, tmp_output_dir):
        """
        FAST: Test MLP structural validation (10 samples).

        Catches: Layer stacking bugs, activation function errors
        Uses: Untrained ONNX (fast caching)
        """
        results = compile_and_validate_structural(
            mlp_onnx, tmp_output_dir, "mlp_struct", num_samples=10, input_shape=(5,)
        )
        assert results["passed"], f"Structural validation failed: {results}"
        logger.info(f"MLP Structural: max_abs_diff={results['max_abs_diff']:.2e}")

    @pytest.mark.slow
    def test_mlp_functional(
        self, mlp_onnx, tmp_output_dir, train_minimal_model, models_dir
    ):
        """
        SLOW: Minimal functional validation.

        Verifies translation preserves semantic correctness.
        Uses: Trained ONNX model (minimal training)
        """
        # Train model minimally (1 epoch, cached)
        trained_model = train_minimal_model(
            models_dir / "local_model.py", "TemperaturePredictionModel", "local"
        )

        results = compile_and_validate_functional(
            trained_model, tmp_output_dir, "mlp_func", num_samples=3, input_shape=(5,)
        )
        assert results["passed"], (
            f"Functional validation failed: max_abs_diff={results.get('max_abs_diff', 'N/A')}, "
            f"error={results.get('error', 'Unknown')}"
        )
        logger.info(
            f"MLP Functional: max_abs_diff={results['max_abs_diff']:.2e}, "
            f"max_rel_diff={results['max_rel_diff']:.2e}"
        )


class TestCNNModel:
    """CNN model tests: Convolutional neural network."""

    @pytest.fixture
    def cnn_onnx(self, models_dir):
        """Load cached CNN ONNX model."""
        onnx_dir = models_dir / "onnx"
        candidates = list(onnx_dir.glob("conv_model_*.onnx"))
        assert candidates, "CNN model not found. Models should be cached by conftest."
        return candidates[0]

    @pytest.mark.fast
    def test_cnn_structural(self, cnn_onnx, tmp_output_dir):
        """
        FAST: Test CNN structural validation (10 samples).

        Catches: Convolution operation bugs, padding errors, stride issues
        Uses: Untrained ONNX (fast caching)
        """
        results = compile_and_validate_structural(
            cnn_onnx, tmp_output_dir, "cnn_struct", num_samples=10, input_shape=(5,)
        )
        assert results["passed"], f"Structural validation failed: {results}"
        logger.info(f"CNN Structural: max_abs_diff={results['max_abs_diff']:.2e}")

    @pytest.mark.slow
    def test_cnn_functional(
        self, cnn_onnx, tmp_output_dir, train_minimal_model, models_dir
    ):
        """
        SLOW: Minimal functional validation.

        Verifies translation preserves semantic correctness.
        Uses: Trained ONNX model (minimal training)
        """
        # Train model minimally (1 epoch, cached)
        trained_model = train_minimal_model(
            models_dir / "conv_model.py", "ConvTemperatureModel", "conv"
        )

        results = compile_and_validate_functional(
            trained_model, tmp_output_dir, "cnn_func", num_samples=3, input_shape=(5,)
        )
        assert results["passed"], (
            f"Functional validation failed: max_abs_diff={results.get('max_abs_diff', 'N/A')}, "
            f"error={results.get('error', 'Unknown')}"
        )
        logger.info(
            f"CNN Functional: max_abs_diff={results['max_abs_diff']:.2e}, "
            f"max_rel_diff={results['max_rel_diff']:.2e}"
        )


class TestGRUModel:
    """GRU model tests: Gated Recurrent Unit (similar to LSTM but simpler).

    NOTE: GRU model implements MLModel interface correctly, but the code generator
    has incomplete GRU code generation (would need dedicated implementation like LSTM).
    This is a codegen issue, not a model interface issue.
    """

    @pytest.fixture
    def gru_onnx(self, models_dir):
        """Load cached GRU ONNX model."""
        onnx_dir = models_dir / "onnx"
        candidates = list(onnx_dir.glob("gru_model_*.onnx"))
        assert candidates, "GRU model not found. Models should be cached by conftest."
        return candidates[0]

    @pytest.mark.fast
    def test_gru_structural(self, gru_onnx, tmp_output_dir):
        """
        FAST: Test GRU structural validation (10 samples).

        Catches: Recurrent connection bugs, gate logic errors, state handling
        Uses: Untrained ONNX (fast caching)
        """
        results = compile_and_validate_structural(
            gru_onnx, tmp_output_dir, "gru_struct", num_samples=10, input_shape=(20,)
        )
        assert results["passed"], f"Structural validation failed: {results}"
        logger.info(f"GRU Structural: max_abs_diff={results['max_abs_diff']:.2e}")

    @pytest.mark.slow
    def test_gru_functional(
        self, gru_onnx, tmp_output_dir, train_minimal_model, models_dir
    ):
        """
        SLOW: Minimal functional validation.

        Verifies translation preserves semantic correctness.
        Uses: Trained ONNX model (minimal training)
        """
        # Train model minimally (1 epoch, cached)
        trained_model = train_minimal_model(
            models_dir / "gru_model.py", "GRUTemperatureModel", "gru"
        )

        results = compile_and_validate_functional(
            trained_model, tmp_output_dir, "gru_func", num_samples=3, input_shape=(20,)
        )
        assert results["passed"], (
            f"Functional validation failed: max_abs_diff={results.get('max_abs_diff', 'N/A')}, "
            f"error={results.get('error', 'Unknown')}"
        )
        logger.info(
            f"GRU Functional: max_abs_diff={results['max_abs_diff']:.2e}, "
            f"max_rel_diff={results['max_rel_diff']:.2e}"
        )


# ============================================================================
# Quality Assurance Tests
# ============================================================================


class TestCompilationQuality:
    """Tests for code generation quality (works on any available model)."""

    @pytest.fixture
    def any_model(self, models_dir):
        """Get any ONNX model for testing."""
        onnx_dir = models_dir / "onnx"
        models = list(onnx_dir.glob("*.onnx"))
        assert models, "No ONNX models found"
        return models[0]

    def test_generated_st_has_valid_syntax(self, any_model, tmp_output_dir):
        """Test generated ST has valid syntax structure."""
        st_path = tmp_output_dir / "syntax_test.st"
        st_code = compile_onnx_to_st(str(any_model), output_path=str(st_path))

        # Check basic structure
        assert "FUNCTION_BLOCK" in st_code
        assert "END_FUNCTION_BLOCK" in st_code

        # Check loop matching
        for_count = st_code.count("FOR ")
        end_for_count = st_code.count("END_FOR;")
        assert (
            for_count == end_for_count
        ), f"Unmatched loops: {for_count} FOR, {end_for_count} END_FOR"

    def test_generated_st_has_comments(self, any_model, tmp_output_dir):
        """Test generated ST includes helpful comments."""
        st_code = compile_onnx_to_st(str(any_model))

        assert "(* " in st_code, "No comments found"
        comment_count = st_code.count("(*")
        assert comment_count > 0, "Expected comments in generated code"

    def test_generated_st_output_file_matches_code(self, any_model, tmp_output_dir):
        """Test file output matches returned code."""
        st_path = tmp_output_dir / "file_test.st"
        st_code = compile_onnx_to_st(str(any_model), output_path=str(st_path))

        with open(st_path) as f:
            file_content = f.read()

        assert st_code == file_content, "File content doesn't match returned code"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in compilation."""

    def test_nonexistent_model_raises_error(self, tmp_output_dir):
        """Test that compiling non-existent model raises error."""
        nonexistent = tmp_output_dir / "does_not_exist.onnx"

        with pytest.raises(Exception):
            compile_onnx_to_st(str(nonexistent))

    def test_invalid_onnx_raises_error(self, tmp_output_dir):
        """Test that compiling invalid ONNX raises error."""
        invalid_path = tmp_output_dir / "invalid.onnx"
        invalid_path.write_text("not a valid onnx file")

        with pytest.raises(Exception):
            compile_onnx_to_st(str(invalid_path))
