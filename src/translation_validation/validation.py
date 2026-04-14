"""
NUMERICAL VALIDATION of generated ST code (after compilation).

This is the SECOND validation layer that runs AFTER compilation completes:
- Loads original ONNX model
- Translates generated ST code to Python
- Runs both on identical test inputs
- Compares numerical outputs element-by-element
- Reports pass/fail with error tolerances (atol, rtol)

DIFFERENT FROM: src/codegen/ir_to_st/validation.py
  That module does STRUCTURAL VALIDATION during compilation:
  - Validates code generation correctness
  - Checks buffer size specifications
  - Verifies multi-output layer handling
  - Catches code generation bugs early

VALIDATION PIPELINE:
  1. python src/codegen/main.py model.onnx
  2.    └─→ src/codegen/ir_to_st/validation.py (structural checks during generation)
  3.    └─→ Returns: model.st file
  4. python src/translation_validation/validation.py model.st [THIS FILE]
  5.    └─→ Loads original ONNX model
  6.    └─→ Translates ST to Python
  7.    └─→ Compares outputs numerically
  8.    └─→ Reports: Validation PASSED / FAILED

EXPECTED RESULTS:
  max_abs_diff < 1e-5  (absolute error < 0.00001)
  max_rel_diff < 1e-5  (relative error < 0.001%)
  All samples PASSED

Example:
    After compilation creates model.st:
    >>> python src/translation_validation/validation.py model.st
    >>> # Output: Validation PASSED [OK]
    >>> # Max absolute difference: 1.62e-06
    >>> # Max relative difference: 5.55e-05
"""

from .st_to_python import translate_st_to_python
from pathlib import Path
import numpy as np
import importlib.util
import logging

logger = logging.getLogger(__name__)


def translate_and_save(st_file: Path, save_file: Path) -> str:
    try:
        with open(st_file, "r") as file:
            st_code = file.read()

        python_code, func_name = translate_st_to_python(st_code)

        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)

        with open(save_file, "w") as file:
            file.write(python_code)

        return func_name
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_onnx_model(file_path: Path):
    """Load ONNX model using ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(file_path))
    return session


def load_keras_model(file_path: Path):
    import tensorflow as tf

    model = tf.keras.models.load_model(file_path)
    return model


def load_high_level_model(file_path: Path):
    if file_path.suffix == ".onnx":
        return load_onnx_model(file_path)
    if file_path.suffix in {".h5", ".keras"}:
        return load_keras_model(file_path)
    raise ValueError(f"Unsupported model file type: {file_path.suffix}")


def run_onnx_inference(session, input_data: np.ndarray) -> np.ndarray:
    """Run inference on ONNX model."""
    input_name = session.get_inputs()[0].name
    expected_shape = session.get_inputs()[0].shape

    # Handle different expected shapes
    # expected_shape might be like [None, 5, 1] or [None, 5] or [None, 1, 5]
    expected_rank = len(expected_shape)

    # Reshape input to match expected rank
    if expected_rank == 3 and input_data.ndim == 2:
        # Model expects 3D, we have 2D - expand dims
        batch_size = input_data.shape[0]
        # Try to infer which dimension should be 1
        # Usually: (batch, seq_len, features) or (batch, features, seq_len)
        if expected_shape[2] == 1 or (isinstance(expected_shape[2], str)):
            # (batch, ?, 1) format
            input_data = np.expand_dims(input_data, axis=-1)
        else:
            # (batch, 1, ?) format
            input_data = np.expand_dims(input_data, axis=1)
    elif expected_rank == 2 and input_data.ndim == 1:
        # Model expects 2D, we have 1D - add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

    result = session.run(None, {input_name: input_data.astype(np.float32)})

    # Squeeze extra dimensions from output if present
    output = result[0]
    while output.ndim > 2:
        output = np.squeeze(output, axis=-1)

    return output


def load_translated_function(py_file: Path, func_name: str):
    """Dynamically load the translated Python function."""
    spec = importlib.util.spec_from_file_location("translated_module", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def compare_inference(
    model,
    translated_func,
    test_inputs: np.ndarray,
    model_type: str = "onnx",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """
    Compare inference results between model and translated function.

    Returns a dict with comparison results.
    """
    results = {
        "passed": True,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "failed_indices": [],
        "sample_comparisons": [],
    }

    # Get model outputs based on type
    if model_type == "onnx":
        model_outputs = run_onnx_inference(model, test_inputs)
    else:
        model_outputs = model.predict(test_inputs, verbose=0)

    logger.info(f"Comparing {len(test_inputs)} test samples...")
    logger.debug(
        f"Model output shape: {model_outputs.shape}, Test inputs shape: {test_inputs.shape}"
    )

    for i, input_data in enumerate(test_inputs):
        translated_output = translated_func(input_data)
        model_output = model_outputs[i]

        # Convert to numpy if needed
        if not isinstance(translated_output, np.ndarray):
            translated_output = np.array(translated_output)

        abs_diff = np.abs(model_output - translated_output)
        max_abs = np.max(abs_diff)
        results["max_abs_diff"] = max(results["max_abs_diff"], max_abs)

        # Relative difference (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / (np.abs(model_output) + 1e-10)
            max_rel = np.max(rel_diff)
            results["max_rel_diff"] = max(results["max_rel_diff"], max_rel)

        if not np.allclose(model_output, translated_output, rtol=rtol, atol=atol):
            results["passed"] = False
            results["failed_indices"].append(i)

            if verbose and i < 5:
                logger.warning(
                    f"Sample {i} FAILED: max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}"
                )

        # Store first few comparisons for debugging
        if i < 3 or verbose:
            results["sample_comparisons"].append(
                {
                    "index": i,
                    "input": (
                        input_data.tolist()
                        if hasattr(input_data, "tolist")
                        else input_data
                    ),
                    "model_output": (
                        model_output.tolist()
                        if hasattr(model_output, "tolist")
                        else model_output
                    ),
                    "translated_output": (
                        translated_output.tolist()
                        if hasattr(translated_output, "tolist")
                        else translated_output
                    ),
                    "abs_diff": (
                        abs_diff.tolist() if hasattr(abs_diff, "tolist") else abs_diff
                    ),
                }
            )

    logger.info(
        f"Comparison complete: max_abs_diff={results['max_abs_diff']:.6e}, "
        f"max_rel_diff={results['max_rel_diff']:.6e}, "
        f"failed={len(results['failed_indices'])}/{len(test_inputs)}"
    )

    return results


def validate_translation(
    st_file: Path,
    model_file: Path,
    test_inputs: np.ndarray,
    save_dir: Path = None,
) -> dict:
    """
    Full validation pipeline: translate ST, load model, compare outputs.
    """
    if save_dir is None:
        save_dir = Path("src/translation_validation/tmp")

    save_file = save_dir / f"{st_file.stem}.py"

    # Translate ST to Python
    func_name = translate_and_save(st_file, save_file)
    if func_name is None:
        return {"error": "Translation failed"}

    translated_func = load_translated_function(save_file, func_name)

    model = load_high_level_model(model_file)
    model_type = "onnx" if model_file.suffix == ".onnx" else "keras"

    results = compare_inference(
        model, translated_func, test_inputs, model_type=model_type
    )
    return results


def generate_test_inputs(
    num_samples: int = 100,
    input_size: int = 5,
    seed: int = 42,
    include_edge_cases: bool = True,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Generate synthetic test inputs for translation validation.
    The inputs are drawn from a normal distribution N(mean, std^2).
    """
    np.random.seed(seed)

    inputs = []

    # Normal range inputs scaled to the model's domain
    n_normal = num_samples - 10 if include_edge_cases else num_samples
    base_samples = np.random.randn(n_normal, input_size)
    inputs.append((base_samples * std + mean).astype(np.float32))

    if include_edge_cases:
        inputs.append(np.zeros((1, input_size), dtype=np.float32))
        inputs.append(np.ones((1, input_size), dtype=np.float32))
        inputs.append(-np.ones((1, input_size), dtype=np.float32))
        # Add some edge cases explicitly scaled to the domain
        inputs.append(np.full((1, input_size), mean, dtype=np.float32))
        inputs.append(np.full((1, input_size), mean + 3 * std, dtype=np.float32))
        inputs.append(np.full((1, input_size), mean - 3 * std, dtype=np.float32))
        inputs.append(
            (np.random.randn(1, input_size) * std * 10 + mean).astype(np.float32)
        )
        inputs.append(
            (np.random.randn(1, input_size) * std * 0.01 + mean).astype(np.float32)
        )
        inputs.append(
            (np.random.uniform(-1, 1, (2, input_size)) * std * 2 + mean).astype(
                np.float32
            )
        )

    return np.vstack(inputs)


def resolve_onnx_for_st(st_file: Path, onnx_dir: Path = None) -> Path:
    """
    Find the ONNX model that corresponds to a given ST file.

    Matching strategy (first match wins):
      1. Exact stem match:       conv_temp.st  → conv_temp.onnx
      2. Prefix/timestamp match: conv_temp.st  → conv_temp_20260310_095535.onnx
    """
    if onnx_dir is None:
        onnx_dir = st_file.parent.parent / "onnx"

    stem = st_file.stem

    # 1. Exact match
    exact = onnx_dir / f"{stem}.onnx"
    if exact.exists():
        return exact

    # 2. Prefix match (handles timestamped names)
    candidates = sorted(onnx_dir.glob(f"{stem}*.onnx"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No ONNX model found for '{st_file.name}' in {onnx_dir}.\n"
        f"  Tried: {exact}\n"
        f"  Glob:  {stem}*.onnx (0 matches)\n"
        f"  Hint:  pass --onnx <path> explicitly."
    )


def infer_input_size(onnx_model_path: Path) -> int:
    """Read the ONNX model to determine the flat input size."""
    import onnx

    model = onnx.load(str(onnx_model_path))
    inp = model.graph.input[0]
    dims = [
        d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim
    ]
    # Skip batch dimension (first), multiply the rest
    flat = 1
    for d in dims[1:]:
        flat *= d
    return flat


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Translation-validation: compare ST output against ONNX model."
    )
    parser.add_argument(
        "st_file",
        type=Path,
        help="Path to the Structured Text (.st) file to validate.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to the ONNX model. If omitted, auto-resolved from the ST filename.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Flat input size (e.g. 5). If omitted, inferred from the ONNX model.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of random test samples (default: 100).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print all sample comparisons, not just the first 3.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help="Mean for the normal distribution (e.g. 20 for temp models).",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=None,
        help="Standard deviation for the normal distribution fallback.",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(name)s: %(message)s")

    st_file = args.st_file
    if not st_file.exists():
        parser.error(f"ST file not found: {st_file}")

    # Resolve ONNX model
    if args.onnx is not None:
        onnx_model_file = args.onnx
    else:
        onnx_model_file = resolve_onnx_for_st(st_file)
    print(f"ST file:    {st_file}")
    print(f"ONNX model: {onnx_model_file}")

    # Determine input size
    input_size = args.input_size or infer_input_size(onnx_model_file)
    print(f"Input size: {input_size}")

    mean = args.mean if args.mean is not None else 0.0
    std = args.std if args.std is not None else 1.0
    print(f"Sampling domain: Normal(mean={mean}, std={std})")

    # Generate test inputs
    test_inputs = generate_test_inputs(
        num_samples=args.num_samples, input_size=input_size, mean=mean, std=std
    )

    # Run full pipeline
    results = validate_translation(st_file, onnx_model_file, test_inputs)

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    # Report
    passed = results.get("passed", False)
    print(f"\n{'=' * 50}")
    print(f"  Validation {'PASSED [OK]' if passed else 'FAILED [FAIL]'}")
    print(f"{'=' * 50}")
    print(f"  Max absolute difference: {results.get('max_abs_diff', 'N/A'):.2e}")
    print(f"  Max relative difference: {results.get('max_rel_diff', 'N/A'):.2e}")

    if not passed:
        failed = results.get("failed_indices", [])
        print(f"  Failed on {len(failed)} / {args.num_samples} samples")

    # Sample comparisons
    comparisons = results.get("sample_comparisons", [])
    show = comparisons if args.verbose else comparisons[:3]
    if show:
        print(f"\n--- Sample Comparisons ({len(show)} shown) ---")
        for sample in show:
            print(f"\n  Sample {sample['index']}:")
            print(
                f"    Input:      {sample['input'][:5]}{'...' if len(sample['input']) > 5 else ''}"
            )
            print(f"    Model:      {sample['model_output']}")
            print(f"    Translated: {sample['translated_output']}")
            print(f"    Abs Diff:   {sample['abs_diff']}")


if __name__ == "__main__":
    main()
