from st_to_python import translate_st_to_python
from pathlib import Path
import numpy as np
import importlib.util


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

    print(f"Model expects shape: {expected_shape}")  # Debug info
    print(f"Input shape before reshape: {input_data.shape}")  # Debug info

    # Handle different expected shapes
    # expected_shape might be like [None, 5, 1] or [None, 5] or [None, 1, 5]
    expected_rank = len(expected_shape)

    if expected_rank == 3:
        # Model expects 3D input
        if input_data.ndim == 2:
            batch_size, features = input_data.shape
            # Check if model expects (batch, features, 1) or (batch, 1, features)
            if expected_shape[1] == features or expected_shape[1] is None:
                # Model expects (batch, features, 1)
                input_data = np.expand_dims(input_data, axis=-1)
            elif expected_shape[2] == features or expected_shape[2] is None:
                # Model expects (batch, 1, features)
                input_data = np.expand_dims(input_data, axis=1)
    elif expected_rank == 2:
        # Model expects 2D input (batch, features) - already correct
        pass

    print(f"Input shape after reshape: {input_data.shape}")  # Debug info

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

        # Store first few comparisons for debugging
        if i < 3 or verbose:
            results["sample_comparisons"].append(
                {
                    "index": i,
                    "input": input_data.tolist(),
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
) -> np.ndarray:
    """
    Generate synthetic test inputs for translation validation.
    """
    np.random.seed(seed)

    inputs = []

    # Normal range inputs (most common case)
    inputs.append(np.random.randn(num_samples - 10, input_size).astype(np.float32))

    if include_edge_cases:
        inputs.append(np.zeros((1, input_size), dtype=np.float32))
        inputs.append(np.ones((1, input_size), dtype=np.float32))
        inputs.append(-np.ones((1, input_size), dtype=np.float32))
        inputs.append(np.full((1, input_size), 0.5, dtype=np.float32))
        inputs.append(np.full((1, input_size), -0.5, dtype=np.float32))
        inputs.append(np.random.randn(1, input_size).astype(np.float32) * 10)
        inputs.append(np.random.randn(1, input_size).astype(np.float32) * 0.01)
        inputs.append(np.random.uniform(-5, 5, (3, input_size)).astype(np.float32))

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
    args = parser.parse_args()

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

    # Generate test inputs
    test_inputs = generate_test_inputs(
        num_samples=args.num_samples, input_size=input_size
    )

    # Run full pipeline
    results = validate_translation(st_file, onnx_model_file, test_inputs)

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    # Report
    passed = results.get("passed", False)
    print(f"\n{'=' * 50}")
    print(f"  Validation {'PASSED ✓' if passed else 'FAILED ✗'}")
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
