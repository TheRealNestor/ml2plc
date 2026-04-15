"""Quick inference on saved benchmark models (temperature classification).

Usage:
  python benchmarks/infer.py CNN1                   # test common temperatures
  python benchmarks/infer.py MLP3 --input 20.5      # single temperature (°C)
  python benchmarks/infer.py LSTM2 --input cold      # preset: cold/normal/hot
  python benchmarks/infer.py CNN1 --input 0.1 0.2 ...  # 20 raw normalized floats
"""

import argparse
import os
import numpy as np

# Suppress TF noise before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from pathlib import Path

KERAS_DIR = Path(__file__).resolve().parent / "python"
CLASS_NAMES = ["Cold", "Normal", "Hot"]
INPUT_SHAPE = (20, 1)
WINDOW = INPUT_SHAPE[0]

# Must match training normalization
TEMP_MIN = -50.0
TEMP_MAX = 170.0

PRESET_NAMES = ["cold", "normal", "hot", "freezing", "boiling"]


def normalize(t):
    """Normalize a temperature value to [0, 1] using the training range."""
    return (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)


def make_preset_input(name):
    """Generate a named preset: a constant temperature window."""
    presets = {
        "freezing": -30.0,
        "cold": 0.0,
        "normal": 20.0,
        "hot": 80.0,
        "boiling": 150.0,
    }
    temp = presets.get(name)
    if temp is None:
        return None
    return np.full(INPUT_SHAPE, normalize(temp), dtype=np.float32)


def make_constant_input(temp_celsius):
    """Fill all timesteps with a normalized temperature value."""
    return np.full(INPUT_SHAPE, normalize(temp_celsius), dtype=np.float32)


def is_float(s):
    """Check if a string can be parsed as a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def print_prediction(pred):
    """Pretty-print class probabilities with bar chart."""
    for cls, prob in zip(CLASS_NAMES, pred):
        bar = "█" * int(prob * 40)
        print(f"  {cls:<8}: {prob:.4f}  {bar}")
    print(f"  → {CLASS_NAMES[np.argmax(pred)]} ({np.max(pred):.2%})")


def run_model(model, x, label):
    """Run inference on a single input and print results."""
    pred = model.predict(x.reshape(1, *INPUT_SHAPE), verbose=0)[0]
    print(f"[{label}]")
    print_prediction(pred)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run temperature inference on a saved benchmark model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
input options:
  (omit)          Test a range of common temperatures
  20.5            A temperature in °C (broadcast to all 20 timesteps)
  cold            A preset: freezing(-30), cold(0), normal(20), hot(80), boiling(150)
  0.1 0.2 ...    Exactly 20 pre-normalized float values [0-1]
""",
    )
    parser.add_argument("model", help="Model name (e.g. CNN1, MLP3, LSTM2)")
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="A temperature in °C, a preset name, or 20 raw normalized floats. "
        "Omit to test a range of common temperatures.",
    )
    args = parser.parse_args()

    # Load model
    keras_path = KERAS_DIR / f"{args.model}.keras"
    if not keras_path.exists():
        print(f"Model not found: {keras_path}")
        available = sorted(p.stem for p in KERAS_DIR.glob("*.keras"))
        if available:
            print(f"Available: {', '.join(available)}")
        else:
            print(
                "No saved models found. Run 'python benchmarks/run_benchmark.py --train' first."
            )
        return

    model = tf.keras.models.load_model(keras_path)
    print(f"Loaded {args.model} ({model.count_params()} params)")
    print(f"Normalization: [{TEMP_MIN}, {TEMP_MAX}] → [0, 1]\n")

    # No --input: test a range of representative temperatures
    if args.input is None:
        test_temps = [
            (-40.0, "Arctic cold"),
            (-20.0, "Deep freeze"),
            (-5.0, "Below zero"),
            (0.0, "Freezing point"),
            (5.0, "Cold day"),
            (10.0, "Cool (boundary)"),
            (15.0, "Mild"),
            (20.5, "Room temp"),
            (25.0, "Warm day"),
            (30.0, "Hot (boundary)"),
            (35.0, "Hot day"),
            (50.0, "Very hot"),
            (80.0, "Extreme heat"),
            (100.0, "Boiling point"),
            (150.0, "Industrial heat"),
        ]
        for temp, desc in test_temps:
            x = make_constant_input(temp)
            run_model(model, x, f"{temp:>7.1f}°C  {desc}")
        return

    # Single argument
    if len(args.input) == 1:
        arg = args.input[0]

        # Check preset name first
        if make_preset_input(arg) is not None:
            x = make_preset_input(arg)
            run_model(model, x, f"preset: {arg}")
            return

        # Single numeric value → temperature in Celsius
        if is_float(arg):
            val = float(arg)
            x = make_constant_input(val)
            run_model(model, x, f"{val}°C")
            return

        print(f"Unknown input: '{arg}'")
        print(f"Expected a temperature in °C or a preset ({', '.join(PRESET_NAMES)}).")
        return

    # Multiple arguments → raw normalized float values
    if not all(is_float(v) for v in args.input):
        bad = [v for v in args.input if not is_float(v)]
        print(f"Non-numeric values: {bad}")
        return

    vals = [float(v) for v in args.input]
    if len(vals) != WINDOW:
        print(f"Expected {WINDOW} values, got {len(vals)}")
        return

    x = np.array(vals, dtype=np.float32).reshape(INPUT_SHAPE)
    run_model(model, x, "custom input (pre-normalized)")


if __name__ == "__main__":
    main()
