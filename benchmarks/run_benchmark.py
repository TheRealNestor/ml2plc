import os
import sys
import argparse
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tf2onnx
import numpy as np


# Ensure codegen is in path
workspace_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_dir / "src"))

from codegen.main import compile_onnx_to_st
from translation_validation.validation import (
    validate_translation,
    generate_test_inputs,
    infer_input_size,
)


# =============================================================================
# Synthetic data generation
# =============================================================================


def generate_synthetic_data(n_samples, input_shape=(20, 1), n_classes=3, seed=42):
    """Generate synthetic 3-class time series data for training.

    Classes:
      0 - Sine-like signals
      1 - Square-like signals
      2 - Gaussian noise
    """
    rng = np.random.RandomState(seed)
    seq_len = input_shape[0]
    t = np.linspace(0, 2 * np.pi, seq_len)

    X, y = [], []
    for _ in range(n_samples):
        cls = rng.randint(0, n_classes)
        if cls == 0:  # Sine
            signal = np.sin(t + rng.uniform(0, 2 * np.pi))
            signal += rng.normal(0, 0.1, seq_len)
        elif cls == 1:  # Square
            signal = np.sign(np.sin(t + rng.uniform(0, 2 * np.pi)))
            signal += rng.normal(0, 0.15, seq_len)
        else:  # Noise
            signal = rng.normal(0, 1.0, seq_len)
        X.append(signal)
        y.append(cls)

    X = np.array(X, dtype=np.float32).reshape(-1, *input_shape)
    y = np.array(y, dtype=np.int32)
    return X, y


def train_model(model, input_shape, epochs, verbose=0):
    """Train a model on synthetic data. Returns (model, test_accuracy)."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    X_train, y_train = generate_synthetic_data(3000, input_shape, seed=42)
    X_test, y_test = generate_synthetic_data(600, input_shape, seed=99)

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=verbose,
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return model, test_acc


def run_inference_demo(model, input_shape):
    """Run inference on representative inputs and print class distributions."""
    seq_len = input_shape[0]
    t = np.linspace(0, 2 * np.pi, seq_len)
    class_names = ["Sine", "Square", "Noise"]

    samples = {
        "Clean sine": np.sin(t).reshape(1, *input_shape).astype(np.float32),
        "Clean square": np.sign(np.sin(t)).reshape(1, *input_shape).astype(np.float32),
        "Pure noise": np.random.RandomState(42)
        .normal(0, 1, (1, *input_shape))
        .astype(np.float32),
        "Noisy sine": (np.sin(t) + np.random.RandomState(7).normal(0, 0.3, seq_len))
        .reshape(1, *input_shape)
        .astype(np.float32),
        "All zeros": np.zeros((1, *input_shape), dtype=np.float32),
    }

    print("    Inference demo:")
    for label, x in samples.items():
        pred = model.predict(x, verbose=0)[0]
        winner = class_names[np.argmax(pred)]
        probs = "  ".join(f"{c}: {p:.3f}" for c, p in zip(class_names, pred))
        print(f"      {label:<14} -> {winner:<7} ({probs})")


# =============================================================================
# Model builders
# =============================================================================


def build_mlp(units_list):
    def builder(input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        for u in units_list:
            model.add(tf.keras.layers.Dense(u, activation="relu"))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))
        return model

    return builder


def build_lstm(units, stacked=False):
    def builder(input_shape):
        if not stacked:
            return tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(units, input_shape=input_shape),
                    tf.keras.layers.Dense(3, activation="softmax"),
                ]
            )
        else:
            return tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        units, return_sequences=True, input_shape=input_shape
                    ),
                    tf.keras.layers.LSTM(units),
                    tf.keras.layers.Dense(3, activation="softmax"),
                ]
            )

    return builder


def build_gru(units, stacked=False):
    def builder(input_shape):
        if not stacked:
            return tf.keras.Sequential(
                [
                    tf.keras.layers.GRU(units, input_shape=input_shape),
                    tf.keras.layers.Dense(3, activation="softmax"),
                ]
            )
        else:
            return tf.keras.Sequential(
                [
                    tf.keras.layers.GRU(
                        units, return_sequences=True, input_shape=input_shape
                    ),
                    tf.keras.layers.GRU(units),
                    tf.keras.layers.Dense(3, activation="softmax"),
                ]
            )

    return builder


def build_cnn(filters, layers=1):
    def builder(input_shape):
        model = tf.keras.Sequential()
        # Reshape (20, 1) to (20, 1, 1) for Conv2D to avoid 1D Squeeze ops in ONNX
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Reshape((input_shape[0], 1, 1)))

        for _ in range(layers):
            model.add(
                tf.keras.layers.Conv2D(
                    filters, kernel_size=(3, 1), activation="relu", padding="same"
                )
            )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))
        return model

    return builder


def build_resnet(blocks, width):
    def builder(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        # Initial projection to width
        x = tf.keras.layers.Dense(width, activation="relu")(x)

        for _ in range(blocks):
            residual = x
            x = tf.keras.layers.Dense(width, activation="relu")(x)
            x = tf.keras.layers.Dense(width)(x)
            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    return builder


def get_file_size_kb(filepath):
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / 1024.0


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="ml2plc benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python run_benchmark.py                        # random weights (fast, default)
  python run_benchmark.py --train                # train 30 epochs then compile
  python run_benchmark.py --train --epochs 50    # train 50 epochs
  python run_benchmark.py --infer                # train + show inference demo
  python run_benchmark.py --infer --verbose 1    # train with progress bars
""",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models on synthetic data before conversion (default: random weights)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs when --train is set (default: 30)",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference demo on sample inputs after each model (implies --train)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Training verbosity: 0=silent, 1=progress bar, 2=one line/epoch (default: 0)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # --infer implies --train
    if args.infer:
        args.train = True

    benchmarks_dir = workspace_dir / "benchmarks"
    keras_dir = benchmarks_dir / "python"
    onnx_dir = benchmarks_dir / "onnx"
    st_dir = benchmarks_dir / "st"
    results_dir = benchmarks_dir / "results"

    for d in [keras_dir, onnx_dir, st_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    input_shape = (20, 1)

    suite = [
        # --- MLPs (deployable <=64KB) ---
        ("MLP1", "MLP", "2 layers, 8/8", build_mlp([8, 8])),
        ("MLP2", "MLP", "2 layers, 16/12", build_mlp([16, 12])),
        ("MLP3", "MLP", "3 layers, 16/12/8", build_mlp([16, 12, 8])),
        ("MLP4", "MLP", "5 layers, 12", build_mlp([12] * 5)),
        ("MLP5", "MLP", "11 layers, 8", build_mlp([8] * 11)),
        ("MLP6", "MLP", "2 layers, 32/24", build_mlp([32, 24])),
        ("MLP7", "MLP", "5 layers, 24", build_mlp([24] * 5)),
        ("MLP8", "MLP", "2 layers, 64/48", build_mlp([64, 48])),
        # --- LSTM (deployable <=64KB) ---
        ("LSTM1", "LSTM", "1 layer, 14 units", build_lstm(14)),
        ("LSTM2", "LSTM", "1 layer, 32 units", build_lstm(32)),
        # --- GRU (deployable <=64KB) ---
        ("GRU1", "GRU", "1 layer, 16 units", build_gru(16)),
        ("GRU2", "GRU", "1 layer, 36 units", build_gru(36)),
        # --- CNN (deployable <=64KB) ---
        ("CNN1", "CNN", "1 layer, 6 filters", build_cnn(6, layers=1)),
        ("CNN2", "CNN", "2 layers, 10 filters", build_cnn(10, layers=2)),
        # --- ResNet (deployable <=64KB) ---
        ("ResNet1", "ResNet", "1 skip block, width 16", build_resnet(1, 16)),
        ("ResNet2", "ResNet", "2 skip blocks, width 32", build_resnet(2, 32)),
        # --- Non-deployable models (>100KB) ---
        ("MLP9", "MLP", "2 layers, 128/96", build_mlp([128, 96])),
        ("LSTM3", "LSTM", "2 layers, 64 units", build_lstm(64, stacked=True)),
        ("GRU3", "GRU", "2 layers, 64 units", build_gru(64, stacked=True)),
        ("CNN3", "CNN", "3 layers, 32 filters", build_cnn(32, layers=3)),
        ("ResNet3", "ResNet", "3 skip blocks, width 64", build_resnet(3, 64)),
    ]

    if args.train:
        print(f"Mode: TRAINED models ({args.epochs} epochs on synthetic data)")
    else:
        print("Mode: RANDOM weights (compiler correctness only)")
    print()

    results = []

    # Build header
    cols = ["Name", "Params"]
    widths = [15, 8]
    if args.train:
        cols.append("Test Acc")
        widths.append(8)
    cols += ["ONNX (KB)", "ST (KB)", "Deployable", "Max Err", "Status"]
    widths += [10, 10, 12, 10, 20]

    header = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    for name, kind, desc, builder in suite:
        # 1. Build Model
        model = builder(input_shape)

        # 2. Get the ground-truth parameter count directly from Keras
        param_count = model.count_params()

        # 3. Optionally train
        test_acc = None
        if args.train:
            print(f"[{name}] Training {args.epochs} epochs...", end=" ", flush=True)
            model, test_acc = train_model(
                model, input_shape, epochs=args.epochs, verbose=args.verbose
            )
            print(f"acc={test_acc:.4f}")

            if args.infer:
                run_inference_demo(model, input_shape)

        # 4. Save and Convert
        keras_path = keras_dir / f"{name}.keras"
        model.save(keras_path, save_format="keras")

        onnx_path = onnx_dir / f"{name}.onnx"
        spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=13, output_path=str(onnx_path)
        )
        onnx_size = get_file_size_kb(onnx_path)

        # 5. Compile to ST
        st_path = st_dir / f"{name}.st"
        try:
            compile_onnx_to_st(
                model_path=str(onnx_path),
                optimize=True,
                output_path=str(st_path),
                fb_name=f"{name}_FB",
            )
            st_size = get_file_size_kb(st_path)

            # 6. Validation
            try:
                flat_input_size = infer_input_size(onnx_path)
                test_inputs = generate_test_inputs(100, flat_input_size)
                val_res = validate_translation(st_path, onnx_path, test_inputs)
                max_err = val_res.get("max_abs_err", val_res.get("max_abs_diff", 0.0))
                status = "OK" if val_res.get("passed", True) else "Validation Failed"
            except Exception as ve:
                max_err = None
                status = "Validation Error"
                print(f"[{name}] Validation error: {ve}")
        except Exception as e:
            st_size = 0.0
            max_err = None
            status = "ST Compile Failed"
            print(f"[{name}] Compile error: {e}")

        deployable = "Yes" if 0 < st_size <= 96.0 else "No"

        # Build result row
        err_str = f"{max_err:<10.4e}" if max_err is not None else f"{0.0:<10.4e}"
        row_vals = [f"{name:<15}", f"{param_count:<8}"]
        if args.train:
            row_vals.append(f"{test_acc:.4f}  " if test_acc is not None else "N/A     ")
        row_vals += [
            f"{onnx_size:<10.1f}",
            f"{st_size:<10.1f}",
            f"{deployable:<12}",
            err_str,
            status,
        ]
        print(" | ".join(row_vals))

        result_entry = {
            "Model Name": name,
            "Type": kind,
            "Description": desc,
            "Parameters": param_count,
            "ONNX Size (KB)": round(onnx_size, 2),
            "ST Size (KB)": round(st_size, 2),
            "Deployable": deployable,
            "Max Abs Error": max_err,
            "Status": status,
        }
        if args.train:
            result_entry["Test Accuracy"] = (
                round(test_acc, 4) if test_acc is not None else None
            )
        results.append(result_entry)

    df = pd.DataFrame(results)
    csv_path = results_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path}")


if __name__ == "__main__":
    main()
