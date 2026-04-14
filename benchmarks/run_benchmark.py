import os
import sys
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tf2onnx

# Ensure codegen is in path
workspace_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_dir / "src"))

from codegen.main import compile_onnx_to_st
from translation_validation.validation import (
    validate_translation,
    generate_test_inputs,
    infer_input_size,
)


def build_mlp_4(input_shape):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


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


def main():
    benchmarks_dir = workspace_dir / "benchmarks"
    keras_dir = benchmarks_dir / "python"
    onnx_dir = benchmarks_dir / "onnx"
    st_dir = benchmarks_dir / "st"
    results_dir = benchmarks_dir / "results"

    for d in [keras_dir, onnx_dir, st_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    input_shape = (20, 1)  # Like the temperature classifier (20 timesteps, 1 feature)

    suite = [
        ("MLP_Model_4", "MLP", "Model 4 rerun from report", build_mlp_4),
        # Targets ~1,000 params
        ("LSTM_Small", "LSTM", "1 layer, 14 units", build_lstm(14)),  # ~1,025 params
        ("GRU_Small", "GRU", "1 layer, 16 units", build_gru(16)),  # ~963 params
        (
            "CNN_Small",
            "CNN",
            "1 layer, 6 filters",
            build_cnn(6, layers=1),
        ),  # ~987 params
        (
            "ResNet_Small",
            "ResNet",
            "1 skip block, width 16",
            build_resnet(1, 16),
        ),  # ~931 params
        # Targets ~4,500 params
        ("LSTM_Medium", "LSTM", "1 layer, 32 units", build_lstm(32)),  # ~4,451 params
        ("GRU_Medium", "GRU", "1 layer, 36 units", build_gru(36)),  # ~4,359 params
        (
            "CNN_Medium",
            "CNN",
            "2 layers, 10 filters",
            build_cnn(10, layers=2),
        ),  # ~4,503 params
        (
            "ResNet_Medium",
            "ResNet",
            "2 skip blocks, width 32",
            build_resnet(2, 32),
        ),  # ~4,995 params
        # Targets ~7,500 params (Safely fits code footprint limit < 96 KB)
        (
            "LSTM_Large",
            "LSTM",
            "2 layers, 24 units",
            build_lstm(24, stacked=True),
        ),  # ~7,275 params
        (
            "GRU_Large",
            "GRU",
            "2 layers, 26 units",
            build_gru(26, stacked=True),
        ),  # ~6,477 params
        (
            "CNN_Large",
            "CNN",
            "2 layers, 16 filters",
            build_cnn(16, layers=2),
        ),  # ~6,035 params
        (
            "ResNet_Large",
            "ResNet",
            "3 skip blocks, width 32",
            build_resnet(3, 32),
        ),  # ~7,139 params
    ]

    results = []

    print(
        f"{'Model Name':<15} | {'Params':<8} | {'ONNX (KB)':<10} | {'ST (KB)':<10} | {'Max Err':<10} | Status"
    )
    print("-" * 85)

    for name, kind, desc, builder in suite:
        model = builder(input_shape)
        param_count = model.count_params()

        keras_path = keras_dir / f"{name}.keras"
        onnx_path = onnx_dir / f"{name}.onnx"
        st_path = st_dir / f"{name}.st"

        max_err = None

        # Save keras
        model.save(keras_path)

        # Export to ONNX
        try:
            spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(
                model, input_signature=spec, opset=13
            )
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            onnx_size = get_file_size_kb(onnx_path)
        except Exception as e:
            print(
                f"{name:<15} | {param_count:<8} | ERROR      |            | ONNX Export Failed: {str(e)}"
            )
            continue

        # Compile to ST
        try:
            compile_onnx_to_st(
                model_path=str(onnx_path),
                optimize=True,
                output_path=str(st_path),
                fb_name=f"{name}_FB",
            )
            st_size = get_file_size_kb(st_path)

            # Validation
            try:
                flat_input_size = infer_input_size(onnx_path)
                test_inputs = generate_test_inputs(
                    num_samples=100, input_size=flat_input_size, mean=20.0, std=10.0
                )
                val_res = validate_translation(st_path, onnx_path, test_inputs)
                max_err = val_res.get("max_abs_diff", None)
                status = (
                    "Success" if val_res.get("passed", False) else "Validation Failed"
                )
            except Exception as ve:
                max_err = None
                status = f"Validation Error"
                print(f"[{name}] Validation error: {ve}")

        except Exception as e:
            st_size = 0.0
            max_err = None
            status = f"ST Compile Failed"
            print(f"[{name}] Compile error: {e}")

        print(
            f"{name:<15} | {param_count:<8} | {onnx_size:<10.1f} | {st_size:<10.1f} | {max_err if max_err is not None else 'N/A':<10} | {status}"
        )

        results.append(
            {
                "Model Name": name,
                "Type": kind,
                "Description": desc,
                "Parameters": param_count,
                "ONNX Size (KB)": round(onnx_size, 2),
                "ST Size (KB)": round(st_size, 2),
                "Max Abs Error": max_err,
                "Status": status,
            }
        )

    df = pd.DataFrame(results)
    csv_path = results_dir / "benchmark_models.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path}")


if __name__ == "__main__":
    main()
