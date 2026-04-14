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


def main():
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

    results = []
    print(
        f"{'Name':<15} | {'Params':<8} | {'ONNX (KB)':<10} | {'ST (KB)':<10} | {'Deployable':<12} | {'Max Err':<10} | {'Status'}"
    )
    print("-" * 100)

    for name, kind, desc, builder in suite:
        # 1. Build Model
        model = builder(input_shape)

        # 2. Get the ground-truth parameter count directly from Keras
        param_count = model.count_params()

        # 3. Save and Convert
        keras_path = keras_dir / f"{name}.keras"
        model.save(keras_path, save_format="keras")

        onnx_path = onnx_dir / f"{name}.onnx"
        spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=13, output_path=str(onnx_path)
        )
        onnx_size = get_file_size_kb(onnx_path)

        # 4. Compile to ST
        st_path = st_dir / f"{name}.st"
        try:
            compile_onnx_to_st(
                model_path=str(onnx_path),
                optimize=True,
                output_path=str(st_path),
                fb_name=f"{name}_FB",
            )
            st_size = get_file_size_kb(st_path)

            # 5. Validation
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

        # Log to console
        print(
            f"{name:<15} | {param_count:<8} | {onnx_size:<10.1f} | {st_size:<10.1f} | {deployable:<12} | {max_err if max_err is not None else 0.0:<10.4e} | {status}"
        )

        results.append(
            {
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
        )

    df = pd.DataFrame(results)
    csv_path = results_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path}")


if __name__ == "__main__":
    main()
