"""
Convolutional temperature classification model.

Same task as local_model.py / local_model2.py (cold / normal / hot classification
from a window of 5 temperature readings), but uses Conv2D + MaxPool + Flatten + Dense
to exercise the non-sequential / spatial layer support in the ml2plc pipeline.

The 1-D temperature window (5, 1) is reshaped to a tiny 2-D "image" so that
standard 2D convolution can be applied. This is intentionally kept very small
so the resulting Structured Text fits comfortably in PLC memory (~96 KB).

Usage:
    python examples/models/conv_model.py [model_name]

This will:
    1.  Create & train the convolutional model
    2.  Save it as .keras  +  .weights.h5
    3.  Export to ONNX via tf2onnx
    4.  Run the ml2plc compiler (ONNX → IR → Structured Text)
    5.  Print a short demo with sample predictions
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import argparse


# ── Temperature thresholds (shared with other example models) ──────────────
COLD_THRESHOLD = 10.0
HOT_THRESHOLD = 30.0
CLASS_NAMES = ["Cold (≤10°C)", "Normal (10-30°C)", "Hot (>30°C)"]


class ConvTemperatureModel:
    """
    A small Conv2D-based classifier for temperature sensor readings.

    Architecture:
        Input (5, 1)
          → Reshape to (1, 5, 1)           # add a "channel" dim for Conv2D
          → Conv2D(8, kernel_size=(1,3), relu, padding=same)
          → Conv2D(16, kernel_size=(1,3), relu, padding=same)
          → MaxPool2D(pool_size=(1,2))
          → Flatten
          → Dense(16, relu)
          → Dense(3, softmax)
    """

    def __init__(self, sequence_length: int = 5, model_name: str | None = None):
        self.model: tf.keras.Model | None = None
        self.sequence_length = sequence_length

        self.models_dir = Path("examples/models/keras")
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"conv_temp_{timestamp}"

        self.model_name = model_name
        self.model_path = self.models_dir / f"{model_name}.keras"

    # ── Model definition ──────────────────────────────────────────────────

    def create_model(self) -> tf.keras.Model:
        """Build a small Conv2D model that fits in PLC memory."""
        inputs = tf.keras.Input(
            shape=(self.sequence_length, 1), name="temperature_input"
        )

        # Reshape (seq_len, 1) → (1, seq_len, 1) so Conv2D sees H=1, W=seq_len, C=1
        x = tf.keras.layers.Reshape((1, self.sequence_length, 1))(inputs)

        # Conv block 1
        x = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 3),
            padding="same",
            activation="relu",
            name="conv1",
        )(x)

        # Conv block 2
        x = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 3),
            padding="same",
            activation="relu",
            name="conv2",
        )(x)

        # Pool: reduce the width dimension by 2
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), name="pool1")(x)

        # Flatten from spatial to 1-D for the dense classifier
        x = tf.keras.layers.Flatten(name="flatten")(x)

        # Dense classifier head
        x = tf.keras.layers.Dense(16, activation="relu", name="dense1")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="output")(x)

        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="conv_temp_model"
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    # ── Data generation / loading ─────────────────────────────────────────

    def generate_training_data(
        self,
        samples: int = 10000,
        csv_output: str = "examples/data/temperature_data.csv",
    ):
        """Generate (or reuse) temperature data with cold/normal/hot labels."""
        np.random.seed(42)
        temperatures, labels = [], []

        for i in range(samples):
            if i == 0:
                temp = np.random.uniform(-10, 40)
            else:
                temp = temperatures[-1] + np.random.normal(0, 2)
                temp = np.clip(temp, -30, 150)

            if np.random.random() < 0.05:
                temp += np.random.uniform(-20, 20)
            elif np.random.random() < 0.02:
                temp = np.random.uniform(-30, 150)

            label = (
                "cold"
                if temp <= COLD_THRESHOLD
                else ("normal" if temp <= HOT_THRESHOLD else "hot")
            )
            temperatures.append(round(temp, 2))
            labels.append(label)

        csv_path = Path(csv_output)
        csv_path.parent.mkdir(exist_ok=True)
        df = pd.DataFrame({"temperature": temperatures, "label": labels})
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        return np.array(temperatures), np.array(labels)

    def load_data_from_csv(self, csv_path: str = "examples/data/temperature_data.csv"):
        """Load temperature data from CSV."""
        p = Path(csv_path)
        if not p.exists():
            return None, None
        df = pd.read_csv(p)
        print(f"Loaded {len(df)} readings from {p}")
        return df["temperature"].values, df["label"].values

    def create_sequences(self, temperatures, labels):
        """Sliding-window sequences → (N, seq_len, 1) + one-hot labels."""
        label_map = {"cold": 0, "normal": 1, "hot": 2}
        X, y = [], []
        for i in range(len(temperatures) - self.sequence_length):
            seq = temperatures[i : i + self.sequence_length]
            X.append(seq)
            cls = label_map.get(labels[i + self.sequence_length], 1)
            one_hot = [0, 0, 0]
            one_hot[cls] = 1
            y.append(one_hot)
        return (
            np.array(X).reshape(-1, self.sequence_length, 1),
            np.array(y),
        )

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        csv_path: str = "examples/data/temperature_data.csv",
        generate_new: bool = False,
        samples: int = 10000,
    ):
        if self.model is None:
            self.create_model()

        if generate_new or not Path(csv_path).exists():
            temps, labels = self.generate_training_data(samples, csv_path)
        else:
            temps, labels = self.load_data_from_csv(csv_path)
            if temps is None:
                temps, labels = self.generate_training_data(samples, csv_path)

        X, y = self.create_sequences(temps, labels)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        print(f"\n{'='*50}")
        print(f"Training Conv model  —  {len(X_train)} train / {len(X_val)} val")
        print(f"Input shape : {X_train.shape}")
        self.model.summary()
        print(f"{'='*50}\n")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=25,
                restore_best_weights=True,
                min_delta=0.001,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
            ),
        ]

        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=callbacks,
        )

        _, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        _, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nTrain accuracy: {train_acc:.4f}  |  Val accuracy: {val_acc:.4f}")

        return temps

    # ── Save / load helpers ───────────────────────────────────────────────

    def save_model(self):
        if self.model is None:
            print("No model to save.")
            return
        self.model.save(self.model_path)
        size_kb = self.model_path.stat().st_size / 1024
        print(f"Model saved → {self.model_path}  ({size_kb:.1f} KB)")

    def save_weights(self):
        if self.model is None:
            print("No model to save weights from.")
            return
        wp = Path(str(self.model_path).replace(".keras", ".weights.h5"))
        self.model.save_weights(wp, overwrite=True)
        print(f"Weights saved → {wp}")

    def load_model(self) -> bool:
        if self.model_path.exists():
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Loaded model from {self.model_path}")
            return True
        print(f"Model not found: {self.model_path}")
        return False

    # ── Prediction / demo ─────────────────────────────────────────────────

    def predict(self, recent_temperatures):
        """Predict class from the most recent `sequence_length` readings."""
        if self.model is None or len(recent_temperatures) < self.sequence_length:
            return None
        seq = np.array(recent_temperatures[-self.sequence_length :]).reshape(
            1, self.sequence_length, 1
        )
        probs = self.model.predict(seq, verbose=0)[0]
        cls = int(np.argmax(probs))
        return {
            "class": cls,
            "class_name": CLASS_NAMES[cls],
            "confidence": float(probs[cls]),
            "probabilities": {
                "cold": float(probs[0]),
                "normal": float(probs[1]),
                "hot": float(probs[2]),
            },
        }

    def demo(self):
        """Run through a handful of scenarios and print predictions."""
        scenarios = [
            ("Room temperature", [19, 20, 21, 20, 19]),
            ("Cold storage", [8, 6, 4, 2, 1]),
            ("Warm room", [25, 26, 28, 27, 26]),
            ("Hot water", [32, 38, 45, 52, 58]),
            ("Freezer", [0, -2, -5, -8, -10]),
            ("Oven", [35, 55, 85, 120, 150]),
        ]
        print(f"\n{'='*55}")
        print("  Conv Temperature Sensor Demo")
        print(f"{'='*55}")
        for name, temps in scenarios:
            pred = self.predict(temps)
            if pred:
                print(
                    f"  {name:20s}  →  {pred['class_name']}  "
                    f"({pred['confidence']:.1%})"
                )
        print()


# ── ONNX export helper ────────────────────────────────────────────────────


def export_to_onnx(keras_path: Path, onnx_path: Path):
    """Convert .keras → .onnx via tf2onnx CLI."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting ONNX: {keras_path} → {onnx_path}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--keras",
            str(keras_path),
            "--output",
            str(onnx_path),
        ],
        check=True,
    )
    print(f"ONNX model saved → {onnx_path}")


def compile_to_st(onnx_path: Path):
    """Run the ml2plc compiler on the ONNX model."""
    print(f"\nCompiling ONNX → Structured Text ...")
    subprocess.run(
        [sys.executable, "src/codegen/main.py", str(onnx_path)],
        check=True,
    )


# ── CLI entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train a Conv2D temperature classifier and compile to ST."
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Name for the model (default: auto-timestamped)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Only train & export, don't run the ST compiler",
    )
    args = parser.parse_args()

    model = ConvTemperatureModel(model_name=args.model_name)

    # 1. Train
    model.train(generate_new=False)

    # 2. Save
    model.save_model()
    model.save_weights()

    # 3. Export to ONNX
    onnx_dir = Path("examples/models/onnx")
    onnx_path = onnx_dir / f"{model.model_name}.onnx"
    export_to_onnx(model.model_path, onnx_path)

    # 4. Compile ONNX → ST
    if not args.skip_compile:
        try:
            compile_to_st(onnx_path)
        except subprocess.CalledProcessError as e:
            print(f"ST compilation failed: {e}")

    # 5. Quick demo
    model.demo()


if __name__ == "__main__":
    main()
