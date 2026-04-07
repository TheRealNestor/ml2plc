"""
GRU-based temperature classification model.

Task: Classify temperature sequences as cold / normal / hot based on
a window of historical readings using a GRU (Gated Recurrent Unit).

GRU is similar to LSTM but with fewer parameters, making it more suitable
for resource-constrained PLCs.

Architecture:
    Input (seq_len, 1)
      → GRU(16, return_sequences=False)
      → Dense(8, relu)
      → Dense(3, softmax)

Usage:
    python examples/models/gru_model.py [--model-name name] [--epochs N]

This will:
    1. Create & train the GRU model
    2. Save it as .keras + .weights.h5
    3. Export to ONNX via tf2onnx
    4. Run the ml2plc compiler (ONNX → IR → Structured Text)
    5. Print a short demo with sample predictions
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


class GRUTemperatureModel:
    """GRU-based classifier for temperature sensor sequences."""

    def __init__(self, sequence_length: int = 20, model_name: str | None = None):
        self.model: tf.keras.Model | None = None
        self.sequence_length = sequence_length

        self.models_dir = Path("examples/models/keras")
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"gru_temp_{timestamp}"

        self.model_name = model_name
        self.model_path = self.models_dir / f"{model_name}.keras"

    def create_model(self) -> tf.keras.Model:
        """Build a GRU model for sequence classification."""
        inputs = tf.keras.Input(
            shape=(self.sequence_length, 1), name="temperature_input"
        )

        # GRU layer: simpler than LSTM, fewer parameters
        x = tf.keras.layers.GRU(16, return_sequences=False, name="gru_layer")(inputs)

        # Dense layers for classification
        x = tf.keras.layers.Dense(8, activation="relu", name="dense_1")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="output")(x)

        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="gru_temp_model"
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def generate_training_data(
        self,
        samples: int = 5000,
        csv_output: str = "examples/data/temperature_data.csv",
    ):
        """Generate temperature data."""
        np.random.seed(42)
        temperatures, labels = [], []

        for i in range(samples):
            if i == 0:
                temp = np.random.uniform(-10, 40)
            else:
                temp = temperatures[-1] + np.random.normal(0, 1.5)
                temp = np.clip(temp, -30, 150)

            if np.random.random() < 0.05:
                temp += np.random.uniform(-15, 15)
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
        """Create sliding-window sequences."""
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

    def train(
        self,
        csv_path: str = "examples/data/temperature_data.csv",
        generate_new: bool = False,
        epochs: int = 100,
        verbose: int = 1,
    ):
        """Train the model with early stopping."""
        if generate_new:
            temps, labels = self.generate_training_data(csv_output=csv_path)
        else:
            temps, labels = self.load_data_from_csv(csv_path)
            if temps is None:
                print("CSV not found. Generating new data...")
                temps, labels = self.generate_training_data(csv_output=csv_path)

        X, y = self.create_sequences(temps, labels)

        print(f"Training on {len(X)} sequences of length {self.sequence_length}...")

        # Add early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10, restore_best_weights=True, verbose=1
        )

        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=32,
            verbose=verbose,
            callbacks=[early_stopping],
        )

    def save_model(self):
        """Save model in Keras format and SavedModel format."""
        # Save as Keras format (.keras file)
        self.model.save(str(self.model_path))
        print(f"Saved model to {self.model_path}")

        # Also save weights separately
        weights_path = self.model_path.with_suffix(".weights.h5")
        self.model.save_weights(str(weights_path))

    def export_to_onnx(self, output_path: str | None = None) -> str:
        """Export model to ONNX format using tf2onnx."""
        if output_path is None:
            # Save ONNX to examples/models/onnx, not keras subdirectory
            onnx_dir = Path("examples/models/onnx")
            onnx_dir.mkdir(exist_ok=True)
            output_path = str(onnx_dir / f"{self.model_name}.onnx")

        Path(output_path).parent.mkdir(exist_ok=True)

        # Create temporary SavedModel for tf2onnx conversion
        saved_model_dir = Path(self.model_path.parent) / ".tmp_saved_model"
        tf.saved_model.save(self.model, str(saved_model_dir))

        try:
            cmd = [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(saved_model_dir),
                "--output",
                output_path,
                "--opset",
                "13",
            ]

            print(f"Exporting to ONNX: {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up temporary SavedModel
            import shutil

            if saved_model_dir.exists():
                shutil.rmtree(saved_model_dir)

            if result.returncode == 0:
                print(f"✓ ONNX export successful: {output_path}")
                return output_path
            else:
                print(f"✗ ONNX export failed:\n{result.stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            sys.exit(1)

    def compile_to_st(self, onnx_path: str, output_path: str | None = None) -> str:
        """Compile ONNX model to Structured Text."""
        if output_path is None:
            # Save ST to examples/models/structured_text
            st_dir = Path("examples/models/structured_text")
            st_dir.mkdir(exist_ok=True)
            output_path = str(st_dir / f"{self.model_name}.st")

        Path(output_path).parent.mkdir(exist_ok=True)

        try:
            cmd = [
                sys.executable,
                "-m",
                "codegen.main",
                onnx_path,
                "-o",
                output_path,
                "--fb-name",
                "GRUTemperatureClassifier",
            ]

            print(f"Compiling ONNX to Structured Text...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ ST compilation successful: {output_path}")
                with open(output_path, "r") as f:
                    return f.read()
            else:
                print(f"✗ ST compilation failed:\n{result.stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"Error compiling to ST: {e}")
            sys.exit(1)

    def demo(self):
        """Run a demo: predict on sample sequences."""
        print("\n" + "=" * 60)
        print("DEMO: GRU Predictions on sample sequences")
        print("=" * 60)

        test_sequences = [
            np.array([5.0] * self.sequence_length),  # Cold
            np.array([20.0] * self.sequence_length),  # Normal
            np.array([35.0] * self.sequence_length),  # Hot
        ]

        for i, seq in enumerate(test_sequences):
            seq_batch = seq.reshape(1, self.sequence_length, 1)
            probs = self.model.predict(seq_batch, verbose=0)[0]
            class_idx = np.argmax(probs)
            confidence = probs[class_idx]
            print(
                f"  Sequence {i+1} (avg {seq.mean():.1f}°C): "
                f"{CLASS_NAMES[class_idx]} ({confidence:.1%})"
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create and compile GRU model for temperature classification"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name (default: auto-generated with timestamp)",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export step",
    )

    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip Structured Text compilation step",
    )

    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip demo predictions",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100, early stopping will terminate early if loss plateaus)",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="Sequence length for training (default: 20)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("GRU Temperature Classification Model")
    print("=" * 60)

    # Create model
    model = GRUTemperatureModel(
        sequence_length=args.seq_len, model_name=args.model_name
    )

    # Create and train
    model.create_model()
    print(f"\nModel architecture:")
    model.model.summary()
    print()
    model.train(epochs=args.epochs)
    model.save_model()

    # Export to ONNX
    if not args.no_export:
        onnx_path = model.export_to_onnx()

        # Compile to Structured Text
        if not args.no_compile:
            st_code = model.compile_to_st(onnx_path)
            print(f"\nGenerated {len(st_code)} characters of ST code")

    # Run demo
    if not args.no_demo:
        model.demo()

    print("\n" + "=" * 60)
    print("✓ Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
