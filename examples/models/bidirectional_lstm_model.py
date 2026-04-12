"""
Bidirectional LSTM-based temperature classification model.

Task: Classify temperature sequences as cold / normal / hot based on
a window of historical readings using a Bidirectional LSTM network.

The model processes sequences in both forward and backward temporal
directions and combines context from both directions.

Architecture:
    Input (seq_len, 1)
      → Bidirectional(LSTM(16, return_sequences=False))
      → Dense(16, relu)
      → Dropout(0.2)
      → Dense(3, softmax)

Usage:
    python examples/models/bidirectional_lstm_model.py [--model-name name] [--epochs N]
"""

import tensorflow as tf
from datetime import datetime
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))

from lstm_model import LSTMTemperatureModel, CLASS_NAMES


class BidirectionalLSTMTemperatureModel(LSTMTemperatureModel):
    """Bidirectional LSTM-based classifier for temperature sensor sequences."""

    def __init__(self, sequence_length: int = 20, model_name: str | None = None):
        super().__init__(sequence_length=sequence_length, model_name=model_name)

        if model_name is None:
            date_str = datetime.now().strftime("%d%m%Y")
            self.model_name = f"bilstm_model_{date_str}"
            self.model_path = self.models_dir / f"{self.model_name}.keras"

    def create_model(self) -> tf.keras.Model:
        """Build a Bidirectional LSTM model for sequence classification."""
        inputs = tf.keras.Input(
            shape=(self.sequence_length, 1), name="temperature_input"
        )

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(16, return_sequences=False, name="lstm_layer"),
            name="bilstm_layer",
        )(inputs)

        x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="output")(x)

        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="bilstm_temp_model"
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def compile_to_st(self, onnx_path: str, output_path: str | None = None) -> str:
        """Compile ONNX model to Structured Text using ml2plc."""
        if output_path is None:
            st_dir = Path("examples/models/structured_text")
            st_dir.mkdir(exist_ok=True)
            output_path = str(st_dir / f"{self.model_name}.st")

        Path(output_path).parent.mkdir(exist_ok=True)

        try:
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "codegen.main",
                onnx_path,
                "-o",
                output_path,
                "--fb-name",
                "BidirectionalLSTMTemperatureClassifier",
            ]

            print("Compiling ONNX to Structured Text...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ ST compilation successful: {output_path}")
                if result.stderr:
                    print(f"(Compiler output: {result.stderr[:200]}...)")

                with open(output_path, "r") as f:
                    st_code = f.read()

                if "Error lowering region" in st_code:
                    print("⚠ Warning: Errors detected in generated code")

                return st_code
            else:
                print(f"✗ ST compilation failed:\n{result.stderr}")
                raise RuntimeError("Structured Text compilation failed")
        except Exception as e:
            print(f"Error compiling to ST: {e}")
            raise

    def demo(self):
        """Run a demo: predict on sample sequences."""
        import numpy as np

        print("\n" + "=" * 60)
        print("DEMO: Bidirectional LSTM Predictions on sample sequences")
        print("=" * 60)

        test_sequences = [
            np.array([5.0] * self.sequence_length),
            np.array([20.0] * self.sequence_length),
            np.array([35.0] * self.sequence_length),
        ]

        for i, seq in enumerate(test_sequences):
            seq_batch = seq.reshape(1, self.sequence_length, 1)
            probs = self.model.predict(seq_batch, verbose=0)[0]
            class_idx = probs.argmax()
            confidence = probs[class_idx]
            print(
                f"  Sequence {i+1} (avg {seq.mean():.1f}°C): "
                f"{CLASS_NAMES[class_idx]} ({confidence:.1%})"
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create and compile Bidirectional LSTM model for temperature classification"
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
        help="Number of training epochs (default: 100)",
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
    print("Bidirectional LSTM Temperature Classification Model")
    print("=" * 60)

    model = BidirectionalLSTMTemperatureModel(
        sequence_length=args.seq_len, model_name=args.model_name
    )

    model.create_model()
    print("\nModel architecture:")
    model.model.summary()
    print()
    model.train(epochs=args.epochs)
    model.save_model()

    if not args.no_export:
        onnx_path = model.export_to_onnx()

        if not args.no_compile:
            st_code = model.compile_to_st(onnx_path)
            print(f"\nGenerated {len(st_code)} characters of ST code")

    if not args.no_demo:
        model.demo()

    print("\n" + "=" * 60)
    print("✓ Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
