"""
Transformer-based temperature classification model.

Task: Classify temperature sequences as cold / normal / hot based on
a window of historical readings using a Transformer architecture.

The Transformer uses multi-head self-attention to capture long-range
dependencies without recurrence, making it suitable for parallel processing
and capturing complex temporal patterns.

Architecture:
    Input (seq_len, 1)
      → Dense (embed_dim=32)
      → Positional Encoding (Add)
      → MultiHeadAttention (4 heads, key_dim=8)
      → Add (residual connection)
      → LayerNormalization
      → FeedForward Network (Dense 64 → Dense 32)
      → Add (residual connection)
      → LayerNormalization
      → GlobalAveragePooling1D
      → Dense(16, relu)
      → Dropout(0.2)
      → Dense(3, softmax)

Transformer Operations in ONNX:
    - MultiHeadAttention: Einsum operations for Q, K, V projections
    - LayerNormalization: Uses ReduceMean, ReduceProd (may not be supported yet)
    - Positional Encoding: Add operation
    - Attention: MatMul, Softmax, Dropout
    - FeedForward: Dense (Gemm/MatMul) with activation

Testing Purpose:
    This model is used to test ml2plc compiler support for Transformer
    architectures. It will identify which operations need to be implemented.

Usage:
    python examples/models/transformer_model.py [--model-name name] [--epochs N]

This will:
    1. Create & train the Transformer model
    2. Save it as .keras + .weights.h5
    3. Export to ONNX via tf2onnx
    4. Attempt to compile with ml2plc (may show unsupported ops)
    5. Print a short demo with sample predictions
    6. Document any missing compiler operations
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import argparse

# Add parent directory to path for base_model import
sys.path.insert(0, str(Path(__file__).parent))
from base_model import MLModel

# ── Temperature thresholds (shared with other example models) ──────────────
COLD_THRESHOLD = 10.0
HOT_THRESHOLD = 30.0
CLASS_NAMES = ["Cold (≤10°C)", "Normal (10-30°C)", "Hot (>30°C)"]


class TransformerTemperatureModel(MLModel):
    """Transformer-based classifier for temperature sensor sequences."""

    def __init__(self, sequence_length: int = 20, model_name: str | None = None):
        self.model: tf.keras.Model | None = None
        self.sequence_length = sequence_length

        self.models_dir = Path("examples/models/keras")
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            date_str = datetime.now().strftime("%d%m%Y")
            model_name = f"transformer_model_{date_str}"

        self.model_name = model_name
        self.model_path = self.models_dir / f"{model_name}.keras"

    def create_model(self) -> tf.keras.Model:
        """Build a Transformer model for sequence classification.

        Architecture:
            Input (seq_len, 1)
              → Dense (embed_dim=32)
              → Positional Encoding
              → TransformerEncoder Block:
                  - MultiHeadAttention (4 heads)
                  - LayerNormalization
                  - FeedForward (Dense 64 → Dense 32)
                  - LayerNormalization
              → GlobalAveragePooling1D
              → Dense(16, relu)
              → Dropout(0.2)
              → Dense(3, softmax)

        Note: This is a full Transformer with all standard operations.
        Some operations (ReduceProd for LayerNorm, etc.) may not yet be
        supported by ml2plc - these will be identified during compilation.
        """
        inputs = tf.keras.Input(
            shape=(self.sequence_length, 1), name="temperature_input"
        )

        embed_dim = 32
        # Project input to embedding dimension
        x = tf.keras.layers.Dense(embed_dim, name="embedding")(inputs)

        # Add positional encoding
        pos_encoding = self._get_positional_encoding(self.sequence_length, embed_dim)
        x = x + pos_encoding

        # Transformer encoder block
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=8, name="mha"
        )(x, x)
        x = tf.keras.layers.Add(name="add_attention")([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x)

        # Feed-forward network
        ff_output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", name="ff_dense_1"),
                tf.keras.layers.Dense(embed_dim, name="ff_dense_2"),
            ],
            name="ff_network",
        )(x)
        x = tf.keras.layers.Add(name="add_ff")([x, ff_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_2")(x)

        # Global pooling to aggregate sequence
        x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

        # Classification head
        x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="output")(x)

        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="transformer_temp_model"
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def _get_positional_encoding(self, seq_length: int, embedding_dim: int):
        """Generate positional encoding for the sequence.

        Standard sinusoidal positional encoding from "Attention is All You Need".
        """
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim)
        )
        pos_encoding = np.zeros((seq_length, embedding_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pos_encoding[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pos_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)

    def generate_training_data(
        self,
        samples: int = 5000,
        csv_output: str = "examples/data/temperature_data.csv",
    ):
        """Generate temperature data with cold/normal/hot labels."""
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
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        **kwargs,
    ):
        """Train the Transformer model."""
        if self.model is None:
            self.create_model()

        # Load or generate data
        temperatures, labels = self.load_data_from_csv()
        if temperatures is None:
            print("CSV not found, generating training data...")
            temperatures, labels = self.generate_training_data()

        X, y = self.create_sequences(temperatures, labels)
        print(f"Training data shape: {X.shape}, labels shape: {y.shape}")

        # Train
        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )

        # Save
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        return history

    def export_to_onnx(self, output_path: str | None = None) -> str:
        """Export model to ONNX format using tf2onnx.

        NOTE: For ml2plc Structured Text compilation, the ONNX model MUST have
        concrete input shapes (no dynamic/None dimensions).

        TensorFlow models typically export with dynamic batch dimensions.
        To fix this for ST compilation, you have options:

        Option 1 (Recommended): Use tf2onnx with input_signature specifying concrete shapes
          - Requires modifying the model export pipeline
          - Best for production use

        Option 2: Use onnx-simplifier to resolve shapes after export
          - onnx-simplifier --input-shape "temperature_input:1,20,1" model.onnx

        Option 3: Modify model input layer to use concrete shape
          - keras_input = tf.keras.Input(shape=(20, 1), batch_size=1)
        """
        if output_path is None:
            # Save ONNX to examples/models/onnx
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

    def demo(self, num_samples: int = 5):
        """Print a quick demo of predictions."""
        if self.model is None:
            self.create_model()

        print("\n" + "=" * 60)
        print(f"Transformer Temperature Classifier Demo ({self.model_name})")
        print("=" * 60)

        temperatures, labels = self.load_data_from_csv()
        if temperatures is None:
            print("No data available for demo")
            return

        X, _ = self.create_sequences(temperatures, labels)

        # Get predictions on first few samples
        predictions = self.model.predict(X[:num_samples])

        for i in range(min(num_samples, len(X))):
            pred_class = np.argmax(predictions[i])
            confidence = predictions[i][pred_class]
            print(
                f"\nSample {i+1}: {CLASS_NAMES[pred_class]} "
                f"(confidence: {confidence:.2%})"
            )
            for j, class_name in enumerate(CLASS_NAMES):
                print(f"  {class_name}: {predictions[i][j]:.4f}")

        print("\n" + "=" * 60 + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Create, train, and export a Transformer temperature model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for the model (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate fresh training data",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    parser.add_argument(
        "--skip-compile", action="store_true", help="Skip ml2plc compilation"
    )

    args = parser.parse_args()

    # Create model
    print("Creating Transformer model...")
    model = TransformerTemperatureModel(model_name=args.model_name)
    model.create_model()
    print(model.model.summary())

    # Generate data if requested
    if args.generate_data:
        print("\nGenerating training data...")
        model.generate_training_data()

    # Train
    print("\nTraining model...")
    model.train(epochs=args.epochs)

    # Demo
    model.demo()

    # Export to ONNX
    if not args.skip_export:
        print("Exporting to ONNX...")
        onnx_path = model.export_to_onnx()

        # Try to compile with ml2plc
        if not args.skip_compile:
            print(f"\nCompiling with ml2plc: {onnx_path}")
            compile_cmd = [sys.executable, "src/codegen/main.py", str(onnx_path), "-v"]
            try:
                result = subprocess.run(
                    compile_cmd, capture_output=True, text=True, check=True
                )
                print(result.stdout)
                print("✓ Compilation successful!")
            except subprocess.CalledProcessError as e:
                print(
                    f"\n⚠ Compilation encountered errors (expected for full Transformer):"
                )
                print(f"stderr:\n{e.stderr}")
                print("\n" + "=" * 70)
                print("MISSING OPERATIONS FOR ml2plc COMPILER:")
                print("=" * 70)

                # Parse stderr for unsupported operations
                stderr = e.stderr
                if "ReduceProd" in stderr:
                    print(
                        "  ✗ ReduceProd - Used in LayerNormalization (variance calculation)"
                    )
                if "Unsupported op" in stderr:
                    print("  ✗ Other unsupported operations detected")
                if "Dimension mismatch" in stderr:
                    print(
                        "  ✗ Dimension tracking issues (may be related to unsupported ops)"
                    )

                print("\nThese operations need to be implemented in ml2plc to support")
                print(
                    "full Transformer models. See src/codegen/onnx_to_ir/layer_extractors.py"
                )
                print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
