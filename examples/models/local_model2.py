import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for base_model import
sys.path.insert(0, str(Path(__file__).parent))
from base_model import MLModel


class TemperaturePredictionModel(MLModel):
    def __init__(
        self, sequence_length=5, model_name=None
    ):  # Small window for real-time
        self.model = None
        self.sequence_length = sequence_length

        # Create models directory if it doesn't exist
        self.models_dir = Path("examples/models/keras")
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            date_str = datetime.now().strftime("%d%m%Y")
            model_name = f"local_model2_{date_str}"

        self.model_path = self.models_dir / f"{model_name}.keras"

    def create_model(self):
        """Create a model with residual connections that fits in PLC memory (~96KB)"""
        inputs = tf.keras.Input(shape=(self.sequence_length, 1))

        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)

        residual1 = x
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Add()([x, residual1])
        x = tf.keras.layers.Activation("relu")(x)

        residual2 = x
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Add()([x, residual2])
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def generate_training_data_with_labels(
        self, samples=10000, csv_output="examples/data/temperature_data.csv"
    ):
        """Generate realistic temperature sensor data with proper labeling and save to CSV"""
        np.random.seed(42)
        temperatures = []
        labels = []

        # Temperature thresholds: Cold: ≤10°C, Normal: 10-30°C, Hot: >30°C
        cold_threshold = 10.0
        hot_threshold = 30.0

        print(f"Generating {samples} temperature samples with labels...")

        for i in range(samples):
            # Generate sequential data with gradual changes
            if i == 0:
                # Start with a random temperature
                temp = np.random.uniform(-10, 40)
            else:
                temp_change = np.random.normal(0, 2)  # Small random change
                temp = temperatures[-1] + temp_change
                temp = np.clip(temp, -30, 150)  # Keep within realistic bounds

            # Occasionally introduce noise or outliers
            if np.random.random() < 0.05:  # 5% chance of noise
                temp += np.random.uniform(-20, 20)  # Large random jump
            elif np.random.random() < 0.02:  # 2% chance of extreme outlier
                temp = np.random.uniform(-30, 150)

            # Label based on actual temperature value
            if temp <= cold_threshold:
                label = "cold"
            elif temp <= hot_threshold:
                label = "normal"
            else:
                label = "hot"

            temperatures.append(round(temp, 2))
            labels.append(label)

        temperatures = np.array(temperatures)
        labels = np.array(labels)

        csv_path = Path(csv_output)
        csv_path.parent.mkdir(exist_ok=True)

        try:
            df = pd.DataFrame({"temperature": temperatures, "label": labels})
            df.to_csv(csv_path, index=False)
            print(f"Data saved to {csv_path}")
        except ImportError:
            # Manual CSV writing if pandas not available
            with open(csv_path, "w") as f:
                f.write("temperature,label\n")
                for temp, label in zip(temperatures, labels):
                    f.write(f"{temp},{label}\n")
            print(f"Data saved to {csv_path} (manual CSV writing)")

        return temperatures, labels

    def create_sequences_from_data(self, temperatures, labels):
        """Create sequences of temperature data for training"""
        X, y = [], []
        label_map = {"cold": 0, "normal": 1, "hot": 2}

        for i in range(len(temperatures) - self.sequence_length):
            sequence = temperatures[i : (i + self.sequence_length)]
            X.append(sequence)

            if labels is not None:
                target_label = labels[i + self.sequence_length]
                class_label = label_map.get(target_label, 1)
            else:
                sequence_mean = np.mean(sequence)
                if sequence_mean <= 10:
                    class_label = 0  # cold
                elif sequence_mean <= 30:
                    class_label = 1  # normal
                else:
                    class_label = 2  # hot

            one_hot = [0, 0, 0]
            one_hot[class_label] = 1
            y.append(one_hot)

        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)

    def train_model(
        self,
        csv_file_path="data/temperature_data.csv",
        generate_new_data=False,
        samples=10000,
    ):
        """Train the model using CSV data or generate new training data"""
        if self.model is None:
            self.create_model()

        # Generate new data if requested or if CSV doesn't exist
        if generate_new_data or not Path(csv_file_path).exists():
            print("Generating new training data...")
            temperatures, labels = self.generate_training_data_with_labels(
                samples, csv_file_path
            )
        else:
            # Try to load existing CSV data
            temperatures, labels = self.load_data_from_csv(csv_file_path)
            if temperatures is None or labels is None:
                print("Failed to load CSV, generating new data...")
                temperatures, labels = self.generate_training_data_with_labels(
                    samples, csv_file_path
                )

        print("Creating training sequences...")
        X, y = self.create_sequences_from_data(temperatures, labels)

        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training model with {len(X_train)} samples...")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")

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

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=16,
            verbose=1,
            callbacks=callbacks,
        )

        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)

        print(f"Training complete! Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        return temperatures

    def predict(self, recent_temperatures, steps_ahead=1):
        """Predict temperature category from recent sensor readings"""
        if self.model is None or len(recent_temperatures) < self.sequence_length:
            return None

        sequence = np.array(recent_temperatures[-self.sequence_length :]).reshape(
            1, self.sequence_length, 1
        )
        predictions = []
        class_names = ["Cold (≤10°C)", "Normal (10-30°C)", "Hot (>30°C)"]

        for i in range(steps_ahead):
            pred_probs = self.model.predict(sequence, verbose=0)[0]
            predicted_class = np.argmax(pred_probs)
            confidence = pred_probs[predicted_class]

            prediction = {
                "class": predicted_class,
                "class_name": class_names[predicted_class],
                "confidence": confidence,
                "probabilities": {
                    "cold": pred_probs[0],
                    "normal": pred_probs[1],
                    "hot": pred_probs[2],
                },
            }
            predictions.append(prediction)

        return predictions

    def save_model(self):
        """Save the complete model in .keras format"""
        if not self.model:
            print("No model to save.")
            return False

        try:
            self.model.save(self.model_path)
            size_mb = self.model_path.stat().st_size / (1024 * 1024)
            print(f"Model saved to {self.model_path.name} ({size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def save_weights(self):
        """Save only the model weights using TensorFlow format"""
        if not self.model:
            print("No model to save weights from.")
            return False

        weights_path = Path(str(self.model_path).replace(".keras", ".weights.h5"))
        try:
            self.model.save_weights(weights_path, overwrite=True)
            print(f"Weights saved to {weights_path.name} (TensorFlow format)")
            return True
        except Exception as e:
            print(f"Error saving weights: {e}")
            return False

    def load_model(self):
        """Load the complete model from .keras format"""
        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                size_mb = self.model_path.stat().st_size / (1024 * 1024)
                print(f"Model loaded from {self.model_path.name} ({size_mb:.2f} MB)")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False

        print(f"Model file not found: {self.model_path.name}")
        return False

    def load_weights(self):
        """Load only weights (requires model architecture to be created first)"""
        if self.model is None:
            print("Model architecture must be created before loading weights")
            return False

        weights_path = Path(str(self.model_path).replace(".keras", ".weights.h5"))

        try:
            self.model.load_weights(weights_path)
            print(f"Weights loaded from {weights_path.name}")
            return True
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return False

    def export_to_onnx(self, output_path: str | None = None) -> str:
        """Export model to ONNX format using tf2onnx."""
        if output_path is None:
            # Save ONNX to examples/models/onnx, not keras subdirectory
            onnx_dir = Path("examples/models/onnx")
            onnx_dir.mkdir(exist_ok=True)
            # Extract date from model name (e.g., "local_model2_10042026" -> "10042026")
            date_part = self.model_path.stem.split("_")[-1]
            output_path = str(onnx_dir / f"local2_model_{date_part}.onnx")

        Path(output_path).parent.mkdir(exist_ok=True)

        # Create temporary SavedModel for tf2onnx conversion
        saved_model_dir = Path(self.model_path.parent) / ".tmp_saved_model"
        tf.saved_model.save(self.model, str(saved_model_dir))

        try:
            import subprocess
            import sys

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

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up temporary SavedModel
            import shutil

            if saved_model_dir.exists():
                shutil.rmtree(saved_model_dir)

            if result.returncode == 0:
                return output_path
            else:
                print(f"Export failed:\n{result.stderr}")
                raise RuntimeError(f"ONNX export failed: {result.stderr}")
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            raise

    def load_data_from_csv(self, csv_file_path="temperature_data.csv"):
        """Load temperature data from CSV file"""
        csv_path = Path(csv_file_path)
        try:
            # Try pandas first, fall back to manual parsing if not available
            try:
                df = pd.read_csv(csv_path)
                temperatures = df["temperature"].values
                labels = df["label"].values
            except ImportError:
                # Manual CSV parsing if pandas not available
                temperatures = []
                labels = []
                with csv_path.open("r") as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        temp, label = line.strip().split(",")
                        temperatures.append(float(temp))
                        labels.append(label)
                temperatures = np.array(temperatures)
                labels = np.array(labels)

            print(f"Loaded {len(temperatures)} temperature readings from {csv_path}")
            return temperatures, labels

        except FileNotFoundError:
            print(f"CSV file {csv_path} not found. Using generated data instead.")
            return None, None
        except Exception as e:
            print(f"Error loading CSV: {e}. Using generated data instead.")
            return None, None

    @staticmethod
    def list_available_models():
        """List all available model files"""
        models_dir = Path("models")
        if not models_dir.exists():
            print("No models folder found.")
            return []

        all_files = list(models_dir.iterdir())

        # Find .keras files
        keras_files = [f for f in all_files if f.suffix == ".keras"]

        # Find weight files (look for .weights.h5 files)
        weight_files = [f for f in all_files if f.name.endswith(".weights.h5")]

        # Get unique model names
        model_names = set()

        for keras_file in keras_files:
            base_name = keras_file.stem
            model_names.add(base_name)

        for weight_file in weight_files:
            if weight_file.name.endswith(".weights.h5"):
                base_name = weight_file.name.replace(".weights.h5", "")
                model_names.add(base_name)

        if model_names:
            print("Available models:")
            for i, model_name in enumerate(sorted(model_names), 1):
                print(f"  {i}. {model_name}")

                # Check for .keras file
                keras_path = models_dir / f"{model_name}.keras"
                if keras_path.exists():
                    size_mb = keras_path.stat().st_size / (1024 * 1024)
                    print(f"     Complete model (.keras): {size_mb:.2f} MB")

                # Check for weights files
                weights_file = f"{model_name}.weights.h5"
                if any(f.name == weights_file for f in all_files):
                    print(f"     Weights only: available")
                print()
        else:
            print("No saved models found in models folder.")

        return sorted(list(model_names))

    def simulate_sensor_demo(self):
        """Simulate putting a temperature sensor into different environments"""
        print("\nTemperature Sensor Demo")
        print("=" * 50)
        print("Simulating sensor readings in different environments...")

        scenarios = [
            ("Room Temperature", [19, 20, 21, 20, 19]),  # Normal range
            ("Cold Storage", [8, 6, 4, 2, 1]),  # Cold range
            ("Refrigerator", [5, 4, 3, 3, 2]),  # Cold range
            ("Warm Room", [25, 26, 28, 27, 26]),  # Normal-high range
            ("Hot Water", [32, 38, 45, 52, 58]),  # Hot range
            ("Ice Water", [12, 8, 5, 2, -1]),  # Normal to cold transition
            ("Boiling Water", [30, 45, 65, 80, 95]),  # Normal to very hot
            ("Freezer", [0, -2, -5, -8, -10]),  # Very cold
            ("Sauna", [40, 50, 60, 70, 75]),  # Very hot
            ("Oven", [35, 55, 85, 120, 150]),  # Extremely hot
            (
                "Mixed Environment",
                [5, 25, 45, 8, 35],
            ),  # Mixed temperatures (test averaging)
            ("Outlier Test", [20, 21, 19, 22, -15]),  # Normal with extreme cold outlier
        ]

        for scenario_name, temps in scenarios:
            print(f"\n{scenario_name}:")
            print(f"   Sensor readings: {temps}")

            if len(temps) >= self.sequence_length:
                prediction = self.predict(temps)
                if prediction:
                    pred = prediction[0]
                    print(
                        f"   Prediction: {pred['class_name']} (confidence: {pred['confidence']:.1%})"
                    )
                    probs = pred["probabilities"]
                    print(
                        f"   Probabilities - Cold: {probs['cold']:.1%}, Normal: {probs['normal']:.1%}, Hot: {probs['hot']:.1%}"
                    )
            else:
                print(
                    f"     Need at least {self.sequence_length} readings for prediction"
                )
            print("   " + "-" * 40)


def main():
    """Demo the temperature prediction model"""
    print("Temperature Prediction System")
    print("=" * 40)

    available_models = TemperaturePredictionModel.list_available_models()

    # Let user choose to train new model or load existing one
    if available_models:
        choice = input("\nTrain new model? (y/n): ").lower().strip()
        if choice == "y":
            model_name = input(
                "Enter model name (or press Enter for auto-generated): "
            ).strip()
            model = TemperaturePredictionModel(
                model_name=model_name if model_name else None
            )
        else:
            # Use the most recent model (last in alphabetical order usually means most recent)
            latest_model = sorted(available_models)[-1]
            model = TemperaturePredictionModel(model_name=latest_model)
            if not model.load_model():
                print("Failed to load model, creating new one...")
                model = TemperaturePredictionModel()
    else:
        model = TemperaturePredictionModel()

    # Train or load model
    if not hasattr(model, "model") or model.model is None:
        if not model.load_model():
            print("\nTraining new model...")
            # Generate fresh training data
            temperatures = model.train_model(generate_new_data=False, samples=15000)

            # Save model and weights
            print("\nSaving model...")
            model.save_model()  # Save complete model
            model.save_weights()  # Also save weights separately

            recent_temps = temperatures[-10:]  # Last 10 readings for demo
        else:
            # Generate some sample data for demo
            recent_temps, _ = model.generate_training_data_with_labels(50)
            recent_temps = recent_temps[-10:]

    # Run sensor simulation demo
    model.simulate_sensor_demo()


if __name__ == "__main__":
    main()
