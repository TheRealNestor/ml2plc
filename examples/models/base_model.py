"""Minimal ML model interface for testing and training.

Defines a simple, flexible interface that any ML model can implement
for consistent ONNX export and training across the codebase.

Philosophy: Only require 3 essential methods. Everything else is optional.
This keeps the interface lean and works for any model type.
"""

from abc import ABC, abstractmethod


class MLModel(ABC):
    """Abstract base class for ML models.

    Minimal interface requiring only:
    1. create_model() - Build/initialize the model architecture
    2. train(epochs, **kwargs) - Train the model with specified epochs
    3. export_to_onnx(output_path) - Export to ONNX format

    All other methods (data loading, preprocessing, etc.) are optional
    and model-specific. This keeps the interface flexible and universal.

    Example:
        class MyModel(MLModel):
            def __init__(self):
                self.model = None

            def create_model(self):
                # Build architecture
                self.model = build_network()

            def train(self, epochs, **kwargs):
                # Train with specified epochs
                data = kwargs.get('data')
                self.model.fit(data, epochs=epochs)

            def export_to_onnx(self, output_path):
                # Export to ONNX
                return export_model(self.model, output_path)
    """

    @abstractmethod
    def create_model(self):
        """Create and initialize the model architecture.

        This should set up all model layers, structure, and compilation.
        Must be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    def train(self, epochs: int, **kwargs):
        """Train the model for a specified number of epochs.

        Args:
            epochs: Number of training epochs
            **kwargs: Additional training parameters (data, learning_rate, etc.)
                     These can be model-specific and optional.
        """
        pass

    @abstractmethod
    def export_to_onnx(self, output_path: str = None) -> str:
        """Export the model to ONNX format.

        Args:
            output_path: Path where ONNX file should be saved.
                        If None, use a sensible default.

        Returns:
            Path to the exported ONNX file
        """
        pass
