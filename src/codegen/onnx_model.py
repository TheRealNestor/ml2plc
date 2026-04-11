"""
Loads and analyzes ONNX models to extract weights, layer information, and model structure for code generation.
"""

import onnx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ONNXModel:
    """
    A class to load and analyze ONNX models, extracting weights, layer information,
    and model structure for later code generation.
    """

    def __init__(self, model_path: str | Path):
        """
        Initialize the analyzer with an ONNX model.

        Args:
            model_path: Path to the ONNX model file (string or Path object)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.graph = None
        self.weights = {}
        self.layers = []
        self.tensor_info = {}  # Maps tensor names to their types and shapes

    def load_model(self) -> bool:
        """
        Load the ONNX model from file.

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"ONNX model file not found: {self.model_path}")
                return False

            self.model = onnx.load(str(self.model_path))
            onnx.checker.check_model(self.model)
            logger.info(f"Successfully loaded ONNX model: {self.model_path}")

            self.graph = self.model.graph

            # Automatically build tensor info, extract weights, and analyze layers
            self._build_tensor_info()
            self.extract_weights()
            self.analyze_layers()

            return True

        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return False

    @staticmethod
    def parse_value_info(value: onnx.ValueInfoProto) -> Dict[str, Any]:
        """Extract dtype and shape from a ValueInfoProto."""
        t = value.type.tensor_type
        elem_type = t.elem_type
        onnx_type = onnx.helper.tensor_dtype_to_string(elem_type)

        shape = []
        for d in t.shape.dim:
            if d.dim_value > 0:  # Fixed dimension
                shape.append(d.dim_value)
            elif d.dim_param:  # Symbolic dimension
                shape.append(str(d.dim_param))
            else:  # Unknown dimension
                shape.append(None)

        return {
            "onnx_type": onnx_type,
            "shape": shape,
        }

    def _build_tensor_info(self):
        """Build tensor_info using ONNX shape inference.

        Extracts dtype and shape for all tensors in the graph:
        1. Graph inputs (excluding initializers)
        2. Graph outputs
        3. Intermediate tensors (from value_info)
        4. All node outputs (comprehensive fallback to ensure complete coverage)

        This ensures all intermediate tensors have type information, even if
        shape_inference doesn't populate value_info for them.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        try:
            inferred = onnx.shape_inference.infer_shapes(self.model)
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}. Using raw graph.")
            inferred = self.model

        tensor_info = {}
        initializer_names = {init.name for init in inferred.graph.initializer}

        # Step 1: Inputs (excluding initializers)
        for v in inferred.graph.input:
            if v.name not in initializer_names:
                tensor_info[v.name] = self.parse_value_info(v)

        # Step 2: Outputs
        for v in inferred.graph.output:
            tensor_info[v.name] = self.parse_value_info(v)

        # Step 3: Intermediate tensors from value_info
        for v in inferred.graph.value_info:
            tensor_info[v.name] = self.parse_value_info(v)

        # Step 4: Comprehensive node output extraction
        # This ensures all intermediate tensors are recorded, even if not in value_info.
        # For each unmapped output, infer dtype from:
        #   a) The node's input types (priority)
        #   b) The ONNX network input dtype (fallback)
        network_input_dtype = None
        for inp in inferred.graph.input:
            if inp.name not in initializer_names:
                parsed = self.parse_value_info(inp)
                if dtype := parsed.get("onnx_type"):
                    network_input_dtype = dtype
                    break  # Use first non-initializer input dtype

        for node in inferred.graph.node:
            for output_name in node.output:
                # Skip if already processed
                if output_name in tensor_info:
                    continue

                # Try to infer dtype from node inputs
                inferred_dtype = None
                for input_name in node.input:
                    if input_name and input_name in tensor_info:
                        dtype_candidate = tensor_info[input_name].get("onnx_type")
                        if dtype_candidate:
                            inferred_dtype = dtype_candidate
                            break

                # Fall back to network input dtype
                if inferred_dtype is None and network_input_dtype:
                    inferred_dtype = network_input_dtype

                if inferred_dtype:
                    tensor_info[output_name] = {
                        "onnx_type": inferred_dtype,
                        "shape": (),  # Shape will be inferred later if needed
                    }
                    logger.debug(
                        f"Inferred tensor type for {output_name}: {inferred_dtype} "
                        f"(from node '{node.op_type}')"
                    )

        self.tensor_info = tensor_info
        logger.info(f"Extracted tensor info for {len(self.tensor_info)} tensors.")

    def refresh_after_model_mutation(self):
        """Refresh all cached analysis after in-place graph/model changes.

        Use this after any pass mutates `self.model` (for example shape
        canonicalization), so `tensor_info`, `weights`, and `layers` stay
        synchronized with the current ModelProto.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        self.graph = self.model.graph
        self._build_tensor_info()
        self.extract_weights()
        self.layers = []
        self.analyze_layers()

    def extract_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract all weights and constants from the model.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping parameter names to numpy arrays
        """
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return {}

        weights = {}
        for initializer in self.graph.initializer:
            tensor_data = onnx.numpy_helper.to_array(initializer)
            weights[initializer.name] = tensor_data

        self.weights = weights
        return weights

    def analyze_layers(self) -> List[Dict[str, Any]]:
        """
        Analyze all layers/nodes in the model.

        Returns:
            List[Dict[str, Any]]: List of layer information dictionaries
        """
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        if self.layers:
            return self.layers

        layers = []
        for node in self.graph.node:
            layer_info = {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {},
            }

            # Extract attributes
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    layer_info["attributes"][attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    layer_info["attributes"][attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    layer_info["attributes"][attr.name] = attr.s.decode("utf-8")
                elif attr.type == onnx.AttributeProto.INTS:
                    layer_info["attributes"][attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    layer_info["attributes"][attr.name] = list(attr.floats)

            layers.append(layer_info)

        self.layers = layers
        return layers

    def get_input_output_info(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get information about model inputs and outputs.

        Returns:
            Tuple containing:
                - input_info: Dict with 'names', 'shapes', 'dtypes' keys
                - output_info: Dict with 'names', 'shapes', 'dtypes' keys
        """
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return {}, {}

        initializer_names = {init.name for init in self.graph.initializer}

        # Input info
        input_names = []
        input_shapes = []
        input_dtypes = []

        for input_tensor in self.graph.input:
            if input_tensor.name not in initializer_names:
                input_names.append(input_tensor.name)

                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(str(dim.dim_param))
                    else:
                        shape.append(None)

                input_shapes.append(shape)
                dtype = onnx.helper.tensor_dtype_to_string(
                    input_tensor.type.tensor_type.elem_type
                )
                input_dtypes.append(dtype)

        # Calculate input size (for first input)
        input_size = 1
        if input_shapes:
            first_shape = input_shapes[0]
            static_dims = [d for d in first_shape if isinstance(d, int) and d > 0]
            if static_dims:
                input_size = int(np.prod(static_dims))

        input_info = {
            "names": input_names,
            "shapes": input_shapes,
            "dtypes": input_dtypes,
            "size": input_size,
        }

        # Output info
        output_names = []
        output_shapes = []
        output_dtypes = []

        for output_tensor in self.graph.output:
            output_names.append(output_tensor.name)

            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(str(dim.dim_param))
                else:
                    shape.append(None)

            output_shapes.append(shape)
            dtype = onnx.helper.tensor_dtype_to_string(
                output_tensor.type.tensor_type.elem_type
            )
            output_dtypes.append(dtype)

        # Calculate output size (for first output)
        output_size = 1
        if output_shapes:
            first_shape = output_shapes[0]
            static_dims = [d for d in first_shape if isinstance(d, int) and d > 0]
            if static_dims:
                output_size = int(np.prod(static_dims))

        output_info = {
            "names": output_names,
            "shapes": output_shapes,
            "dtypes": output_dtypes,
            "size": output_size,
        }

        return input_info, output_info

    def print_model_summary(self):
        """Print a comprehensive summary of the model."""
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return

        print("\n" + "=" * 60)
        print("ONNX MODEL SUMMARY")
        print("=" * 60)

        print(f"Model path: {self.model_path}")
        print(f"IR Version: {self.model.ir_version}")
        print(f"Producer: {self.model.producer_name} {self.model.producer_version}")

        input_info, output_info = self.get_input_output_info()

        print(f"\nInputs ({len(input_info['names'])}):")
        for name, shape, dtype in zip(
            input_info["names"], input_info["shapes"], input_info["dtypes"]
        ):
            print(f"  - {name}: shape={shape}, dtype={dtype}")

        print(f"\nOutputs ({len(output_info['names'])}):")
        for name, shape, dtype in zip(
            output_info["names"], output_info["shapes"], output_info["dtypes"]
        ):
            print(f"  - {name}: shape={shape}, dtype={dtype}")

        layers = self.analyze_layers()
        print(f"\nLayers ({len(layers)}):")
        layer_types: dict[str, int] = {}
        for layer in layers:
            op = layer["op_type"]
            layer_types[op] = layer_types.get(op, 0) + 1
            print(
                f"  - {layer['name'] or '<unnamed>'}: "
                f"type={op}, inputs={layer['inputs']}, outputs={layer['outputs']}"
            )

        print("\nLayer type counts:")
        for op, count in sorted(layer_types.items()):
            print(f"  {op}: {count}")
        print("=" * 60)


def load_and_analyze_onnx_model(model_path: str | Path) -> ONNXModel:
    """
    Convenience function to load and analyze an ONNX model.

    Args:
        model_path: Path to the ONNX model file (string or Path object)

    Returns:
        ONNXModel: Loaded ONNX model, or None if loading failed
    """
    analyzer = ONNXModel(model_path)
    if analyzer.load_model():
        analyzer.analyze_layers()
        analyzer.print_model_summary()
        return analyzer
    else:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and analyze ONNX model")
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Name of the ONNX model file (without .onnx extension)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    models_dir = Path("examples") / "models" / "onnx"

    if not models_dir.exists():
        logger.error(f"ONNX models directory not found: {models_dir}")
        exit(1)

    onnx_models = list(models_dir.glob("*.onnx"))

    if not onnx_models:
        logger.error(f"No ONNX models found in {models_dir}")
        exit(1)

    # Select model
    if args.model_name:
        model_path = models_dir / f"{args.model_name}.onnx"
        if not model_path.exists():
            logger.error(f"Model '{args.model_name}.onnx' not found in {models_dir}")
            logger.info("\nAvailable models:")
            for model in onnx_models:
                logger.info(f"  - {model.stem}")
            exit(1)
    else:
        model_path = onnx_models[0]
        logger.info(f"No model specified, using: {model_path.stem}\n")

    analyzer = load_and_analyze_onnx_model(model_path)

    if analyzer:
        logger.info(f"\nExtracted {len(analyzer.weights)} weight tensors")
        logger.info(f"Found {len(analyzer.layers)} layers")
