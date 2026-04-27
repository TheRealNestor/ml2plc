#!/usr/bin/env python3
"""
TensorFlow Model Quantization Tool

Refer also to: https://ai.google.dev/edge/litert/models/post_training_quantization

Post-training quantization techniques for TensorFlow Lite models:


QUANTIZATION METHODS:
  --type dynamic  (DEFAULT) - Dynamic range quantization
                             - 4x smaller, 2x-3x speedup on CPU
                             - Weights quantized to int8, activations dynamically quantized
                             - No representative dataset required
  
  --type float16            - Float16 quantization  
                             - 2x smaller, GPU acceleration
                             - Weights quantized to float16
                             - No representative dataset required
  
  --type int8               - Full integer quantization (integer with float fallback)
                             - 4x smaller, 3x+ speedup on CPU, Edge TPU, Microcontrollers
                             - All weights and activations quantized to int8
                             - Input/output remain float32 for compatibility
                             - Requires representative dataset for calibration
  
  --type int_only           - Integer only quantization
                             - Same size/speed benefits as int8
                             - Input/output also quantized to int8
                             - Compatible with integer-only hardware (microcontrollers, Edge TPU)
                             - Requires representative dataset for calibration

REPRESENTATIVE DATASET:
  - REQUIRED for: int8, int_only
  - NOT REQUIRED for: dynamic, float16
  - Used to calibrate activation ranges for quantization
  - Should be ~100-500 samples from training/validation data
  - Use --data flag to provide CSV file with input data

Usage examples:
  python quantize_model.py model.keras                                    # dynamic (default)
  python quantize_model.py model.keras --type float16                     # float16
  python quantize_model.py model.keras --type int8 --data data.csv        # full integer
  python quantize_model.py model.keras --type int_only --data data.csv    # integer only
"""

import argparse
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


class ModelQuantizer:
    def __init__(self, model_path, output_path=None, quantization_type="dynamic"):
        """
        Initialize the ModelQuantizer.
        
        Args:
            model_path (str): Path to the input model
            output_path (str): Path for the output quantized model
            quantization_type (str): Type of quantization to apply
        """
        self.model_path = Path(model_path)
        self.quantization_type = quantization_type
        
        if output_path:
            self.output_path = Path(output_path)
        else:
            # Generate output filename based on input and quantization type
            base_name = self.model_path.stem  # Get filename without extension
            suffix = f"_{quantization_type}_quant.tflite"
            output_filename = base_name + suffix
            self.output_path = self.model_path.parent / output_filename
        
        self.model = None
        self.converter = None
    
    def load_model(self):
        """Load the TensorFlow model from file."""
        try:
            if self.model_path.suffix.lower() in ['.h5', '.keras']:
                print(f"Loading Keras model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            elif self.model_path.is_dir():
                print(f"Loading SavedModel from {self.model_path}")
                self.converter = tf.lite.TFLiteConverter.from_saved_model(str(self.model_path))
            else:
                raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def create_representative_dataset(self, data_path=None, num_samples=100):
        """
        Create a representative dataset for full integer quantization.
        
        Args:
            data_path (str): Path to the dataset CSV file
            num_samples (int): Number of samples to use for calibration
        """
        if data_path and Path(data_path).exists():
            print(f"Loading representative dataset from {data_path}")
            try:
                df = pd.read_csv(data_path)
                
                # Check if this is sliding window data (multiple temp columns)
                temp_columns = [col for col in df.columns if col.startswith('temp_t')]
                
                if temp_columns and len(temp_columns) >= 5:
                    print(f"Found sliding window data with {len(temp_columns)} temperature columns")
                    # Use the sliding window columns as input features
                    data = df[temp_columns].values[:num_samples]
                    
                    def representative_dataset_gen():
                        for sample in data:
                            # Reshape to match model input: (1, sequence_length, 1)
                            # Each sample is already a row of 5 temperature values
                            yield [sample.reshape(1, len(temp_columns), 1).astype(np.float32)]
                    
                    print(f"Created representative dataset with {len(data)} samples, shape: (1, {len(temp_columns)}, 1)")
                    return representative_dataset_gen
                    
                elif 'temperature' in df.columns:
                    print("Found single temperature column, creating sequences...")
                    # Handle original format with single temperature column
                    temperatures = df['temperature'].values
                    
                    # Create sliding windows from single temperature data
                    sequence_length = 5  # Default sequence length
                    sequences = []
                    for i in range(len(temperatures) - sequence_length + 1):
                        if len(sequences) >= num_samples:
                            break
                        sequences.append(temperatures[i:i + sequence_length])
                    
                    def representative_dataset_gen():
                        for seq in sequences:
                            yield [seq.reshape(1, sequence_length, 1).astype(np.float32)]
                    
                    return representative_dataset_gen
                    
                else:
                    print("No recognized temperature data format found")
                    raise ValueError("CSV must contain either sliding window columns (temp_t-*) or a 'temperature' column")
                
            except Exception as e:
                print(f"Error creating representative dataset: {e}")
                print("Using random data for calibration...")
        
        # Fallback to random data
        print("Creating representative dataset with random data...")
        
        def representative_dataset_gen():
            for _ in range(num_samples):
                # Generate random input data based on model input shape
                if self.model:
                    input_shape = list(self.model.input_shape)
                    input_shape[0] = 1  # Set batch size to 1
                    random_input = np.random.random(input_shape).astype(np.float32)
                else:
                    # Default shape for temperature sensor model: (1, 5, 1)
                    random_input = np.random.random((1, 5, 1)).astype(np.float32)
                yield [random_input]
        
        return representative_dataset_gen
    
    def apply_dynamic_quantization(self):
        """Apply dynamic range quantization (weights only)."""
        print("Applying dynamic range quantization...")
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return self.converter.convert()
    
    def apply_int8_quantization(self, data_path=None):
        """Apply full integer quantization (int8)."""
        print("Applying full integer (int8) quantization...")
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set up representative dataset for calibration
        representative_dataset = self.create_representative_dataset(data_path)
        self.converter.representative_dataset = representative_dataset
        
        return self.converter.convert()
    
    def apply_float16_quantization(self):
        """Apply float16 quantization."""
        print("Applying float16 quantization...")
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.target_spec.supported_types = [tf.float16]
        return self.converter.convert()
    
    def apply_int_only_quantization(self, data_path=None):
        """Apply integer-only quantization."""
        print("Applying integer-only quantization...")
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set up representative dataset for calibration
        representative_dataset = self.create_representative_dataset(data_path)
        self.converter.representative_dataset = representative_dataset
        
        # Ensure compatibility with integer-only devices
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8
        
        return self.converter.convert()
    
    def quantize(self, data_path=None):
        """
        Perform the specified quantization on the model.
        
        Args:
            data_path (str): Path to dataset for calibration (required for int8 and int_only)
        """
        self.load_model()
        
        if self.quantization_type == "dynamic":
            quantized_model = self.apply_dynamic_quantization()
        elif self.quantization_type == "int8":
            quantized_model = self.apply_int8_quantization(data_path)
        elif self.quantization_type == "float16":
            quantized_model = self.apply_float16_quantization()
        elif self.quantization_type == "int_only":
            quantized_model = self.apply_int_only_quantization(data_path)
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")
        
        # Save the quantized model
        print(f"Saving quantized model to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'wb') as f:
            f.write(quantized_model)
        
        # Print model size comparison
        original_size = self.model_path.stat().st_size if self.model_path.is_file() else 0
        quantized_size = self.output_path.stat().st_size
        
        if original_size > 0:
            compression_ratio = (1 - quantized_size / original_size) * 100
            print("\nQuantization complete!")
            print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
            print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
            print(f"Size reduction: {compression_ratio:.1f}%")
        else:
            print("\nQuantization complete!")
            print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize TensorFlow models for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("model", help="Path to the input model (.keras, .h5, or SavedModel directory)")
    parser.add_argument("--type", "-t", choices=["dynamic", "int8", "float16", "int_only"], 
                       default="dynamic", help="Type of quantization to apply (default: dynamic)")
    parser.add_argument("--output", "-o", help="Output path for quantized model")
    parser.add_argument("--data", "-d", help="Path to dataset CSV for calibration (required for int8 and int_only)")
    parser.add_argument("--samples", "-s", type=int, default=100, 
                       help="Number of samples to use for calibration (default: 100)")
    
    args = parser.parse_args()
    
    # Validate input model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path '{args.model}' does not exist!")
        sys.exit(1)
    
    # Check if data is required but not provided
    if args.type in ["int8", "int_only"] and not args.data:
        print(f"Warning: {args.type} quantization works best with calibration data.")
        print("Consider providing --data flag with a CSV file for better results.")
        print("Proceeding with random data for calibration...\n")
    
    # Validate data path if provided
    if args.data and not Path(args.data).exists():
        print(f"Error: Data path '{args.data}' does not exist!")
        sys.exit(1)
    
    try:
        # Create quantizer and perform quantization
        quantizer = ModelQuantizer(
            model_path=args.model,
            output_path=args.output,
            quantization_type=args.type
        )
        
        quantizer.quantize(data_path=args.data)
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()