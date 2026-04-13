# ml2plc

Tool suite for converting ONNX neural network models to IEC 61131-3 Structured Text (ST) for PLCs.

## Features

- Inspect and summarize ONNX models (weights, layers, inputs/outputs)
- Convert ONNX models to intermediate representation (IR)
- Run shape validation, constant folding, and selected Einsum lowering during conversion
- Generate Structured Text code for feedforward layers (MatMul/Gemm + activations)
- Generate Structured Text code for spatial layers (Conv/Pool/BatchNorm)
- Generate Structured Text code for recurrent layers (LSTM/GRU stateful regions)
- Generate Structured Text code for shape/data-movement and reduction ops used in practical ONNX graphs
- Validate translation by converting ST code back to Python and comparing outputs

## Usage

1. Place ONNX models in `examples/models/onnx/`
2. Run the main compiler:

   ```sh
   python src/codegen/main.py examples/models/onnx/your_model.onnx
   ```

   This generates ST code from the ONNX model.

3. Optional: Validate translation (was used to ensure correctness of the code generator)

   ```sh
   python src/translation_validation/validation.py path/to/generated.st path/to/save.py
   ```

## Installation

**Windows:**

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

**Linux/macOS:**

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Troubleshooting

- Ensure ONNX files exist and are valid
- Ensure runtime-relevant tensor shapes are fully static before conversion
- Unsupported ONNX operators fail fast with a summary of missing extractors
- For complex architectures (for example full Transformer variants), support depends on exported operator patterns and shape contracts

## Support notes

- The compiler now includes first-class support for recurrent/stateful lowering (LSTM/GRU) and common spatial operators.
- Some operators are pattern-limited (for example selected Einsum equations) to keep generated ST deterministic and PLC-friendly.
- If compilation fails, use the reported operator list to prioritize new extractor/generator additions.
