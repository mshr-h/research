# onnx-to-fx Prototype

Prototype Python library that converts ONNX models into PyTorch `torch.fx.GraphModule` objects so that existing ONNX exports can be fine-tuned or re-used inside PyTorch training workflows.

## Installation

```bash
pip install -e .
```

The project depends on recent stable releases of PyTorch, ONNX, and ONNX Runtime. The prototype has been tested with the following versions:

- PyTorch 2.2.0
- ONNX 1.15.0
- ONNX Runtime 1.16.3

## Usage

```python
import onnx
import torch
from onnx_to_fx import convert_onnx_to_fx

model = onnx.load("model.onnx")
fx_module = convert_onnx_to_fx(model)

fx_module.train()
optim = torch.optim.SGD(fx_module.parameters(), lr=1e-3)
```

You can also pass a filesystem path to `convert_onnx_to_fx` directly.

```python
fx_module = convert_onnx_to_fx("/path/to/model.onnx")
```

Set `debug=True` to print the generated FX graph to stdout for quick inspection.

## Supported Operators

The prototype currently covers a focused subset of operators that are common in feed-forward and convolutional models:

- `Gemm` (mapped to `nn.Linear`)
- `MatMul`
- `Conv`
- `Relu`
- `BatchNormalization`
- `MaxPool`
- `AveragePool`
- `GlobalAveragePool`
- `Flatten`
- `Reshape`
- `Identity`
- `Dropout` (passthrough)
- `Constant`

For operators outside of this list the converter raises `UnsupportedOperatorError` with details about the offending node. Contributions that extend the operator registry can be added incrementally in `src/onnx_to_fx/ops.py`.

## Limitations

- Only symmetric padding is currently supported for convolution and pooling layers.
- The converter assumes weights and biases are provided via ONNX initializers.
- Batch normalization is lowered to `nn.BatchNorm1d`/`nn.BatchNorm2d` based on available shape metadata.
- Advanced control-flow operators (loops, conditionals) are not supported yet.

## Development

Run the test suite from the project root with:

```bash
pytest
```

The tests exercise a simple multilayer perceptron end-to-end and validate parity against ONNX Runtime as well as error handling for unsupported operators.

## Extensibility

The conversion pipeline separates ONNX parsing and operator lowering. To add a new operator:

1. Implement a handler in `src/onnx_to_fx/ops.py` using the `@register_op` decorator.
2. Reuse helper methods from `GraphBuilder` to fetch inputs, attributes, and register modules or constants.
3. Add corresponding tests that validate graph structure and numerical correctness.
