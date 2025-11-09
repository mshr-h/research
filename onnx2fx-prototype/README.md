# onnx-to-fx Prototype

Prototype Python library that converts ONNX models into PyTorch `torch.fx.GraphModule`
objects so exported networks can participate in PyTorch fine-tuning, quantization, or
analysis workflows without going back to the original source graph.

The converter powers several integration tests that round-trip torchvision architectures
(ResNet, MobileNet V2/V3, EfficientNet, EfficientNet V2, ConvNeXt-Tiny, RegNet X/Y,
SqueezeNet, VGG, AlexNet, Inception V3) and it comes with per-operator unit tests that
compare against ONNX Runtime output for every supported op.

## Installation

Requirements:

- Python 3.10+
- PyTorch ≥ 2.2
- ONNX ≥ 1.15
- ONNX Runtime ≥ 1.16

Install in editable mode (recommended for development):

```bash
git clone https://github.com/example/onnx-to-fx.git
cd onnx-to-fx
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

The `dev` extras install pytest, torchvision, and other tooling that the test suite makes use of.

## Usage

### Converting from an ONNX file

```python
import torch
from onnx_to_fx import convert_onnx_to_fx

fx_module = convert_onnx_to_fx("model.onnx")
fx_module.eval()
with torch.no_grad():
    y = fx_module(torch.randn(1, 3, 224, 224))
```

### Converting from an in-memory `onnx.ModelProto`

```python
import onnx
from onnx_to_fx import convert_onnx_to_fx

proto = onnx.load("model.onnx")
fx_module = convert_onnx_to_fx(proto, debug=True)  # prints the FX graph
```

### Converting from `torch.onnx.ONNXProgram`

PyTorch’s exporter can return an `ONNXProgram` instead of writing directly to disk. The converter
accepts this object directly and unwraps it into a `ModelProto` internally:

```python
from torch.onnx import ONNXProgram
from onnx_to_fx import convert_onnx_to_fx

# assume `program` was produced by torch.onnx
fx_module = convert_onnx_to_fx(program.model_proto)
```

### Running the converted module

`convert_onnx_to_fx` returns a regular `torch.fx.GraphModule`. You can use it just like any other
nn.Module: attach optimizers, run `forward` in training or eval mode, script it, etc. Initializers
become parameters or buffers on the returned module so you can fine-tune weights out of the box.

## Supported Operators

The following operators have dedicated lowering rules (and unit tests):

- Arithmetic: `Add`, `Mul`, `Div`, `MatMul`, `Gemm`, `Clip`, `Sigmoid`, `HardSigmoid`, `HardSwish`, `Erf`
- Tensor shaping: `Transpose`, `Reshape`, `Flatten`, `Concat`, `Cast`, `CastLike`, `Identity`
- Neural nets: `Conv`, `BatchNormalization`, `LayerNormalization`, `Relu`, `Dropout` (no-op)
- Pooling: `MaxPool`, `AveragePool`, `GlobalAveragePool`
- Reductions: `ReduceMean`
- Sequences & control flow: `SequenceEmpty`, `If`, `Loop`
- Infrastructure: `Constant`, `Gather`

If the converter encounters any other operator it raises `UnsupportedOperatorError` with a helpful message.
Adding support is straightforward: implement a handler in `src/onnx_to_fx/ops.py`, register it with the
`@register_op` decorator, and add a matching test in `tests/test_ops.py`.

## Supported Models

The integration suite currently round-trips these torchvision networks end-to-end:

- ResNet18
- MobileNet V2 and V3 Large
- EfficientNet-B0, EfficientNet-V2-S
- ConvNeXt Tiny
- RegNet X 1.6 GF, RegNet Y 1.6 GF
- VGG11, AlexNet, SqueezeNet 1.0
- Inception V3

Adding additional models is simply a matter of appending to `MODEL_TEST_CASES` in `tests/test_e2e_models.py`.

## Limitations

- Only symmetric padding is supported for convolution/pooling; asymmetric padding raises `UnsupportedOperatorError`.
- We require ONNX initializers for weights and biases. Dynamic parameter tensors are not yet handled.
- Loop support is limited to cases where the loop body operates on compile-time-known values (as in ConvNeXt’s adaptive pooling).
- Control-flow evaluation currently folds outputs into constants; propagating loop-carried tensors back into FX nodes is still on the roadmap.
- Some ONNX opsets may emit attributes or data types that are not yet covered—please open an issue if you hit one.

## Development Workflow

1. Install dependencies with `pip install -e .[dev]`.
2. Run `pytest` to execute both the per-operator and end-to-end suites.
3. When adding an operator, create a new case in `tests/test_ops.py` so numerical equivalence is checked against ONNX Runtime.
4. Keep the README’s operator table in sync as capabilities grow.

## License

Apache-2.0

The entire repository is released under the Apache License, Version 2.0. See `LICENSE` for the full text.
