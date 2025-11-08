"""End-to-end tests that convert full torchvision models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from torchvision import models

from onnx_to_fx import convert_onnx_to_fx


def _export_model_to_onnx(model: torch.nn.Module, sample: torch.Tensor) -> onnx.ModelProto:
    """Export a Torch model to ONNX and load it back for conversion tests."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        torch.onnx.export(
            model,
            sample,
            path.as_posix(),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
        )
        return onnx.load(path.as_posix())


MODEL_TEST_CASES = (
    pytest.param(models.resnet18, (1, 3, 224, 224), id="resnet18"),
    pytest.param(models.mobilenet_v2, (1, 3, 224, 224), id="mobilenet_v2"),
    pytest.param(models.mobilenet_v3_large, (1, 3, 224, 224), id="mobilenet_v3_large"),
    pytest.param(models.vgg11, (1, 3, 224, 224), id="vgg11"),
    pytest.param(models.alexnet, (1, 3, 224, 224), id="alexnet"),
    pytest.param(models.efficientnet_b0, (1, 3, 224, 224), id="efficientnet_b0"),
    pytest.param(models.efficientnet_v2_s, (1, 3, 224, 224), id="efficientnet_v2_s"),
    pytest.param(models.convnext_tiny, (1, 3, 224, 224), id="convnext_tiny"),
    pytest.param(models.inception_v3, (1, 3, 299, 299), id="inception_v3"),
    pytest.param(models.regnet_x_1_6gf, (1, 3, 224, 224), id="regnet_x_1_6gf"),
    pytest.param(models.regnet_y_1_6gf, (1, 3, 224, 224), id="regnet_y_1_6gf"),
    pytest.param(models.squeezenet1_0, (1, 3, 224, 224), id="squeezenet1_0"),
)


@pytest.mark.parametrize("model_factory,input_shape", MODEL_TEST_CASES)
def test_torchvision_models_back_to_back(model_factory, input_shape) -> None:
    torch.manual_seed(0)
    model = model_factory(weights=None)
    model.eval()

    sample = torch.randn(*input_shape)
    with torch.no_grad():
        expected = model(sample).detach().numpy()

    onnx_model = _export_model_to_onnx(model, sample)
    graph_module = convert_onnx_to_fx(onnx_model)
    graph_module.eval()
    with torch.no_grad():
        actual = graph_module(sample).detach().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
