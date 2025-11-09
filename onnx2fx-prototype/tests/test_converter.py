# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the ONNX to FX converter."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch

from onnxscript import script
from onnxscript.onnx_opset import opset13 as op

from onnx_to_fx import UnsupportedOperatorError, convert_onnx_to_fx

from .common import ort_run


@script()
def _mlp_graph(input, w1, b1, w2, b2):
    hidden = op.Gemm(input, w1, b1, transB=1)
    hidden_relu = op.Relu(hidden)
    output = op.Gemm(hidden_relu, w2, b2, transB=1)
    return output


@script()
def _sin_graph(input):
    output = op.Sin(input)
    return output


def _build_mlp_model(*, input_dim: int = 4, hidden_dim: int = 3, output_dim: int = 2) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    weight1 = rng.standard_normal((hidden_dim, input_dim)).astype(np.float32)
    bias1 = rng.standard_normal((hidden_dim,)).astype(np.float32)
    weight2 = rng.standard_normal((output_dim, hidden_dim)).astype(np.float32)
    bias2 = rng.standard_normal((output_dim,)).astype(np.float32)

    model = _mlp_graph.to_model_proto()
    graph = model.graph

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_dim])
    hidden_info = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [None, hidden_dim])
    hidden_relu_info = helper.make_tensor_value_info("hidden_relu", TensorProto.FLOAT, [None, hidden_dim])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, output_dim])

    graph.input.clear()
    graph.input.extend([input_info])

    graph.output[0].CopyFrom(output_info)

    graph.value_info.clear()
    graph.value_info.extend([hidden_info, hidden_relu_info])

    graph.ClearField("initializer")
    graph.initializer.extend(
        [
            numpy_helper.from_array(weight1, name="w1"),
            numpy_helper.from_array(bias1, name="b1"),
            numpy_helper.from_array(weight2, name="w2"),
            numpy_helper.from_array(bias2, name="b2"),
        ]
    )

    for node, name in zip(graph.node, ("fc1", "relu1", "fc2")):
        node.name = name

    graph.name = "mlp"
    model.ir_version = 11
    onnx.checker.check_model(model)
    return model


def test_convert_mlp_produces_expected_graph_and_output() -> None:
    model = _build_mlp_model()
    graph_module = convert_onnx_to_fx(model)

    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    assert placeholders and placeholders[0].name == "input"
    assert placeholders[0].meta["onnx_shape"][1] == 4

    linear_modules = {name: module for name, module in graph_module.named_modules() if name}
    assert "fc1" in linear_modules and isinstance(linear_modules["fc1"], torch.nn.Linear)
    assert "fc2" in linear_modules and isinstance(linear_modules["fc2"], torch.nn.Linear)

    torch.manual_seed(0)
    sample = torch.randn(2, 4, dtype=torch.float32)
    graph_module.eval()
    with torch.no_grad():
        fx_out = graph_module(sample)

    ort_out = ort_run(model, {"input": sample.numpy()})[0]
    np.testing.assert_allclose(fx_out.detach().numpy(), ort_out, rtol=1e-5, atol=1e-5)


def test_output_names_are_preserved() -> None:
    model = _build_mlp_model()
    graph_module = convert_onnx_to_fx(model)

    final_linear = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_module" and node.target == "fc2"
    ]
    assert final_linear, "Expected to find the final linear layer in the FX graph"
    assert "output" in final_linear[0].meta.get("onnx_outputs", [])


def _build_sin_model() -> onnx.ModelProto:
    model = _sin_graph.to_model_proto()
    graph = model.graph

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    graph.input.clear()
    graph.input.extend([input_info])
    graph.output[0].CopyFrom(output_info)

    graph.node[0].name = "sin1"
    graph.name = "sin_graph"
    model.ir_version = 11
    onnx.checker.check_model(model)
    return model


def test_unsupported_operator_raises_clear_error() -> None:
    model = _build_sin_model()

    with pytest.raises(UnsupportedOperatorError):
        convert_onnx_to_fx(model)


def _build_constant_if_model(condition: bool) -> onnx.ModelProto:
    cond_tensor = helper.make_tensor("cond_value", TensorProto.BOOL, [1], [int(condition)])
    cond_node = helper.make_node("Constant", [], ["cond"], value=cond_tensor)

    true_value = helper.make_tensor("true_value", TensorProto.FLOAT, [2], [1.0, 2.0])
    false_value = helper.make_tensor("false_value", TensorProto.FLOAT, [2], [-1.0, -2.0])
    then_node = helper.make_node("Constant", [], ["then_out"], value=true_value)
    else_node = helper.make_node("Constant", [], ["else_out"], value=false_value)

    then_graph = helper.make_graph(
        [then_node],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_graph = helper.make_graph(
        [else_node],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )

    if_node = helper.make_node("If", ["cond"], ["output"], then_branch=then_graph, else_branch=else_graph)
    graph = helper.make_graph(
        [cond_node, if_node],
        "if_graph",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("condition", (True, False))
def test_if_with_constant_condition(condition: bool) -> None:
    model = _build_constant_if_model(condition)
    graph_module = convert_onnx_to_fx(model)

    with torch.no_grad():
        result = graph_module()
    expected = np.array([1.0, 2.0], dtype=np.float32) if condition else np.array([-1.0, -2.0], dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=0)
