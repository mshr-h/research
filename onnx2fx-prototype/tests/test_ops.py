# SPDX-License-Identifier: Apache-2.0
"""Unit tests that cover each supported ONNX operator in isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch

from onnx_to_fx import convert_onnx_to_fx

from .common import ort_run


def _make_model(
    *,
    nodes: List[onnx.NodeProto],
    inputs: List[onnx.ValueInfoProto],
    outputs: List[onnx.ValueInfoProto],
    initializers: List[onnx.TensorProto] | None = None,
    opset: int = 18,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "single_op_graph",
        inputs,
        outputs,
        initializer=initializers or [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version or 7, 11)
    onnx.checker.check_model(model)
    return model


def _float_tensor(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array.astype(np.float32), name=name)


def _input(name: str, shape: Tuple[int, ...]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _output(name: str, shape: Tuple[int, ...]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _rand(shape: Tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


@dataclass
class OpTestCase:
    name: str
    builder: Callable[[], Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]]


def _constant_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    value = _float_tensor("const_tensor", np.array([1.0, -3.0], dtype=np.float32))
    const = helper.make_node("Constant", [], ["output"], value=value)
    model = _make_model(nodes=[const], inputs=[], outputs=[_output("output", (2,))])
    return model, [], {}


def _sequence_empty_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    seq = helper.make_node("SequenceEmpty", [], ["seq"])
    value = _float_tensor("const_tensor", np.array([2.0, 5.0], dtype=np.float32))
    const = helper.make_node("Constant", [], ["output"], value=value)
    model = _make_model(nodes=[seq, const], inputs=[], outputs=[_output("output", (2,))])
    return model, [], {}


def _unary_case(op_type: str) -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3)
    node = helper.make_node(op_type, ["input"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
    )
    feeds = {"input": _rand(input_shape, seed=1)}
    return model, ["input"], feeds


def _hard_sigmoid_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (4, 4)
    node = helper.make_node("HardSigmoid", ["input"], ["output"], alpha=0.3, beta=0.4)
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
        opset=14,
    )
    feeds = {"input": _rand(input_shape, seed=2)}
    return model, ["input"], feeds


def _hard_swish_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (4, 4)
    node = helper.make_node("HardSwish", ["input"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
        opset=14,
    )
    feeds = {"input": _rand(input_shape, seed=3)}
    return model, ["input"], feeds


def _transpose_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3, 4)
    node = helper.make_node("Transpose", ["input"], ["output"], perm=[0, 2, 1])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (2, 4, 3))],
    )
    feeds = {"input": _rand(input_shape, seed=25)}
    return model, ["input"], feeds


def _binary_case(op_type: str) -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3)
    node = helper.make_node(op_type, ["lhs", "rhs"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("lhs", input_shape), _input("rhs", input_shape)],
        outputs=[_output("output", input_shape)],
    )
    feeds = {
        "lhs": _rand(input_shape, seed=4),
        "rhs": _rand(input_shape, seed=5),
    }
    return model, ["lhs", "rhs"], feeds


def _matmul_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    a_shape = (2, 3)
    b_shape = (3, 4)
    node = helper.make_node("MatMul", ["lhs", "rhs"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("lhs", a_shape), _input("rhs", b_shape)],
        outputs=[_output("output", (2, 4))],
    )
    feeds = {
        "lhs": _rand(a_shape, seed=6),
        "rhs": _rand(b_shape, seed=7),
    }
    return model, ["lhs", "rhs"], feeds


def _gemm_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3)
    weight = _float_tensor("w", _rand((4, 3), seed=8))
    bias = _float_tensor("b", _rand((4,), seed=9))
    node = helper.make_node("Gemm", ["input", "w", "b"], ["output"], transB=1)
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (2, 4))],
        initializers=[weight, bias],
    )
    feeds = {"input": _rand(input_shape, seed=10)}
    return model, ["input"], feeds


def _clip_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3)
    min_tensor = numpy_helper.from_array(np.array(-0.5, dtype=np.float32), name="min")
    max_tensor = numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="max")
    node = helper.make_node("Clip", ["input", "min", "max"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
        initializers=[min_tensor, max_tensor],
    )
    feeds = {"input": _rand(input_shape, seed=11)}
    return model, ["input"], feeds


def _conv_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (1, 3, 5, 5)
    weight = _float_tensor("w", _rand((4, 3, 3, 3), seed=12))
    bias = _float_tensor("b", _rand((4,), seed=13))
    node = helper.make_node("Conv", ["input", "w", "b"], ["output"], strides=[1, 1], pads=[1, 1, 1, 1])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (1, 4, 5, 5))],
        initializers=[weight, bias],
    )
    feeds = {"input": _rand(input_shape, seed=14)}
    return model, ["input"], feeds


def _batch_norm_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (1, 3, 4, 4)
    scale = _float_tensor("scale", _rand((3,), seed=15))
    bias = _float_tensor("bias", _rand((3,), seed=16))
    mean = _float_tensor("mean", np.zeros((3,), dtype=np.float32))
    var = _float_tensor("var", np.ones((3,), dtype=np.float32))
    node = helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean", "var"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
        initializers=[scale, bias, mean, var],
    )
    feeds = {"input": _rand(input_shape, seed=17)}
    return model, ["input"], feeds


def _pool_case(op_type: str) -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (1, 3, 4, 4)
    attrs = {"kernel_shape": [2, 2], "strides": [2, 2]}
    node = helper.make_node(op_type, ["input"], ["output"], **attrs)
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (1, 3, 2, 2))],
    )
    feeds = {"input": _rand(input_shape, seed=18)}
    return model, ["input"], feeds


def _global_avg_pool_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (1, 2, 3, 3)
    node = helper.make_node("GlobalAveragePool", ["input"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (1, 2, 1, 1))],
    )
    feeds = {"input": _rand(input_shape, seed=19)}
    return model, ["input"], feeds


def _reduce_mean_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3, 4)
    node = helper.make_node("ReduceMean", ["input"], ["output"], axes=[2], keepdims=1)
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (2, 3, 1))],
        opset=13,
    )
    feeds = {"input": _rand(input_shape, seed=20)}
    return model, ["input"], feeds


def _gather_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    data_shape = (4, 2)
    indices = numpy_helper.from_array(np.array([0, 2], dtype=np.int64), name="indices")
    node = helper.make_node("Gather", ["data", "indices"], ["output"], axis=0)
    model = _make_model(
        nodes=[node],
        inputs=[_input("data", data_shape)],
        outputs=[_output("output", (2, 2))],
        initializers=[indices],
    )
    feeds = {"data": _rand(data_shape, seed=26)}
    return model, ["data"], feeds


def _layer_norm_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 4)
    scale = _float_tensor("scale", np.ones((4,), dtype=np.float32))
    bias = _float_tensor("bias", np.zeros((4,), dtype=np.float32))
    node = helper.make_node("LayerNormalization", ["input", "scale", "bias"], ["output"], epsilon=1e-5)
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
        initializers=[scale, bias],
    )
    feeds = {"input": _rand(input_shape, seed=27)}
    return model, ["input"], feeds


def _cast_like_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 2)
    reference = numpy_helper.from_array(np.array([1, 0], dtype=np.int32), name="ref")
    node = helper.make_node("CastLike", ["input", "ref"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT32, list(input_shape))],
        initializers=[reference],
    )
    feeds = {"input": _rand(input_shape, seed=28)}
    return model, ["input"], feeds


def _flatten_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3, 4)
    node = helper.make_node("Flatten", ["input"], ["output"], axis=1)
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (2, 12))],
    )
    feeds = {"input": _rand(input_shape, seed=21)}
    return model, ["input"], feeds


def _reshape_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 2, 2)
    shape_tensor = numpy_helper.from_array(np.array([2, 4], dtype=np.int64), name="shape")
    node = helper.make_node("Reshape", ["input", "shape"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", (2, 4))],
        initializers=[shape_tensor],
    )
    feeds = {"input": _rand(input_shape, seed=22)}
    return model, ["input"], feeds


def _identity_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (3, 3)
    node = helper.make_node("Identity", ["input"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
    )
    feeds = {"input": _rand(input_shape, seed=23)}
    return model, ["input"], feeds


def _dropout_case() -> Tuple[onnx.ModelProto, List[str], Dict[str, np.ndarray]]:
    input_shape = (2, 3)
    node = helper.make_node("Dropout", ["input"], ["output"])
    model = _make_model(
        nodes=[node],
        inputs=[_input("input", input_shape)],
        outputs=[_output("output", input_shape)],
    )
    feeds = {"input": _rand(input_shape, seed=24)}
    return model, ["input"], feeds


OP_TEST_CASES = [
    OpTestCase("constant", _constant_case),
    OpTestCase("sequence_empty", _sequence_empty_case),
    OpTestCase("relu", lambda: _unary_case("Relu")),
    OpTestCase("sigmoid", lambda: _unary_case("Sigmoid")),
    OpTestCase("erf", lambda: _unary_case("Erf")),
    OpTestCase("hard_sigmoid", _hard_sigmoid_case),
    OpTestCase("hard_swish", _hard_swish_case),
    OpTestCase("transpose", _transpose_case),
    OpTestCase("add", lambda: _binary_case("Add")),
    OpTestCase("div", lambda: _binary_case("Div")),
    OpTestCase("mul", lambda: _binary_case("Mul")),
    OpTestCase("matmul", _matmul_case),
    OpTestCase("gemm", _gemm_case),
    OpTestCase("clip", _clip_case),
    OpTestCase("conv", _conv_case),
    OpTestCase("batch_norm", _batch_norm_case),
    OpTestCase("layer_norm", _layer_norm_case),
    OpTestCase("max_pool", lambda: _pool_case("MaxPool")),
    OpTestCase("avg_pool", lambda: _pool_case("AveragePool")),
    OpTestCase("global_average_pool", _global_avg_pool_case),
    OpTestCase("gather", _gather_case),
    OpTestCase("reduce_mean", _reduce_mean_case),
    OpTestCase("flatten", _flatten_case),
    OpTestCase("reshape", _reshape_case),
    OpTestCase("identity", _identity_case),
    OpTestCase("cast_like", _cast_like_case),
    OpTestCase("dropout", _dropout_case),
]


@pytest.mark.parametrize("case", OP_TEST_CASES, ids=lambda case: case.name)
def test_supported_ops(case: OpTestCase) -> None:
    model, input_order, feeds = case.builder()
    graph_module = convert_onnx_to_fx(model)
    graph_module.eval()

    torch_inputs = [torch.from_numpy(feeds[name]) for name in input_order]
    with torch.no_grad():
        fx_out = graph_module(*torch_inputs) if torch_inputs else graph_module()

    ort_out = ort_run(model, feeds)[0]
    np.testing.assert_allclose(fx_out.detach().numpy(), ort_out, rtol=1e-5, atol=1e-5)
