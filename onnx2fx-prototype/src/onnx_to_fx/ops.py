"""Operator registry that maps ONNX ops to FX lowering handlers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable, Dict, Sequence

import onnx
from onnx import helper
import torch
from torch import fx, nn
from torch.nn import functional as F

from .errors import ConversionError, UnsupportedOperatorError
from .utils import onnx_dtype_to_torch, tensor_proto_to_torch

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from .graph_builder import GraphBuilder


OpHandler = Callable[["GraphBuilder", onnx.NodeProto], object]
OP_REGISTRY: Dict[str, OpHandler] = {}


def register_op(op_type: str) -> Callable[[OpHandler], OpHandler]:
    """Decorator used to register ONNX operator handlers."""

    def decorator(func: OpHandler) -> OpHandler:
        OP_REGISTRY[op_type] = func
        return func

    return decorator


def _cast_like_tensor(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=reference.dtype)


def _gather_axis0(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = torch.index_select(data, 0, indices.reshape(-1).long())
    tail_shape = data.shape[1:]
    return flat.reshape(tuple(indices.shape) + tail_shape)


class StaticGraphEvaluator:
    """Evaluates small ONNX graphs made entirely of statically known values."""

    def __init__(self, builder: "GraphBuilder") -> None:
        self.builder = builder

    def evaluate(
        self,
        graph: onnx.GraphProto,
        *,
        tensor_env: Dict[str, torch.Tensor] | None = None,
        sequence_env: Dict[str, list] | None = None,
    ) -> list[torch.Tensor | list]:
        tensors = {
            name: value.clone().detach()
            for name, value in (tensor_env.items() if tensor_env else [])
        }
        sequences = {
            name: list(value)
            for name, value in (sequence_env.items() if sequence_env else [])
        }

        for node in graph.node:
            self._execute_node(node, tensors, sequences)

        outputs: list[torch.Tensor | list] = []
        for value in graph.output:
            name = value.name
            if name in tensors:
                outputs.append(tensors[name])
            elif name in sequences:
                outputs.append(sequences[name])
            elif name in self.builder.tensor_values:
                outputs.append(self.builder.tensor_values[name])
            elif name in self.builder.sequence_values:
                outputs.append(self.builder.sequence_values[name])
            else:
                raise ConversionError(f"Static evaluator could not resolve '{name}'")
        return outputs

    # ------------------------------------------------------------------
    def _get_tensor(
        self,
        name: str,
        tensors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not name:
            raise ConversionError("Empty tensor name in static evaluation")
        if name in tensors:
            return tensors[name]
        if name in self.builder.tensor_values:
            return self.builder.tensor_values[name].clone().detach()
        if name in self.builder.initializers:
            return self.builder.initializers[name].clone().detach()
        raise ConversionError(f"Tensor '{name}' not available in static evaluation")

    def _get_sequence(
        self,
        name: str,
        sequences: Dict[str, list],
    ) -> list:
        if name in sequences:
            return sequences[name]
        if name in self.builder.sequence_values:
            return list(self.builder.sequence_values[name])
        raise ConversionError(f"Sequence '{name}' not available in static evaluation")

    def _set_tensor(
        self,
        name: str,
        value: torch.Tensor,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        tensors[name] = value.clone().detach()

    def _set_sequence(
        self,
        name: str,
        value: list,
        sequences: Dict[str, list],
    ) -> None:
        sequences[name] = list(value)

    def _tensor_to_list(self, tensor: torch.Tensor) -> list[int]:
        return [int(v) for v in tensor.reshape(-1).tolist()]

    def _execute_node(
        self,
        node: onnx.NodeProto,
        tensors: Dict[str, torch.Tensor],
        sequences: Dict[str, list],
    ) -> None:
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        op = node.op_type

        if op == "Identity":
            name = node.input[0]
            if name in tensors or name in self.builder.tensor_values or name in self.builder.initializers:
                self._set_tensor(node.output[0], self._get_tensor(name, tensors), tensors)
            else:
                self._set_sequence(node.output[0], self._get_sequence(name, sequences), sequences)
        elif op == "Constant":
            value = attrs.get("value")
            if not isinstance(value, onnx.TensorProto):
                raise ConversionError("Static evaluator only supports tensor constants")
            tensor = tensor_proto_to_torch(value)
            self._set_tensor(node.output[0], tensor, tensors)
        elif op == "Sub":
            a = self._get_tensor(node.input[0], tensors)
            b = self._get_tensor(node.input[1], tensors)
            self._set_tensor(node.output[0], torch.sub(a, b), tensors)
        elif op == "Add":
            a = self._get_tensor(node.input[0], tensors)
            b = self._get_tensor(node.input[1], tensors)
            self._set_tensor(node.output[0], torch.add(a, b), tensors)
        elif op == "Mul":
            a = self._get_tensor(node.input[0], tensors)
            b = self._get_tensor(node.input[1], tensors)
            self._set_tensor(node.output[0], torch.mul(a, b), tensors)
        elif op == "Reshape":
            data = self._get_tensor(node.input[0], tensors)
            shape_tensor = self._get_tensor(node.input[1], tensors)
            shape = tuple(int(dim) for dim in shape_tensor.view(-1).tolist())
            self._set_tensor(node.output[0], torch.reshape(data, shape), tensors)
        elif op == "Gather":
            data = self._get_tensor(node.input[0], tensors)
            indices = self._get_tensor(node.input[1], tensors).long()
            axis = int(attrs.get("axis", 0))
            if axis not in (0, -data.dim()):
                raise ConversionError("Static evaluator only supports axis=0 for Gather")
            self._set_tensor(node.output[0], _gather_axis0(data, indices), tensors)
        elif op == "Slice":
            data = self._get_tensor(node.input[0], tensors)
            starts = self._tensor_to_list(self._get_tensor(node.input[1], tensors))
            ends = self._tensor_to_list(self._get_tensor(node.input[2], tensors))
            axes = (
                self._tensor_to_list(self._get_tensor(node.input[3], tensors))
                if len(node.input) >= 4 and node.input[3]
                else list(range(len(starts)))
            )
            steps = (
                self._tensor_to_list(self._get_tensor(node.input[4], tensors))
                if len(node.input) >= 5 and node.input[4]
                else [1] * len(starts)
            )
            slices = [slice(None)] * data.dim()
            for start, end, axis, step in zip(starts, ends, axes, steps):
                final_end = None if end >= 9223372036854775800 else end
                slices[axis] = slice(start, final_end, step)
            self._set_tensor(node.output[0], data[tuple(slices)], tensors)
        elif op == "Expand":
            data = self._get_tensor(node.input[0], tensors)
            target_shape = tuple(int(dim) for dim in self._get_tensor(node.input[1], tensors).view(-1).tolist())
            expanded = data
            while expanded.dim() < len(target_shape):
                expanded = expanded.unsqueeze(0)
            self._set_tensor(node.output[0], expanded.expand(target_shape), tensors)
        elif op == "Range":
            start = self._get_tensor(node.input[0], tensors).item()
            limit = self._get_tensor(node.input[1], tensors).item()
            delta = self._get_tensor(node.input[2], tensors).item()
            result = torch.arange(float(start), float(limit), float(delta), dtype=torch.float32)
            self._set_tensor(node.output[0], result, tensors)
        elif op == "Equal":
            a = self._get_tensor(node.input[0], tensors)
            b = self._get_tensor(node.input[1], tensors)
            self._set_tensor(node.output[0], torch.eq(a, b), tensors)
        elif op == "Concat":
            axis = int(attrs.get("axis", 0))
            to_cat = [self._get_tensor(name, tensors) for name in node.input if name]
            self._set_tensor(node.output[0], torch.cat(to_cat, dim=axis), tensors)
        elif op == "ConcatFromSequence":
            seq = self._get_sequence(node.input[0], sequences)
            axis = int(attrs.get("axis", 0))
            new_axis = int(attrs.get("new_axis", 0))
            tensors_to_cat = [t.clone() for t in seq]
            if new_axis:
                result = torch.stack(tensors_to_cat, dim=axis)
            else:
                result = torch.cat(tensors_to_cat, dim=axis)
            self._set_tensor(node.output[0], result, tensors)
        elif op == "Cast":
            tensor = self._get_tensor(node.input[0], tensors)
            to = int(attrs["to"])
            dtype = onnx_dtype_to_torch(to)
            if dtype is None:
                raise ConversionError("Static evaluator Cast missing target dtype")
            self._set_tensor(node.output[0], tensor.to(dtype), tensors)
        elif op == "If":
            cond = self._get_tensor(node.input[0], tensors)
            cond_value = bool(cond.reshape(-1)[0].item())
            branch_name = "then_branch" if cond_value else "else_branch"
            branch_graph = next((attr.g for attr in node.attribute if attr.name == branch_name), None)
            if branch_graph is None:
                raise ConversionError("Static evaluator If missing branch graph")
            branch_outputs = self.evaluate(
                branch_graph,
                tensor_env={k: v.clone().detach() for k, v in tensors.items()},
                sequence_env={k: list(v) for k, v in sequences.items()},
            )
            for name, value in zip(node.output, branch_outputs):
                if isinstance(value, torch.Tensor):
                    self._set_tensor(name, value, tensors)
                else:
                    self._set_sequence(name, value, sequences)
        elif op == "SequenceInsert":
            seq = self._get_sequence(node.input[0], sequences)
            tensor = self._get_tensor(node.input[1], tensors)
            axis = int(attrs.get("axis", len(seq)))
            seq_copy = list(seq)
            if axis < 0:
                axis += len(seq_copy) + 1
            axis = max(0, min(axis, len(seq_copy)))
            seq_copy.insert(axis, tensor)
            self._set_sequence(node.output[0], seq_copy, sequences)
        else:
            raise UnsupportedOperatorError(f"Static evaluator does not support '{op}'")


@register_op("Constant")
def constant(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    tensor: torch.Tensor
    if "value" in attrs:
        value = attrs["value"]
        if not isinstance(value, onnx.TensorProto):
            raise ConversionError("'value' attribute of Constant node is not a TensorProto")
        tensor = tensor_proto_to_torch(value)
    elif "value_float" in attrs:
        tensor = torch.tensor(attrs["value_float"], dtype=torch.float32)
    elif "value_floats" in attrs:
        tensor = torch.tensor(attrs["value_floats"], dtype=torch.float32)
    elif "value_int" in attrs:
        tensor = torch.tensor(attrs["value_int"], dtype=torch.int64)
    elif "value_ints" in attrs:
        tensor = torch.tensor(attrs["value_ints"], dtype=torch.int64)
    else:
        raise ConversionError("Constant node is missing a supported value attribute")
    name_hint = node.output[0] if node.output else node.name
    return builder.create_constant_attr(node, tensor, name_hint=name_hint, trainable=False)


@register_op("SequenceEmpty")
def sequence_empty(builder: "GraphBuilder", node: onnx.NodeProto):
    outputs = [name for name in node.output if name]
    for name in outputs:
        builder.set_sequence_value(name, [])
    return None


@register_op("Relu")
def relu(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    x = builder.get_value(node.input[0])
    return builder.call_function(node, torch.relu, (x,))


@register_op("Add")
def add(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(node, operator.add, (lhs, rhs))


@register_op("Div")
def div(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(node, operator.truediv, (lhs, rhs))


@register_op("Mul")
def mul(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(node, operator.mul, (lhs, rhs))


@register_op("Sigmoid")
def sigmoid(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    x = builder.get_value(node.input[0])
    return builder.call_function(node, torch.sigmoid, (x,))


@register_op("MatMul")
def matmul(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    lhs = builder.get_value(node.input[0])
    rhs = builder.get_value(node.input[1])
    return builder.call_function(node, torch.matmul, (lhs, rhs))


@register_op("Transpose")
def transpose(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    x = builder.get_value(node.input[0])
    attrs = builder.get_attributes(node)
    perm = attrs.get("perm")
    if perm is None:
        info = builder.get_value_info(node.input[0])
        if info is None or not info.shape:
            raise UnsupportedOperatorError("Transpose requires 'perm' when rank is unknown")
        perm = list(range(len(info.shape) - 1, -1, -1))
    perm_tuple = tuple(int(p) for p in perm)
    return builder.call_function(node, torch.permute, (x, perm_tuple))


@register_op("Gemm")
def gemm(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))

    if alpha != 1.0 or beta not in (0.0, 1.0) or trans_a != 0:
        raise UnsupportedOperatorError(
            "Gemm conversion currently supports alpha=1.0, beta in {0,1}, and transA=0"
        )

    input_tensor = builder.get_value(node.input[0])
    weight_name = node.input[1]
    weight_tensor = builder.get_initializer_tensor(weight_name)
    if trans_b == 0:
        weight_tensor = weight_tensor.t()

    bias_tensor = None
    if len(node.input) >= 3 and node.input[2]:
        bias_tensor = builder.get_initializer_tensor(node.input[2])
        if beta == 0.0:
            bias_tensor = torch.zeros_like(bias_tensor)

    out_features, in_features = weight_tensor.shape
    linear = nn.Linear(in_features, out_features, bias=bias_tensor is not None)
    linear.weight.data.copy_(weight_tensor)
    if bias_tensor is not None:
        linear.bias.data.copy_(bias_tensor.view_as(linear.bias))

    module_name = builder.register_module(node, linear, base_name=node.name or "linear")
    return builder.call_module(node, module_name, (input_tensor,))


def _hard_sigmoid_impl(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return torch.clamp(x * alpha + beta, min=0.0, max=1.0)


def _hard_swish_impl(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return x * torch.clamp(alpha * x + beta, min=0.0, max=1.0)


@register_op("Clip")
def clip(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    x = builder.get_value(node.input[0])
    attrs = builder.get_attributes(node)
    min_value = attrs.get("min")
    max_value = attrs.get("max")

    if len(node.input) >= 2 and node.input[1]:
        min_tensor = builder.get_tensor_value(node.input[1])
        if min_tensor.numel() != 1:
            raise UnsupportedOperatorError("Clip currently requires scalar min inputs")
        min_value = float(min_tensor.item())
    if len(node.input) >= 3 and node.input[2]:
        max_tensor = builder.get_tensor_value(node.input[2])
        if max_tensor.numel() != 1:
            raise UnsupportedOperatorError("Clip currently requires scalar max inputs")
        max_value = float(max_tensor.item())

    kwargs = {}
    if min_value is not None:
        kwargs["min"] = float(min_value)
    if max_value is not None:
        kwargs["max"] = float(max_value)
    return builder.call_function(node, torch.clamp, (x,), kwargs=kwargs)


@register_op("Erf")
def erf(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    x = builder.get_value(node.input[0])
    return builder.call_function(node, torch.erf, (x,))


@register_op("HardSigmoid")
def hard_sigmoid(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    alpha = float(attrs.get("alpha", 0.2))
    beta = float(attrs.get("beta", 0.5))
    x = builder.get_value(node.input[0])
    return builder.call_function(node, _hard_sigmoid_impl, (x, alpha, beta))


@register_op("HardSwish")
def hard_swish(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    alpha = float(attrs.get("alpha", 1.0 / 6.0))
    beta = float(attrs.get("beta", 0.5))
    x = builder.get_value(node.input[0])
    return builder.call_function(node, _hard_swish_impl, (x, alpha, beta))


@register_op("Concat")
def concat(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    axis = int(attrs.get("axis", 0))
    tensors = [builder.get_value(name) for name in node.input if name]
    if not tensors:
        raise ConversionError("Concat node has no inputs")
    return builder.call_function(node, torch.cat, (tuple(tensors),), kwargs={"dim": axis})


def _conv_padding(attrs: Dict[str, object]) -> Sequence[int] | int:
    pads = attrs.get("pads")
    auto_pad = attrs.get("auto_pad")
    if auto_pad and auto_pad != b"NOTSET" and auto_pad != "NOTSET":
        raise UnsupportedOperatorError("auto_pad modes other than NOTSET are not supported")
    if pads is None:
        return 0
    pads_list = list(pads)
    if len(pads_list) == 2:
        return tuple(int(p) for p in pads_list)
    if len(pads_list) == 4:
        top, left, bottom, right = pads_list
        if top != bottom or left != right:
            raise UnsupportedOperatorError("Asymmetric padding is not supported for Conv ops")
        return (int(top), int(left))
    raise ConversionError("Unsupported pads configuration for Conv op")


@register_op("Conv")
def conv(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    weight = builder.get_initializer_tensor(node.input[1])
    bias = None
    if len(node.input) >= 3 and node.input[2]:
        bias = builder.get_initializer_tensor(node.input[2])

    strides = tuple(int(s) for s in attrs.get("strides", [1, 1]))
    dilations = tuple(int(d) for d in attrs.get("dilations", [1, 1]))
    groups = int(attrs.get("group", 1))
    padding = _conv_padding(attrs)

    out_channels = int(weight.shape[0])
    kernel_size = tuple(int(k) for k in weight.shape[2:])
    in_channels = int(weight.shape[1] * groups)

    conv_module = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=strides,
        padding=padding,
        dilation=dilations,
        groups=groups,
        bias=bias is not None,
    )
    conv_module.weight.data.copy_(weight)
    if bias is not None:
        conv_module.bias.data.copy_(bias.view_as(conv_module.bias))

    module_name = builder.register_module(node, conv_module, base_name=node.name or "conv")
    input_tensor = builder.get_value(node.input[0])
    return builder.call_module(node, module_name, (input_tensor,))


@register_op("BatchNormalization")
def batch_norm(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    epsilon = float(attrs.get("epsilon", 1e-5))
    momentum = float(attrs.get("momentum", 0.9))

    scale = builder.get_initializer_tensor(node.input[1])
    bias = builder.get_initializer_tensor(node.input[2])
    running_mean = builder.get_initializer_tensor(node.input[3])
    running_var = builder.get_initializer_tensor(node.input[4])

    num_features = int(scale.numel())
    input_info = builder.get_value_info(node.input[0])
    rank = len(input_info.shape) if input_info and input_info.shape else None

    if rank is not None and rank <= 2:
        bn_module = nn.BatchNorm1d(num_features, eps=epsilon, momentum=momentum)
    else:
        bn_module = nn.BatchNorm2d(num_features, eps=epsilon, momentum=momentum)

    bn_module.weight.data.copy_(scale.view_as(bn_module.weight))
    bn_module.bias.data.copy_(bias.view_as(bn_module.bias))
    bn_module.running_mean.data.copy_(running_mean.view_as(bn_module.running_mean))
    bn_module.running_var.data.copy_(running_var.view_as(bn_module.running_var))

    module_name = builder.register_module(node, bn_module, base_name=node.name or "batchnorm")
    input_tensor = builder.get_value(node.input[0])
    return builder.call_module(node, module_name, (input_tensor,))


@register_op("LayerNormalization")
def layer_normalization(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    epsilon = float(attrs.get("epsilon", 1e-5))

    x = builder.get_value(node.input[0])
    scale = builder.get_initializer_tensor(node.input[1])
    bias = builder.get_initializer_tensor(node.input[2])

    normalized_shape = tuple(int(dim) for dim in scale.shape)
    layer_norm = nn.LayerNorm(normalized_shape, eps=epsilon, elementwise_affine=True)
    layer_norm.weight.data.copy_(scale.view_as(layer_norm.weight))
    layer_norm.bias.data.copy_(bias.view_as(layer_norm.bias))

    module_name = builder.register_module(node, layer_norm, base_name=node.name or "layernorm")
    return builder.call_module(node, module_name, (x,))


@register_op("Gather")
def gather(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    axis = int(attrs.get("axis", 0))
    if axis not in (0,):
        raise UnsupportedOperatorError("Gather currently supports axis=0")
    data = builder.get_value(node.input[0])
    indices = builder.get_value(node.input[1])
    return builder.call_function(node, _gather_axis0, (data, indices))


@register_op("CastLike")
def cast_like(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    source = builder.get_value(node.input[0])
    reference = builder.get_value(node.input[1])
    return builder.call_function(node, _cast_like_tensor, (source, reference))


def _pool_padding(attrs: Dict[str, object]) -> Sequence[int] | int:
    pads = attrs.get("pads")
    if pads is None:
        return 0
    pads_list = list(pads)
    if len(pads_list) == 2:
        return tuple(int(p) for p in pads_list)
    if len(pads_list) == 4:
        top, left, bottom, right = pads_list
        if top != bottom or left != right:
            raise UnsupportedOperatorError("Asymmetric padding is not supported for pooling ops")
        return (int(top), int(left))
    raise ConversionError("Unsupported pads configuration for pooling op")


@register_op("MaxPool")
def max_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    kernel_shape = tuple(int(k) for k in attrs.get("kernel_shape", []))
    if not kernel_shape:
        raise ConversionError("MaxPool node missing kernel_shape attribute")
    stride = tuple(int(s) for s in attrs.get("strides", kernel_shape))
    dilation = tuple(int(d) for d in attrs.get("dilations", [1, 1]))
    ceil_mode = bool(attrs.get("ceil_mode", 0))
    padding = _pool_padding(attrs)
    x = builder.get_value(node.input[0])
    return builder.call_function(
        node,
        F.max_pool2d,
        (x,),
        kwargs={
            "kernel_size": kernel_shape,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
        },
    )


@register_op("AveragePool")
def average_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    kernel_shape = tuple(int(k) for k in attrs.get("kernel_shape", []))
    if not kernel_shape:
        raise ConversionError("AveragePool node missing kernel_shape attribute")
    stride = tuple(int(s) for s in attrs.get("strides", kernel_shape))
    padding = _pool_padding(attrs)
    count_include_pad = bool(attrs.get("count_include_pad", 0))
    x = builder.get_value(node.input[0])
    return builder.call_function(
        node,
        F.avg_pool2d,
        (x,),
        kwargs={
            "kernel_size": kernel_shape,
            "stride": stride,
            "padding": padding,
            "count_include_pad": count_include_pad,
        },
    )


@register_op("GlobalAveragePool")
def global_average_pool(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    x = builder.get_value(node.input[0])
    return builder.call_function(node, F.adaptive_avg_pool2d, (x, (1, 1)))


@register_op("ReduceMean")
def reduce_mean(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    keepdims = bool(attrs.get("keepdims", 1))
    noop_with_empty_axes = bool(
        attrs.get("noop_with_empty_axes", attrs.get("noop_with_dim", 0))
    )

    axes = attrs.get("axes")
    if axes is not None:
        axes = tuple(int(axis) for axis in axes)
    elif len(node.input) >= 2 and node.input[1]:
        axes_tensor = builder.get_tensor_value(node.input[1])
        axes_list = [int(a) for a in axes_tensor.view(-1).tolist()]
        axes = tuple(axes_list)
    else:
        axes = None

    if (axes is None or len(axes) == 0) and noop_with_empty_axes:
        return builder.get_value(node.input[0])

    x = builder.get_value(node.input[0])
    kwargs = {"keepdim": keepdims}
    if axes is None or len(axes) == 0:
        return builder.call_function(node, torch.mean, (x,), kwargs=kwargs)

    if len(axes) == 1:
        dim = axes[0]
    else:
        dim = axes
    kwargs["dim"] = dim
    return builder.call_function(node, torch.mean, (x,), kwargs=kwargs)


@register_op("If")
def if_op(builder: "GraphBuilder", node: onnx.NodeProto):
    if not node.input:
        raise ConversionError("If node missing condition input")
    try:
        cond_tensor = builder.get_tensor_value(node.input[0])
    except ConversionError as exc:
        raise UnsupportedOperatorError("If condition must be statically known") from exc
    if cond_tensor.numel() != 1:
        raise UnsupportedOperatorError("If condition must be a scalar tensor")
    cond_value = bool(cond_tensor.reshape(-1)[0].item())
    branch_name = "then_branch" if cond_value else "else_branch"
    branch_graph = next((attr.g for attr in node.attribute if attr.name == branch_name), None)
    if branch_graph is None:
        raise ConversionError(f"If node missing '{branch_name}' attribute")
    evaluator = StaticGraphEvaluator(builder)
    branch_outputs = evaluator.evaluate(branch_graph)
    output_names = [name for name in node.output if name]
    if len(branch_outputs) != len(output_names):
        raise ConversionError(
            f"If branch produced {len(branch_outputs)} outputs but graph declares {len(output_names)}"
        )
    for name, value in zip(output_names, branch_outputs):
        if isinstance(value, torch.Tensor):
            builder.tensor_values[name] = value.clone().detach()
        elif isinstance(value, list):
            builder.sequence_values[name] = list(value)
        else:
            raise ConversionError("Unsupported value type returned from If branch")
    return None


@register_op("Loop")
def loop(builder: "GraphBuilder", node: onnx.NodeProto):
    body = next((attr.g for attr in node.attribute if attr.name == "body"), None)
    if body is None:
        raise ConversionError("Loop node missing body graph")

    inputs = list(node.input)
    max_trip_count = None
    if inputs and inputs[0]:
        trip_tensor = builder.get_tensor_value(inputs[0])
        if trip_tensor.numel() != 1:
            raise UnsupportedOperatorError("Loop trip count must be scalar")
        max_trip_count = int(trip_tensor.reshape(-1)[0].item())

    if len(inputs) >= 2 and inputs[1]:
        cond_tensor = builder.get_tensor_value(inputs[1])
        cond_value = bool(cond_tensor.reshape(-1)[0].item())
    else:
        cond_value = True

    state_inputs: list[torch.Tensor | list] = []
    for name in inputs[2:]:
        if name in builder.sequence_values:
            state_inputs.append(list(builder.sequence_values[name]))
        else:
            try:
                state_inputs.append(builder.get_tensor_value(name))
            except ConversionError as exc:
                raise UnsupportedOperatorError("Loop initial state must be statically known") from exc

    num_state = len(state_inputs)
    evaluator = StaticGraphEvaluator(builder)
    iteration = 0

    while True:
        if max_trip_count is not None and iteration >= max_trip_count:
            break
        if not cond_value:
            break

        tensor_env = {
            body.input[0].name: torch.tensor(iteration, dtype=torch.int64),
            body.input[1].name: torch.tensor(cond_value, dtype=torch.bool),
        }
        sequence_env: Dict[str, list] = {}
        for value_info, value in zip(body.input[2:], state_inputs):
            name = value_info.name
            if isinstance(value, list):
                sequence_env[name] = list(value)
            else:
                tensor_env[name] = value.clone().detach()

        outputs = evaluator.evaluate(body, tensor_env=tensor_env, sequence_env=sequence_env)
        if len(outputs) < 1 + num_state:
            raise UnsupportedOperatorError("Loop body returned insufficient outputs")

        cond_tensor = outputs[0]
        if not isinstance(cond_tensor, torch.Tensor):
            raise ConversionError("Loop condition must be a tensor")
        cond_value = bool(cond_tensor.reshape(-1)[0].item())

        new_states = outputs[1 : 1 + num_state]
        state_inputs = [
            value.clone().detach() if isinstance(value, torch.Tensor) else list(value)
            for value in new_states
        ]
        if len(outputs) > 1 + num_state:
            raise UnsupportedOperatorError("Loop scan outputs are not supported")

        iteration += 1

    for name, value in zip(node.output, state_inputs):
        if isinstance(value, list):
            builder.sequence_values[name] = list(value)
        else:
            builder.tensor_values[name] = value.clone().detach()
    return None


@register_op("Flatten")
def flatten(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    attrs = builder.get_attributes(node)
    axis = int(attrs.get("axis", 1))
    x = builder.get_value(node.input[0])
    return builder.call_function(node, torch.flatten, (x,), kwargs={"start_dim": axis})


@register_op("Reshape")
def reshape(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    data = builder.get_value(node.input[0])
    shape_tensor = builder.get_tensor_value(node.input[1])
    shape = tuple(int(dim) for dim in shape_tensor.view(-1).tolist())
    return builder.call_function(node, torch.reshape, (data, shape))


@register_op("Identity")
def identity(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    return builder.get_value(node.input[0])


@register_op("Dropout")
def dropout(builder: "GraphBuilder", node: onnx.NodeProto) -> fx.Node:
    # Dropout is typically omitted during inference and training graphs can handle it separately.
    return builder.get_value(node.input[0])


__all__ = ["OP_REGISTRY", "register_op"]
