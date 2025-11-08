"""Core graph builder that lowers ONNX graphs to PyTorch FX graphs."""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Sequence

import onnx
from onnx import helper, shape_inference
import torch
from torch import fx, nn

from .errors import ConversionError, UnsupportedOperatorError
from .utils import ValueInfo, build_value_info_map, sanitize_name, tensor_proto_to_torch
from .ops import OP_REGISTRY


class FXModuleRoot(nn.Module):
    """Root module that hosts parameters, buffers, and submodules for FX."""

    pass


class GraphBuilder:
    """Convert an ONNX model into a ``torch.fx.GraphModule``."""

    def __init__(self, model: onnx.ModelProto, *, debug: bool = False) -> None:
        self.model = model
        self.debug = debug
        self.graph = fx.Graph()
        self.root_module = FXModuleRoot()

        self.initializers: Dict[str, torch.Tensor] = {
            tensor.name: tensor_proto_to_torch(tensor)
            for tensor in model.graph.initializer
        }
        self.tensor_values: Dict[str, torch.Tensor] = dict(self.initializers)
        self.sequence_values: Dict[str, list] = {}
        self.constant_attr_nodes: Dict[str, fx.Node] = {}
        self.env: Dict[str, fx.Node] = {}
        self.value_info: Dict[str, ValueInfo] = build_value_info_map(model)
        self.graph_outputs: List[str] = [value.name for value in model.graph.output]
        self.input_names: List[str] = []
        self.module_name_counters: Dict[str, int] = collections.defaultdict(int)
        self.attr_name_counters: Dict[str, int] = collections.defaultdict(int)
        self.placeholder_name_counters: Dict[str, int] = collections.defaultdict(int)
        self.value_users: Dict[str, int] = self._count_value_users(model.graph)

    # ---------------------------------------------------------------------
    # High-level API
    # ---------------------------------------------------------------------

    @classmethod
    def from_model(cls, model_or_path: onnx.ModelProto | str, *, debug: bool = False) -> "GraphBuilder":
        """Factory that loads a model if needed and performs shape inference."""

        if isinstance(model_or_path, str):
            model = onnx.load(model_or_path)
        elif isinstance(model_or_path, onnx.ModelProto):
            model = model_or_path
        else:
            raise TypeError("model_or_path must be a path or onnx.ModelProto")

        try:
            # Shape inference enriches ValueInfo so we can propagate metadata.
            inferred = shape_inference.infer_shapes(model)
            model = inferred
        except Exception:
            # Fall back to the original model if inference fails.
            pass
        return cls(model, debug=debug)

    def build(self) -> fx.GraphModule:
        """Perform the conversion and return the constructed GraphModule."""

        self._create_placeholders()
        self._convert_nodes()
        self._create_outputs()
        graph_module = fx.GraphModule(self.root_module, self.graph)
        graph_module.graph.lint()
        if self.debug:
            print(graph_module.graph)
        return graph_module

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def get_value(self, name: str) -> fx.Node:
        """Retrieve an FX node corresponding to a named ONNX value."""

        if not name:
            raise ConversionError("Empty value name encountered in ONNX graph")
        if name in self.env:
            node = self.env[name]
            if node is None:
                raise ConversionError(f"Value '{name}' is not produced in the FX graph")
            return node
        if name in self.tensor_values:
            tensor = self.tensor_values[name]
            attr_node = self._create_attr_node(name, tensor, trainable=False)
            self.env[name] = attr_node
            return attr_node
        if name in self.constant_attr_nodes:
            return self.constant_attr_nodes[name]
        if name in self.initializers:
            return self._get_initializer_attr_node(name)
        raise ConversionError(f"Value '{name}' is unknown in the current graph context")

    def get_initializer_tensor(self, name: str) -> torch.Tensor:
        """Return the Torch tensor backing an ONNX initializer."""

        try:
            tensor = self.initializers[name]
        except KeyError as exc:
            raise ConversionError(f"Initializer '{name}' was not found in the model") from exc
        return tensor.clone().detach()

    def get_tensor_value(self, name: str) -> torch.Tensor:
        """Return a tensor for a value that is statically known at conversion time."""

        if name in self.tensor_values:
            return self.tensor_values[name].clone().detach()
        raise ConversionError(f"Tensor value for '{name}' is not statically known")

    def get_attributes(self, node: onnx.NodeProto) -> Dict[str, object]:
        """Convert ONNX node attributes into a Python dictionary."""

        attrs: Dict[str, object] = {}
        for attr in node.attribute:
            attrs[attr.name] = helper.get_attribute_value(attr)
        return attrs

    def get_value_info(self, name: str) -> Optional[ValueInfo]:
        """Fetch stored metadata for a given ONNX value if available."""

        return self.value_info.get(name)

    def register_module(self, node: onnx.NodeProto, module: nn.Module, *, base_name: Optional[str] = None) -> str:
        """Register a submodule on the FX root and return its qualified name."""

        raw_name = base_name or node.name or node.op_type.lower()
        sanitized = sanitize_name(raw_name)
        counter = self.module_name_counters[sanitized]
        self.module_name_counters[sanitized] += 1
        name = sanitized if counter == 0 else f"{sanitized}_{counter}"
        self.root_module.add_module(name, module)
        return name

    def call_module(
        self,
        node: onnx.NodeProto,
        module_name: str,
        args: Sequence[fx.Node],
        kwargs: Optional[Dict[str, object]] = None,
        *,
        output_idx: int = 0,
    ) -> fx.Node:
        fx_node = self.graph.call_module(module_name, args=tuple(args), kwargs=kwargs or {})
        self._annotate_node(fx_node, node, output_idx)
        return fx_node

    def call_function(
        self,
        node: onnx.NodeProto,
        target,
        args: Sequence[fx.Node | object],
        kwargs: Optional[Dict[str, object]] = None,
        *,
        output_idx: int = 0,
    ) -> fx.Node:
        fx_node = self.graph.call_function(target, args=tuple(args), kwargs=kwargs or {})
        self._annotate_node(fx_node, node, output_idx)
        return fx_node

    def create_constant_attr(
        self,
        node: onnx.NodeProto,
        tensor: torch.Tensor,
        *,
        name_hint: Optional[str] = None,
        trainable: Optional[bool] = None,
    ) -> fx.Node:
        """Register a tensor as an attribute on the root module and return its node."""

        base_name = name_hint or node.name or (node.output[0] if node.output else "const")
        attr_node = self._create_attr_node(base_name, tensor, trainable=trainable)
        return attr_node

    def set_output_value(self, name: str, node: fx.Node) -> None:
        if name:
            self.env[name] = node

    def set_sequence_value(self, name: str, value: list) -> None:
        self.sequence_values[name] = value

    def get_sequence_value(self, name: str) -> list:
        try:
            return self.sequence_values[name]
        except KeyError as exc:
            raise ConversionError(f"Sequence value '{name}' is not statically known") from exc

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------

    def _count_value_users(self, graph: onnx.GraphProto) -> Dict[str, int]:
        counter: Dict[str, int] = collections.Counter()
        for node in graph.node:
            for input_name in node.input:
                if input_name:
                    counter[input_name] += 1
        for output in graph.output:
            if output.name:
                counter[output.name] += 1
        return counter

    def _create_placeholders(self) -> None:
        initializer_names = set(self.initializers.keys())
        for value in self.model.graph.input:
            if value.name in initializer_names:
                continue
            sanitized = sanitize_name(value.name)
            counter = self.placeholder_name_counters[sanitized]
            self.placeholder_name_counters[sanitized] += 1
            preferred_name = sanitized if counter == 0 else f"{sanitized}_{counter}"

            placeholder = self.graph.placeholder(preferred_name)
            if placeholder.name != preferred_name:
                placeholder.name = preferred_name
            info = self.get_value_info(value.name)
            if info:
                placeholder.meta["onnx_shape"] = info.shape
                placeholder.meta["onnx_dtype"] = info.dtype
            placeholder.meta["onnx_op_type"] = "Input"
            placeholder.meta["onnx_name"] = value.name
            placeholder.meta["onnx_sanitized_name"] = preferred_name
            self.env[value.name] = placeholder
            self.input_names.append(value.name)

    def _convert_nodes(self) -> None:
        for node in self.model.graph.node:
            handler = OP_REGISTRY.get(node.op_type)
            if handler is None:
                raise UnsupportedOperatorError(
                    f"Operator '{node.op_type}' is not supported (node name: '{node.name}')"
                )
            result = handler(self, node)
            if result is None:
                continue
            outputs = [name for name in node.output if name]
            if isinstance(result, fx.Node):
                self._assign_single_output(node, result, outputs)
            elif isinstance(result, Sequence):
                self._assign_sequence_output(node, result, outputs)
            elif isinstance(result, dict):
                for out_name, fx_node in result.items():
                    self.env[out_name] = fx_node
            else:
                raise ConversionError(
                    f"Handler for op '{node.op_type}' returned unsupported value type"
                )

    def _assign_single_output(self, node: onnx.NodeProto, fx_node: fx.Node, outputs: List[str]) -> None:
        if not outputs:
            return
        self.env[outputs[0]] = fx_node
        for extra in outputs[1:]:
            if self.value_users.get(extra, 0) > 0:
                raise UnsupportedOperatorError(
                    f"Operator '{node.op_type}' produces multiple outputs which are not supported"
                )

    def _assign_sequence_output(
        self,
        node: onnx.NodeProto,
        nodes: Sequence[fx.Node],
        outputs: List[str],
    ) -> None:
        filtered_outputs = outputs[: len(nodes)]
        if len(filtered_outputs) != len(nodes):
            raise ConversionError(
                f"Operator '{node.op_type}' produced {len(nodes)} tensors, "
                f"but graph declares {len(outputs)} outputs"
            )
        for name, fx_node in zip(filtered_outputs, nodes):
            self.env[name] = fx_node
        for extra in outputs[len(nodes) :]:
            if self.value_users.get(extra, 0) > 0:
                raise UnsupportedOperatorError(
                    f"Operator '{node.op_type}' produces more outputs than handled"
                )

    def _create_outputs(self) -> None:
        output_nodes = [self.get_value(name) for name in self.graph_outputs]
        if len(output_nodes) == 1:
            self.graph.output(output_nodes[0])
        else:
            self.graph.output(tuple(output_nodes))

    def _get_initializer_attr_node(self, name: str) -> fx.Node:
        if name in self.constant_attr_nodes:
            return self.constant_attr_nodes[name]
        tensor = self.initializers[name]
        attr_node = self._create_attr_node(name, tensor, trainable=None)
        self.constant_attr_nodes[name] = attr_node
        return attr_node

    def _create_attr_node(
        self,
        source_name: str,
        tensor: torch.Tensor,
        *,
        trainable: Optional[bool],
    ) -> fx.Node:
        sanitized = sanitize_name(source_name)
        counter = self.attr_name_counters[sanitized]
        self.attr_name_counters[sanitized] += 1
        attr_name = sanitized if counter == 0 else f"{sanitized}_{counter}"

        tensor_copy = tensor.clone().detach()
        if trainable is None:
            trainable = tensor_copy.dtype.is_floating_point or tensor_copy.dtype.is_complex

        if trainable:
            parameter = nn.Parameter(tensor_copy)
            self.root_module.register_parameter(attr_name, parameter)
        else:
            self.root_module.register_buffer(attr_name, tensor_copy)

        attr_node = self.graph.create_node("get_attr", attr_name, (), {})
        attr_node.meta["onnx_name"] = source_name
        attr_node.meta["onnx_op_type"] = "Constant"
        attr_node.meta["onnx_shape"] = list(tensor.shape)
        attr_node.meta["onnx_dtype"] = tensor.dtype

        self.constant_attr_nodes[source_name] = attr_node
        self.tensor_values[source_name] = tensor.clone().detach()
        return attr_node

    def _annotate_node(self, fx_node: fx.Node, node: onnx.NodeProto, output_idx: int) -> None:
        outputs = [name for name in node.output if name]
        value_name = outputs[output_idx] if output_idx < len(outputs) else node.name
        info = self.get_value_info(value_name) if value_name else None
        fx_node.meta["onnx_op_type"] = node.op_type
        fx_node.meta["onnx_name"] = node.name or value_name
        if outputs:
            fx_node.meta["onnx_outputs"] = outputs
        if info:
            fx_node.meta["onnx_shape"] = info.shape
            fx_node.meta["onnx_dtype"] = info.dtype


__all__ = ["GraphBuilder", "FXModuleRoot"]
