# SPDX-License-Identifier: Apache-2.0
"""Internal utility helpers for ONNX to FX conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper


_DTYPE_MAP: Dict[int, torch.dtype] = {
    TensorProto.FLOAT: torch.float32,
    TensorProto.DOUBLE: torch.float64,
    TensorProto.FLOAT16: torch.float16,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.INT64: torch.int64,
    TensorProto.INT32: torch.int32,
    TensorProto.INT16: torch.int16,
    TensorProto.INT8: torch.int8,
    TensorProto.UINT8: torch.uint8,
    TensorProto.BOOL: torch.bool,
}


@dataclass(frozen=True)
class ValueInfo:
    """Container for light-weight ONNX value metadata."""

    shape: Optional[List[Optional[int]]]
    dtype: Optional[torch.dtype]


def tensor_proto_to_torch(tensor: TensorProto) -> torch.Tensor:
    """Convert an ONNX ``TensorProto`` to a ``torch.Tensor``.

    A copy is always returned to decouple the Torch tensor from the ONNX buffer.
    """

    np_array = numpy_helper.to_array(tensor)
    torch_dtype = _DTYPE_MAP.get(tensor.data_type)
    result = torch.from_numpy(np.array(np_array))
    if torch_dtype is not None and result.dtype != torch_dtype:
        result = result.to(torch_dtype)
    return result.clone().detach()


def onnx_dtype_to_torch(dtype: int) -> Optional[torch.dtype]:
    """Map an ONNX TensorProto data type enum to a Torch dtype, if known."""

    return _DTYPE_MAP.get(dtype)


def extract_tensor_shape(value: onnx.ValueInfoProto) -> Optional[List[Optional[int]]]:
    """Extract a list-based representation of a tensor shape from a value info."""

    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: List[Optional[int]] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(None)
        else:
            dims.append(None)
    return dims


def extract_tensor_dtype(value: onnx.ValueInfoProto) -> Optional[torch.dtype]:
    """Extract the Torch dtype that corresponds to a value info if available."""

    tensor_type = value.type.tensor_type
    elem_type = tensor_type.elem_type
    return _DTYPE_MAP.get(elem_type)


def build_value_info_map(model: onnx.ModelProto) -> Dict[str, ValueInfo]:
    """Collect ONNX value info metadata for quick lookup during conversion."""

    all_infos: Iterable[onnx.ValueInfoProto] = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    info_map: Dict[str, ValueInfo] = {}
    for value_info in all_infos:
        info_map[value_info.name] = ValueInfo(
            shape=extract_tensor_shape(value_info),
            dtype=extract_tensor_dtype(value_info),
        )
    return info_map


_NAME_SANITIZER = re.compile(r"[^0-9a-zA-Z_]")


def sanitize_name(name: str) -> str:
    """Sanitize a string so it can be used as an attribute / module name."""

    clean = _NAME_SANITIZER.sub("_", name)
    if clean and clean[0].isdigit():
        clean = f"_{clean}"
    return clean or "const"
