# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Union

import onnx
from torch import fx

from .graph_builder import GraphBuilder


def convert_onnx_to_fx(
    model: Union[onnx.ModelProto, str],
    *,
    debug: bool = False,
) -> fx.GraphModule:
    """Convert an ONNX model (or file path) into a ``torch.fx.GraphModule``.

    Parameters
    ----------
    model : Union[onnx.ModelProto, str]
        Either an in-memory ``onnx.ModelProto`` or a filesystem path to an ONNX
        model file.
    debug : bool, optional
        When ``True`` prints the resulting FX graph for inspection.

    Returns
    -------
    torch.fx.GraphModule
        A graph module that mirrors the ONNX graph and can be further edited or
        used for fine-tuning in PyTorch.
    """

    if isinstance(model, str):
        model = onnx.load(model)
    elif isinstance(model, onnx.ModelProto):
        model = model
    else:
        raise TypeError("model must be a path or onnx.ModelProto")

    builder = GraphBuilder.from_model(model, debug=debug)
    return builder.build()
