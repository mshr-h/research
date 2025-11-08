"""User-facing conversion entry-points."""

from __future__ import annotations

from typing import Union

import onnx
from torch import fx

from .errors import ConversionError, UnsupportedOperatorError
from .graph_builder import GraphBuilder


def convert_onnx_to_fx(
    model: Union[onnx.ModelProto, str],
    *,
    debug: bool = False,
) -> fx.GraphModule:
    """Convert an ONNX model (or file path) into a ``torch.fx.GraphModule``.

    Parameters
    ----------
    model:
        Either an in-memory ``onnx.ModelProto`` or a filesystem path to an ONNX
        model file.
    debug:
        When ``True`` prints the resulting FX graph for inspection.

    Returns
    -------
    torch.fx.GraphModule
        A graph module that mirrors the ONNX graph and can be further edited or
        used for fine-tuning in PyTorch.

    Raises
    ------
    UnsupportedOperatorError
        If the model contains operators that are not yet implemented by the
        converter.
    ConversionError
        If any other issue prevents the conversion from succeeding.
    """

    try:
        builder = GraphBuilder.from_model(model, debug=debug)
        return builder.build()
    except UnsupportedOperatorError:
        raise
    except ConversionError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected failures
        raise ConversionError("Unexpected failure while converting ONNX model") from exc
