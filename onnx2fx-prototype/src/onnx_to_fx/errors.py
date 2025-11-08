"""Custom exception types used by the onnx_to_fx package."""

from __future__ import annotations


class ConversionError(RuntimeError):
    """Raised when a model conversion fails for any reason."""


class UnsupportedOperatorError(ConversionError):
    """Raised when encountering an ONNX operator that is not yet supported."""
