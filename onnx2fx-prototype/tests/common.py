# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort


def ort_run(model: onnx.ModelProto, feeds: Dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in model.graph.output]
    return session.run(output_names, feeds)
