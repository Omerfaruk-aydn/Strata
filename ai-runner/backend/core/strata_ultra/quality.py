"""Quality and sparsity metrics for low-bit tensor experiments."""

from __future__ import annotations

import math
from typing import Sequence


def tensor_quality(reference: Sequence[float], reconstructed: Sequence[float]) -> dict[str, float]:
    if not reference or len(reference) != len(reconstructed):
        raise ValueError("reference and reconstructed tensors must have equal non-empty length")
    errors = [float(actual) - float(expected) for expected, actual in zip(reference, reconstructed)]
    mse = sum(error * error for error in errors) / len(errors)
    reference_norm = math.sqrt(sum(float(value) ** 2 for value in reference))
    reconstructed_norm = math.sqrt(sum(float(value) ** 2 for value in reconstructed))
    dot = sum(float(a) * float(b) for a, b in zip(reference, reconstructed))
    cosine = dot / (reference_norm * reconstructed_norm) if reference_norm and reconstructed_norm else 0.0
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "max_abs_error": max(abs(error) for error in errors),
        "cosine_similarity": cosine,
        "reference_nonzero_ratio": sum(value != 0 for value in reference) / len(reference),
        "reconstructed_nonzero_ratio": sum(value != 0 for value in reconstructed) / len(reconstructed),
    }
