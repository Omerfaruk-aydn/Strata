"""Strata Ultra: experimental ultra-low-bit tensor primitives.

This package is intentionally independent from llama.cpp.  The first milestone
provides a deterministic ternary tensor codec that can be used by future CPU,
CUDA, and Vulkan execution backends.
"""

from .format import STRATA_FORMAT_VERSION, TensorHeader
from .ternary import decode_ternary, encode_ternary

__all__ = [
    "STRATA_FORMAT_VERSION",
    "TensorHeader",
    "encode_ternary",
    "decode_ternary",
]
