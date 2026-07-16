"""Optional ctypes bridge for the compiled Strata CUDA backend."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

_LIBRARY: Optional[ctypes.CDLL] = None


def _candidates() -> list[Path]:
    configured = os.environ.get("STRATA_CUDA_LIBRARY", "").strip()
    if configured:
        return [Path(configured).expanduser()]
    suffixes = ("strata_cuda.dll", "libstrata_cuda.so", "libstrata_cuda.dylib")
    roots = [Path(__file__).resolve().parents[3] / "native", Path.home() / ".ai-runner" / "bin"]
    return [root / name for root in roots for name in suffixes]


def _load() -> Optional[ctypes.CDLL]:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    for candidate in _candidates():
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
            function = library.strata_cuda_ternary_matvec
            function.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ]
            function.restype = ctypes.c_int
            _LIBRARY = library
            return library
        except (OSError, AttributeError):
            continue
    return None


def cuda_available() -> bool:
    return _load() is not None


def matvec_cuda(record, vector: list[float]) -> list[float]:
    """Execute one ternary tensor matvec through the optional native ABI."""
    if record.codec != "ternary-q05":
        raise ValueError("CUDA backend currently accepts only ternary-q05 tensors")
    if len(vector) != record.cols:
        raise ValueError(f"vector length {len(vector)} != tensor columns {record.cols}")
    library = _load()
    if library is None:
        raise RuntimeError("Strata CUDA backend is not installed; build native/ with STRATA_ENABLE_CUDA=ON")
    packed = (ctypes.c_uint8 * len(record.payload)).from_buffer_copy(record.payload)
    scale_count = (record.rows * record.cols + record.group_size - 1) // record.group_size
    if len(record.scales) != scale_count * 4:
        raise ValueError(f"invalid scale table for tensor {record.name}")
    scales = (ctypes.c_float * scale_count).from_buffer_copy(record.scales)
    values = (ctypes.c_float * record.cols)(*vector)
    output = (ctypes.c_float * record.rows)()
    status = library.strata_cuda_ternary_matvec(
        packed, scales, values, output, record.rows, record.cols, record.group_size
    )
    if status != 0:
        raise RuntimeError(f"Strata CUDA matvec failed with CUDA error code {status}")
    return list(output)
