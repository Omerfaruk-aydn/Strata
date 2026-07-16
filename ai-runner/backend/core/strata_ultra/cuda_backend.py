"""Optional ctypes bridge for the compiled Strata CUDA backend."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from typing import Optional

_LIBRARY: Optional[ctypes.CDLL] = None
_DLL_DIRECTORIES: list[object] = []


def _prepare_windows_dll_search() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    configured = os.environ.get("STRATA_CUDA_LIBRARY", "").strip()
    directories = [Path(configured).expanduser().resolve().parent] if configured else []
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        directories.append(Path(cuda_path).expanduser().resolve() / "bin")
    for directory in directories:
        if directory.is_dir():
            try:
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
            except OSError:
                continue


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
    _prepare_windows_dll_search()
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
            kv_decode = library.strata_cuda_kv_decode
            kv_decode.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            kv_decode.restype = ctypes.c_int
            _LIBRARY = library
            return library
        except (OSError, AttributeError):
            continue
    return None


def cuda_available() -> bool:
    return _load() is not None


def matvec_cuda(record, vector: list[float]) -> list[float]:
    """Execute one ternary tensor matvec through the optional native ABI."""
    for field in ("rows", "cols", "group_size"):
        value = getattr(record, field)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{field} must be an integer in the uint32 range")
    if record.codec != "ternary-q05":
        raise ValueError("CUDA backend currently accepts only ternary-q05 tensors")
    if len(vector) != record.cols:
        raise ValueError(f"vector length {len(vector)} != tensor columns {record.cols}")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("CUDA matvec vector must contain finite values")
    packed_bytes = (record.rows * record.cols + 3) // 4
    if len(record.payload) != packed_bytes:
        raise ValueError(f"invalid packed payload for tensor {record.name}: {len(record.payload)} != {packed_bytes}")
    library = _load()
    if library is None:
        raise RuntimeError("Strata CUDA backend is not installed; build native/ with STRATA_ENABLE_CUDA=ON")
    packed = (ctypes.c_uint8 * len(record.payload)).from_buffer_copy(record.payload)
    scale_count = (record.rows * record.cols + record.group_size - 1) // record.group_size
    if len(record.scales) != scale_count * 4:
        raise ValueError(f"invalid scale table for tensor {record.name}")
    scales = (ctypes.c_float * scale_count).from_buffer_copy(record.scales)
    if any(not math.isfinite(value) for value in scales):
        raise ValueError(f"invalid non-finite scale table for tensor {record.name}")
    values = (ctypes.c_float * record.cols)(*vector)
    output = (ctypes.c_float * record.rows)()
    status = library.strata_cuda_ternary_matvec(
        packed, scales, values, output, record.rows, record.cols, record.group_size
    )
    if status != 0:
        raise RuntimeError(f"Strata CUDA matvec failed with CUDA error code {status}")
    return list(output)


def decode_kv_cuda(cache) -> list[float]:
    """Decode sign1 or ternary05 KV storage through the native CUDA ABI."""
    if cache.mode not in {"sign1", "ternary05"}:
        raise ValueError("CUDA KV decode supports only sign1 and ternary05")
    if not isinstance(cache.count, int) or not 1 <= cache.count <= 0xFFFFFFFF:
        raise ValueError("KV count must be in the uint32 range")
    if not isinstance(cache.group_size, int) or not 1 <= cache.group_size <= 0xFFFFFFFF:
        raise ValueError("KV group_size must be in the uint32 range")
    bits = 1 if cache.mode == "sign1" else 2
    payload_bytes = (cache.count + (7 if bits == 1 else 3)) // (8 if bits == 1 else 4)
    scale_count = (cache.count + cache.group_size - 1) // cache.group_size
    if len(cache.payload) != payload_bytes:
        raise ValueError(f"invalid CUDA KV payload length: {len(cache.payload)} != {payload_bytes}")
    if len(cache.scales) != scale_count:
        raise ValueError(f"invalid CUDA KV scale count: {len(cache.scales)} != {scale_count}")
    library = _load()
    if library is None:
        raise RuntimeError("Strata CUDA backend is not installed; build native/ with STRATA_ENABLE_CUDA=ON")
    packed = (ctypes.c_uint8 * len(cache.payload)).from_buffer_copy(cache.payload)
    scales = (ctypes.c_float * scale_count)(*cache.scales)
    output = (ctypes.c_float * cache.count)()
    status = library.strata_cuda_kv_decode(packed, scales, output, cache.count, cache.group_size, bits)
    if status != 0:
        raise RuntimeError(f"Strata CUDA KV decode failed with CUDA error code {status}")
    return list(output)
