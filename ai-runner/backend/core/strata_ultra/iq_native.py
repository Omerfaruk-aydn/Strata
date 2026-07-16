"""Optional native GGML IQ decoder bridge."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

_LIBRARY: Optional[ctypes.CDLL] = None


def _load() -> Optional[ctypes.CDLL]:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    configured = os.environ.get("STRATA_IQ_LIBRARY", "").strip()
    candidates = [Path(configured)] if configured else []
    for root in (Path(__file__).resolve().parents[3] / "native", Path.home() / ".ai-runner" / "bin"):
        candidates.extend(root / name for name in ("strata_iq.dll", "libstrata_iq.so", "libstrata_iq.dylib"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
            function = library.strata_ggml_dequant_iq
            function.argtypes = [
                ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
            ]
            function.restype = ctypes.c_int
            _LIBRARY = library
            return library
        except (OSError, AttributeError):
            continue
    return None


def native_iq_available() -> bool:
    return _load() is not None


def decode_iq_native(type_id: int, raw: bytes, value_count: int) -> tuple[float, ...]:
    library = _load()
    if library is None:
        raise RuntimeError("Native GGML IQ decoder is not installed; build native/ with STRATA_GGML_ROOT")
    source = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    output = (ctypes.c_float * value_count)()
    status = library.strata_ggml_dequant_iq(type_id, source, len(raw), output, value_count)
    if status != 0:
        raise RuntimeError(f"Native GGML IQ decoder failed with status {status}")
    return tuple(output)
