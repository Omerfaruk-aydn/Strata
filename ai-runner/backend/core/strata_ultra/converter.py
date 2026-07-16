"""GGUF-to-Strata conversion for F32/F16 source tensors.

The converter intentionally supports floating-point tensors first.  GGUF quantized block formats
must be decoded with architecture/type-specific kernels before requantization;
silently treating those bytes as floats would create a corrupt model.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO

from ..model_loader import _extract_metadata, _read_exact
from .container import StrataContainerWriter, TensorRecord
from .ternary import encode_ternary

GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
_QK = 32


def _decode_q4_0(raw: bytes, count: int) -> tuple[float, ...]:
    if count % _QK or len(raw) != (count // _QK) * 18:
        raise ValueError("Invalid Q4_0 tensor block size")
    values = []
    for offset in range(0, len(raw), 18):
        scale = struct.unpack_from("<e", raw, offset)[0]
        packed = raw[offset + 2:offset + 18]
        for index in range(16):
            values.append(scale * ((packed[index] & 0x0F) - 8))
        for index in range(16):
            values.append(scale * ((packed[index] >> 4) - 8))
    return tuple(values)


def _decode_q8_0(raw: bytes, count: int) -> tuple[float, ...]:
    if count % _QK or len(raw) != (count // _QK) * 34:
        raise ValueError("Invalid Q8_0 tensor block size")
    values = []
    for offset in range(0, len(raw), 34):
        scale = struct.unpack_from("<e", raw, offset)[0]
        values.extend(scale * value for value in struct.unpack_from("<32b", raw, offset + 2))
    return tuple(values)


def _read_string(stream: BinaryIO) -> str:
    size = struct.unpack("<Q", _read_exact(stream, 8))[0]
    if size == 0 or size > 1_000_000:
        raise ValueError("GGUF tensor name is invalid")
    return _read_exact(stream, size).decode("utf-8")


def convert_gguf_to_strata(
    source: str | Path,
    target: str | Path,
    *,
    group_size: int = 128,
    max_tensor_bytes: int = 2 * 1024 * 1024 * 1024,
) -> dict:
    """Convert an F32 GGUF into a checksummed experimental STRATA-Q0.5 file."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if not source.is_file() or source.suffix.lower() != ".gguf":
        raise FileNotFoundError("Source GGUF file was not found")
    if group_size <= 0 or max_tensor_bytes <= 0:
        raise ValueError("group_size and max_tensor_bytes must be positive")

    with source.open("rb") as stream:
        if _read_exact(stream, 4) != GGUF_MAGIC:
            raise ValueError("Invalid GGUF magic")
        version, tensor_count, metadata_count = struct.unpack("<IQQ", _read_exact(stream, 20))
        if version not in {1, 2, 3}:
            raise ValueError(f"Unsupported GGUF version: {version}")
        metadata = _extract_metadata(stream, metadata_count, version)
        infos = []
        for _ in range(tensor_count):
            name = _read_string(stream)
            n_dims = struct.unpack("<I", _read_exact(stream, 4))[0]
            if n_dims == 0 or n_dims > 8:
                raise ValueError(f"Invalid dimensions for tensor {name}")
            dims = struct.unpack(f"<{n_dims}Q", _read_exact(stream, 8 * n_dims))
            tensor_type, offset = struct.unpack("<IQ", _read_exact(stream, 12))
            infos.append((name, dims, tensor_type, offset))
        alignment = int(metadata.get("general.alignment", 32) or 32)
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        file_size = source.stat().st_size
        writer = StrataContainerWriter({
            "source": source.name,
            "source_format": "GGUF",
            "profile": "STRATA-Q0.5",
            "source_version": version,
            "architecture": metadata.get("general.architecture", ""),
            "group_size": group_size,
        })
        converted = 0
        for name, dims, tensor_type, offset in infos:
            if tensor_type not in {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}:
                supported = "F32/F16/Q4_0/Q8_0 only in this converter"
                raise ValueError(f"Tensor {name} uses GGUF type {tensor_type}; {supported}")
            count = 1
            for dim in dims:
                count *= dim
            if tensor_type == GGML_TYPE_F32:
                byte_count = count * 4
            elif tensor_type == GGML_TYPE_F16:
                byte_count = count * 2
            elif tensor_type == GGML_TYPE_Q4_0:
                byte_count = (count // _QK) * 18
            else:
                byte_count = (count // _QK) * 34
            if byte_count > max_tensor_bytes or data_start + offset + byte_count > file_size:
                raise ValueError(f"Tensor {name} exceeds safe input bounds")
            stream.seek(data_start + offset)
            raw = _read_exact(stream, byte_count)
            if tensor_type == GGML_TYPE_F32:
                values = struct.unpack(f"<{count}f", raw)
            elif tensor_type == GGML_TYPE_F16:
                values = struct.unpack(f"<{count}e", raw)
            elif tensor_type == GGML_TYPE_Q4_0:
                values = _decode_q4_0(raw, count)
            else:
                values = _decode_q8_0(raw, count)
            packed, scales = encode_ternary(values, group_size)
            scales_raw = struct.pack(f"<{len(scales)}f", *scales)
            rows = int(dims[0])
            cols = int(count // rows)
            writer.add_tensor(TensorRecord(name, rows, cols, group_size, "ternary-q05", packed, scales_raw))
            converted += 1
        writer.write(target)
    return {
        "source": str(source),
        "target": str(target),
        "tensor_count": converted,
        "source_bytes": os.path.getsize(source),
        "target_bytes": os.path.getsize(target),
        "codec": "ternary-q05",
    }
