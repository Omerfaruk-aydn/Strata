"""GGUF-to-Strata conversion for F32/F16 source tensors.

The converter intentionally supports floating-point tensors first.  GGUF quantized block formats
must be decoded with architecture/type-specific kernels before requantization;
silently treating those bytes as floats would create a corrupt model.
"""

from __future__ import annotations

import os
import json
import struct
from pathlib import Path
from typing import BinaryIO

from ..model_loader import _extract_metadata, _read_exact
from .container import StrataContainerWriter, TensorRecord
from .ternary import decode_ternary, encode_ternary
from .sparse_codec import decode_sparse05, encode_sparse05
from .quality import tensor_quality
from .iq_registry import get_iq_codec
from .iq_native import decode_iq_native, native_iq_available

GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_IQ4_NL = 20
_QK = 32
_QK_K = 256
_KVALUES_IQ4_NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _decode_iq4_nl(raw: bytes, count: int) -> tuple[float, ...]:
    """Decode GGML IQ4_NL blocks (32 values, f16 scale + 16 nibbles)."""
    block_size = 18
    if count % 32 or len(raw) != (count // 32) * block_size:
        raise ValueError("Invalid IQ4_NL tensor block size")
    values = []
    for offset in range(0, len(raw), block_size):
        scale = struct.unpack_from("<e", raw, offset)[0]
        packed = raw[offset + 2:offset + block_size]
        values.extend(scale * _KVALUES_IQ4_NL[byte & 0x0F] for byte in packed)
        values.extend(scale * _KVALUES_IQ4_NL[byte >> 4] for byte in packed)
    return tuple(values)


def _decode_q2_k(raw: bytes, count: int) -> tuple[float, ...]:
    """Decode GGML Q2_K using the upstream 256-value super-block layout."""
    block_size = 2 + 2 + 16 + 64
    if count % _QK_K or len(raw) != (count // _QK_K) * block_size:
        raise ValueError("Invalid Q2_K tensor block size")
    values = []
    for offset in range(0, len(raw), block_size):
        d, dmin = struct.unpack_from("<ee", raw, offset)
        scales = raw[offset + 4:offset + 20]
        quant = raw[offset + 20:offset + 84]
        q_offset = 0
        scale_index = 0
        for _ in range(2):
            shift = 0
            for _ in range(4):
                sc = scales[scale_index]
                scale_index += 1
                scale, minimum = d * (sc & 0x0F), dmin * (sc >> 4)
                for local in range(16):
                    values.append(scale * ((quant[q_offset + local] >> shift) & 3) - minimum)
                sc = scales[scale_index]
                scale_index += 1
                scale, minimum = d * (sc & 0x0F), dmin * (sc >> 4)
                for local in range(16):
                    values.append(scale * ((quant[q_offset + 16 + local] >> shift) & 3) - minimum)
                shift += 2
            q_offset += 32
    return tuple(values)


def _decode_q3_k(raw: bytes, count: int) -> tuple[float, ...]:
    """Decode GGML Q3_K using the upstream high-bit mask and scale packing."""
    block_size = 32 + 64 + 12 + 2
    if count % _QK_K or len(raw) != (count // _QK_K) * block_size:
        raise ValueError("Invalid Q3_K tensor block size")
    values = []
    for offset in range(0, len(raw), block_size):
        hmask = raw[offset:offset + 32]
        quant = raw[offset + 32:offset + 96]
        packed = raw[offset + 96:offset + 108]
        d_all = struct.unpack_from("<e", raw, offset + 108)[0]
        aux = bytearray(16)
        aux[:12] = packed
        tmp = struct.unpack_from("<I", aux, 8)[0]
        a0, a1 = struct.unpack_from("<II", aux, 0)
        struct.pack_into("<I", aux, 8, ((a0 >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4))
        struct.pack_into("<I", aux, 12, ((a1 >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4))
        struct.pack_into("<I", aux, 0, (a0 & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4))
        struct.pack_into("<I", aux, 4, (a1 & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4))
        scales = struct.unpack("<16b", aux)
        q_offset = 0
        scale_index = 0
        mask = 1
        for _ in range(2):
            shift = 0
            for _ in range(4):
                dl = d_all * (scales[scale_index] - 32)
                scale_index += 1
                for local in range(16):
                    high = 0 if (hmask[local] & mask) else 4
                    values.append(dl * (((quant[q_offset + local] >> shift) & 3) - high))
                dl = d_all * (scales[scale_index] - 32)
                scale_index += 1
                for local in range(16):
                    high = 0 if (hmask[local + 16] & mask) else 4
                    values.append(dl * (((quant[q_offset + local + 16] >> shift) & 3) - high))
                shift += 2
                mask <<= 1
            q_offset += 32
    return tuple(values)


def _get_scale_min_q4_k(index: int, scales: bytes) -> tuple[int, int]:
    if index < 4:
        return scales[index] & 0x3F, scales[index + 4] & 0x3F
    return (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4), ((scales[index + 4] >> 4) & 0x0F) | ((scales[index] >> 6) << 4)


def _decode_q4_k(raw: bytes, count: int) -> tuple[float, ...]:
    block_size = 2 + 2 + 12 + 128
    if count % _QK_K or len(raw) != (count // _QK_K) * block_size:
        raise ValueError("Invalid Q4_K tensor block size")
    values = []
    for offset in range(0, len(raw), block_size):
        d = struct.unpack_from("<e", raw, offset)[0]
        minimum = struct.unpack_from("<e", raw, offset + 2)[0]
        scales = raw[offset + 4:offset + 16]
        quant = raw[offset + 16:offset + 144]
        scale_index = 0
        for start in range(0, _QK_K, 64):
            scale, minimum_code = _get_scale_min_q4_k(scale_index, scales)
            d1, m1 = d * scale, minimum * minimum_code
            scale, minimum_code = _get_scale_min_q4_k(scale_index + 1, scales)
            d2, m2 = d * scale, minimum * minimum_code
            for index in range(32):
                values.append(d1 * (quant[start // 2 + index] & 0x0F) - m1)
            for index in range(32):
                values.append(d2 * (quant[start // 2 + index] >> 4) - m2)
            scale_index += 2
    return tuple(values)


def _decode_q5_k(raw: bytes, count: int) -> tuple[float, ...]:
    block_size = 2 + 2 + 12 + 128 + 32
    if count % _QK_K or len(raw) != (count // _QK_K) * block_size:
        raise ValueError("Invalid Q5_K tensor block size")
    values = []
    for offset in range(0, len(raw), block_size):
        d = struct.unpack_from("<e", raw, offset)[0]
        minimum = struct.unpack_from("<e", raw, offset + 2)[0]
        scales = raw[offset + 4:offset + 16]
        low = raw[offset + 16:offset + 144]
        high = raw[offset + 144:offset + 176]
        scale_index = 0
        high_mask_1, high_mask_2 = 1, 2
        for start in range(0, _QK_K, 64):
            scale, minimum_code = _get_scale_min_q4_k(scale_index, scales)
            d1, m1 = d * scale, minimum * minimum_code
            scale, minimum_code = _get_scale_min_q4_k(scale_index + 1, scales)
            d2, m2 = d * scale, minimum * minimum_code
            for index in range(32):
                q = (low[start // 2 + index] & 0x0F) + (16 if high[index] & high_mask_1 else 0)
                values.append(d1 * q - m1)
            for index in range(32):
                q = (low[start // 2 + index] >> 4) + (16 if high[index] & high_mask_2 else 0)
                values.append(d2 * q - m2)
            high_mask_1 <<= 2
            high_mask_2 <<= 2
            scale_index += 2
    return tuple(values)


def _decode_q6_k(raw: bytes, count: int) -> tuple[float, ...]:
    block_size = 2 + 128 + 64 + 16
    if count % _QK_K or len(raw) != (count // _QK_K) * block_size:
        raise ValueError("Invalid Q6_K tensor block size")
    values = []
    for offset in range(0, len(raw), block_size):
        d = struct.unpack_from("<e", raw, offset)[0]
        low = raw[offset + 2:offset + 130]
        high = raw[offset + 130:offset + 194]
        scales = struct.unpack_from("<16b", raw, offset + 194)
        low_offset, high_offset, scale_offset = 0, 0, 0
        for start in range(0, _QK_K, 128):
            for index in range(32):
                q1 = (low[low_offset + index] & 0x0F) | (((high[high_offset + index] >> 0) & 3) << 4)
                q2 = (low[low_offset + index] >> 4) | (((high[high_offset + index] >> 2) & 3) << 4)
                q3 = (low[low_offset + index + 32] & 0x0F) | (((high[high_offset + index] >> 4) & 3) << 4)
                q4 = (low[low_offset + index + 32] >> 4) | (((high[high_offset + index] >> 6) & 3) << 4)
                values.extend((
                    d * scales[scale_offset] * (q1 - 32),
                    d * scales[scale_offset + 1] * (q2 - 32),
                    d * scales[scale_offset + 2] * (q3 - 32),
                    d * scales[scale_offset + 3] * (q4 - 32),
                ))
            low_offset += 64
            high_offset += 32
            scale_offset += 4
    return tuple(values)


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


def _extract_tokenizer_metadata(metadata: dict) -> dict:
    """Keep JSON-safe tokenizer contract fields in the Strata manifest."""
    selected = {}
    for key, value in metadata.items():
        if not key.startswith("tokenizer.ggml."):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        selected[key] = value
    return selected


def convert_gguf_to_strata(
    source: str | Path,
    target: str | Path,
    *,
    group_size: int = 128,
    max_tensor_bytes: int = 2 * 1024 * 1024 * 1024,
    target_codec: str = "ternary-q05",
    sparse_threshold: float = 0.125,
) -> dict:
    """Convert an F32 GGUF into a checksummed experimental STRATA-Q0.5 file."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if not source.is_file() or source.suffix.lower() != ".gguf":
        raise FileNotFoundError("Source GGUF file was not found")
    if group_size <= 0 or max_tensor_bytes <= 0:
        raise ValueError("group_size and max_tensor_bytes must be positive")
    if target_codec not in {"ternary-q05", "sparse05"}:
        raise ValueError("target_codec must be 'ternary-q05' or 'sparse05'")
    if sparse_threshold < 0:
        raise ValueError("sparse_threshold must be non-negative")

    with source.open("rb") as stream:
        if _read_exact(stream, 4) != GGUF_MAGIC:
            raise ValueError("Invalid GGUF magic")
        version, tensor_count, metadata_count = struct.unpack("<IQQ", _read_exact(stream, 20))
        if version not in {1, 2, 3}:
            raise ValueError(f"Unsupported GGUF version: {version}")
        metadata = _extract_metadata(stream, metadata_count, version)
        tokenizer_metadata = _extract_tokenizer_metadata(metadata)
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
            "tokenizer_metadata": tokenizer_metadata,
        })
        converted = 0
        native_iq = native_iq_available()
        base_supported_types = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
            GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL,
        }
        quality_totals = {"mse": 0.0, "rmse": 0.0, "max_abs_error": 0.0, "cosine_similarity": 0.0}
        for name, dims, tensor_type, offset in infos:
            iq_codec = get_iq_codec(tensor_type)
            if iq_codec is not None and not iq_codec.decodable and not native_iq:
                raise ValueError(
                    f"Tensor {name} uses {iq_codec.name}; its verified decoder is not available yet"
                )
            if tensor_type not in base_supported_types and not (iq_codec and native_iq):
                    supported = "F32/F16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL plus native GGML IQ1/IQ2/IQ3/IQ4_XS when available"
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
            elif tensor_type == GGML_TYPE_Q8_0:
                byte_count = (count // _QK) * 34
            elif tensor_type == GGML_TYPE_Q2_K:
                byte_count = (count // _QK_K) * 84
            elif tensor_type == GGML_TYPE_Q3_K:
                byte_count = (count // _QK_K) * 110
            elif tensor_type == GGML_TYPE_Q4_K:
                byte_count = (count // _QK_K) * 144
            elif tensor_type == GGML_TYPE_Q5_K:
                byte_count = (count // _QK_K) * 176
            elif tensor_type == GGML_TYPE_IQ4_NL:
                byte_count = (count // _QK) * 18
            elif iq_codec is not None and native_iq:
                if count % iq_codec.block_values:
                    raise ValueError(f"Tensor {name} is not aligned to {iq_codec.block_values}-value {iq_codec.name} blocks")
                byte_count = (count // iq_codec.block_values) * int(iq_codec.block_bytes or 0)
            else:
                byte_count = (count // _QK_K) * 210
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
            elif tensor_type == GGML_TYPE_Q8_0:
                values = _decode_q8_0(raw, count)
            elif tensor_type == GGML_TYPE_Q2_K:
                values = _decode_q2_k(raw, count)
            elif tensor_type == GGML_TYPE_Q3_K:
                values = _decode_q3_k(raw, count)
            elif tensor_type == GGML_TYPE_Q4_K:
                values = _decode_q4_k(raw, count)
            elif tensor_type == GGML_TYPE_Q5_K:
                values = _decode_q5_k(raw, count)
            elif tensor_type == GGML_TYPE_IQ4_NL:
                values = _decode_iq4_nl(raw, count)
            elif iq_codec is not None and native_iq:
                values = decode_iq_native(tensor_type, raw, count)
            else:
                values = _decode_q6_k(raw, count)
            if target_codec == "sparse05":
                packed, scales = encode_sparse05(values, group_size, threshold=sparse_threshold)
                reconstructed = decode_sparse05(packed, scales, count, group_size)
            else:
                packed, scales = encode_ternary(values, group_size)
                reconstructed = decode_ternary(packed, scales, count, group_size)
            metrics = tensor_quality(values, reconstructed)
            for key in quality_totals:
                quality_totals[key] += float(metrics[key])
            scales_raw = struct.pack(f"<{len(scales)}f", *scales)
            rows = int(dims[0])
            cols = int(count // rows)
            writer.add_tensor(TensorRecord(name, rows, cols, group_size, target_codec, packed, scales_raw))
            converted += 1
        writer.metadata["conversion_quality"] = {
            key: round(value / converted, 8) for key, value in quality_totals.items()
        } if converted else None
        writer.metadata["sparse_threshold"] = sparse_threshold if target_codec == "sparse05" else None
        writer.write(target)
    return {
        "source": str(source),
        "target": str(target),
        "tensor_count": converted,
        "source_bytes": os.path.getsize(source),
        "target_bytes": os.path.getsize(target),
        "codec": target_codec,
        "sparse_threshold": sparse_threshold if target_codec == "sparse05" else None,
        "quality": {key: round(value / converted, 8) for key, value in quality_totals.items()} if converted else None,
    }
