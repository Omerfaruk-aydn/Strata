"""Deterministic codec and memory benchmark helpers for Strata Ultra."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .kv_cache import encode_kv, kv_memory_report
from .quality import tensor_quality
from .sparse_codec import decode_sparse05, encode_sparse05
from .ternary import decode_ternary, encode_ternary


@dataclass(frozen=True)
class CodecBenchmark:
    value_count: int
    encode_ms: float
    decode_ms: float
    packed_bytes: int
    compression_vs_f16: float


def run_codec_benchmark(value_count: int = 16_384, group_size: int = 128, sparse_threshold: float = 0.125) -> dict[str, Any]:
    if value_count <= 0:
        raise ValueError("value_count must be positive")
    if group_size <= 0 or value_count % group_size:
        raise ValueError("group_size must be positive and divide value_count")
    if sparse_threshold < 0:
        raise ValueError("sparse_threshold must be non-negative")
    values = [((index % 17) - 8) / 8.0 for index in range(value_count)]
    start = perf_counter()
    packed, scales = encode_ternary(values, group_size)
    encode_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    decoded = decode_ternary(packed, scales, len(values), group_size)
    decode_ms = (perf_counter() - start) * 1000
    sparse_start = perf_counter()
    sparse_packed, sparse_scales = encode_sparse05(values, group_size, threshold=sparse_threshold)
    sparse_encode_ms = (perf_counter() - sparse_start) * 1000
    sparse_start = perf_counter()
    sparse_decoded = decode_sparse05(sparse_packed, sparse_scales, len(values), group_size)
    sparse_decode_ms = (perf_counter() - sparse_start) * 1000
    sparse_quality = tensor_quality(values, sparse_decoded)
    sparse_bytes = len(sparse_packed) + len(sparse_scales) * 4
    packed_bytes = len(packed) + len(scales) * 4
    return {
        "codec": "ternary-q05",
        "value_count": value_count,
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "packed_bytes": packed_bytes,
        "compression_vs_f16": round((1 - packed_bytes / (value_count * 2)) * 100, 2),
        "decode_values_per_second": round(value_count / max(decode_ms / 1000, 1e-9), 2),
        "decoded_values": len(decoded),
        "sparse05": {
            "threshold": sparse_threshold,
            "encode_ms": round(sparse_encode_ms, 3),
            "decode_ms": round(sparse_decode_ms, 3),
            "packed_bytes": sparse_bytes,
            "compression_vs_f16": round((1 - sparse_bytes / (value_count * 2)) * 100, 2),
            "quality": {key: round(value, 8) for key, value in sparse_quality.items()},
        },
        "kv_memory": kv_memory_report(value_count, group_size),
    }
