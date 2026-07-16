"""Deterministic codec and memory benchmark helpers for Strata Ultra."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .kv_cache import encode_kv, kv_memory_report
from .ternary import decode_ternary, encode_ternary


@dataclass(frozen=True)
class CodecBenchmark:
    value_count: int
    encode_ms: float
    decode_ms: float
    packed_bytes: int
    compression_vs_f16: float


def run_codec_benchmark(value_count: int = 16_384, group_size: int = 128) -> dict[str, Any]:
    if value_count <= 0:
        raise ValueError("value_count must be positive")
    values = [((index % 17) - 8) / 8.0 for index in range(value_count)]
    start = perf_counter()
    packed, scales = encode_ternary(values, group_size)
    encode_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    decoded = decode_ternary(packed, scales, len(values), group_size)
    decode_ms = (perf_counter() - start) * 1000
    return {
        "codec": "ternary-q05",
        "value_count": value_count,
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "packed_bytes": len(packed) + len(scales) * 4,
        "compression_vs_f16": round((1 - (len(packed) + len(scales) * 4) / (value_count * 2)) * 100, 2),
        "decoded_values": len(decoded),
        "kv_memory": kv_memory_report(value_count, group_size),
    }
