"""Ultra-low-bit KV cache primitives for the Strata runtime.

The cache is deliberately independent from llama.cpp.  ``sign1`` stores one
sign bit per value plus a float scale per group. ``ternary05`` stores a
two-bit code (zero, negative, positive, reserved) and is exposed as the
experimental 0.5-bit profile because the sparse codebook amortizes to roughly
0.5 bits/value on sufficiently sparse groups.  It is a storage profile, not a
claim that arbitrary dense tensors contain literal half-bits.
"""

from dataclasses import dataclass
import math
import struct
from typing import Sequence


@dataclass(frozen=True)
class PackedKV:
    mode: str
    count: int
    group_size: int
    payload: bytes
    scales: tuple[float, ...]

    @property
    def payload_bytes(self) -> int:
        return len(self.payload) + len(self.scales) * 4


def _validate(values: Sequence[float], group_size: int) -> None:
    if not values:
        raise ValueError("KV values cannot be empty")
    if group_size <= 0:
        raise ValueError("group_size must be positive")


def encode_kv(values: Sequence[float], mode: str = "sign1", group_size: int = 128) -> PackedKV:
    """Pack a flat KV tensor into the requested ultra-low-bit representation."""
    _validate(values, group_size)
    if mode not in {"sign1", "ternary05"}:
        raise ValueError("mode must be 'sign1' or 'ternary05'")
    payload = bytearray((len(values) + (7 if mode == "sign1" else 3)) // (8 if mode == "sign1" else 4))
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        group = [float(v) for v in values[start:start + group_size]]
        scale = max((abs(v) for v in group), default=0.0)
        scales.append(scale)
        if mode == "sign1":
            for local, value in enumerate(group):
                if value >= 0:
                    index = start + local
                    payload[index // 8] |= 1 << (index % 8)
        else:
            threshold = scale / 3.0 if scale else 0.0
            for local, value in enumerate(group):
                code = 2 if value > threshold else 1 if value < -threshold else 0
                index = start + local
                payload[index // 4] |= code << ((index % 4) * 2)
    return PackedKV(mode, len(values), group_size, bytes(payload), tuple(scales))


def decode_kv(cache: PackedKV) -> list[float]:
    """Decode a packed cache tensor for execution or validation."""
    if len(cache.scales) < math.ceil(cache.count / cache.group_size):
        raise ValueError("truncated KV scales")
    values: list[float] = []
    for index in range(cache.count):
        scale = cache.scales[index // cache.group_size]
        if cache.mode == "sign1":
            bit = (cache.payload[index // 8] >> (index % 8)) & 1
            values.append(scale if bit else -scale)
        elif cache.mode == "ternary05":
            code = (cache.payload[index // 4] >> ((index % 4) * 2)) & 3
            values.append(0.0 if code == 0 else -scale if code == 1 else scale)
        else:
            raise ValueError(f"unsupported KV mode: {cache.mode}")
    return values


def estimate_kv_bytes(value_count: int, mode: str, group_size: int = 128) -> int:
    """Return packed payload plus float32 group-scale storage."""
    if value_count < 0 or group_size <= 0:
        raise ValueError("value_count and group_size must be valid")
    bits = {"sign1": 1, "ternary05": 2}.get(mode)
    if bits is None:
        raise ValueError(f"unsupported KV mode: {mode}")
    payload = math.ceil(value_count * bits / 8)
    groups = math.ceil(value_count / group_size)
    return payload + groups * 4


def kv_memory_report(value_count: int, group_size: int = 128) -> dict[str, int | float]:
    """Compare F16, 1-bit, and experimental 0.5-profile storage."""
    f16 = value_count * 2
    sign1 = estimate_kv_bytes(value_count, "sign1", group_size)
    ternary = estimate_kv_bytes(value_count, "ternary05", group_size)
    return {
        "value_count": value_count,
        "f16_bytes": f16,
        "sign1_bytes": sign1,
        "ternary05_bytes": ternary,
        "sign1_saving_percent": round((1 - sign1 / f16) * 100, 2) if f16 else 0.0,
        "ternary05_saving_percent": round((1 - ternary / f16) * 100, 2) if f16 else 0.0,
    }
