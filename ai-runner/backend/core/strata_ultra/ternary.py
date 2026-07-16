"""Deterministic ternary codec used by the experimental STRATA-Q0.5 mode.

Each value is represented by two packed bits: 00 = zero, 01 = -1, 10 = +1.
Group scales are kept separately as float32 values.  The effective storage is
below one byte per weight for useful group sizes, while preserving a reversible
decode path for runtime testing.
"""

from array import array
import math
from typing import Iterable, Sequence


def encode_ternary(values: Iterable[float], group_size: int = 128) -> tuple[bytes, list[float]]:
    values = [float(v) for v in values]
    if not values:
        raise ValueError("values cannot be empty")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    packed = bytearray((len(values) + 3) // 4)
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        scale = max((abs(v) for v in group), default=0.0)
        scales.append(scale)
        threshold = scale / 3.0 if scale else 0.0
        for local, value in enumerate(group):
            code = 2 if value > threshold else 1 if value < -threshold else 0
            index = start + local
            packed[index // 4] |= code << ((index % 4) * 2)
    return bytes(packed), scales


def decode_ternary(packed: bytes, scales: Sequence[float], count: int, group_size: int = 128) -> list[float]:
    if count < 0 or group_size <= 0:
        raise ValueError("count and group_size must be valid")
    if len(packed) < math.ceil(count / 4) or len(scales) < math.ceil(count / group_size):
        raise ValueError("truncated ternary tensor")
    result: list[float] = []
    for index in range(count):
        code = (packed[index // 4] >> ((index % 4) * 2)) & 0x03
        scale = float(scales[index // group_size])
        result.append(0.0 if code == 0 else -scale if code == 1 else scale)
    return result
