"""Sparse05 variable-length codec for genuinely sparse ultra-low-bit tensors."""

from __future__ import annotations

import struct
from typing import Sequence


def _put_varint(value: int) -> bytes:
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _get_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value, shift = 0, 0
    while offset < len(payload):
        byte = payload[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7
        if shift > 35:
            break
    raise ValueError("truncated sparse05 varint")


def encode_sparse05(values: Sequence[float], group_size: int = 128, threshold: float = 0.0) -> tuple[bytes, tuple[float, ...]]:
    if not values or group_size <= 0:
        raise ValueError("values and group_size must be valid")
    payload = bytearray()
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        group = [float(value) for value in values[start:start + group_size]]
        scale = max((abs(value) for value in group), default=0.0)
        scales.append(scale)
        nonzero = [index for index, value in enumerate(group) if abs(value) > threshold]
        payload.extend(struct.pack("<H", len(nonzero)))
        previous = -1
        for index in nonzero:
            delta = index - previous - 1
            sign = 1 if group[index] >= 0 else 0
            payload.extend(_put_varint((delta << 1) | sign))
            previous = index
    return bytes(payload), tuple(scales)


def decode_sparse05(payload: bytes, scales: Sequence[float], count: int, group_size: int = 128) -> list[float]:
    if count <= 0 or group_size <= 0:
        raise ValueError("count and group_size must be valid")
    groups = (count + group_size - 1) // group_size
    if len(scales) < groups:
        raise ValueError("truncated sparse05 scales")
    output = [0.0] * count
    offset = 0
    for group_index, start in enumerate(range(0, count, group_size)):
        if offset + 2 > len(payload):
            raise ValueError("truncated sparse05 group")
        nonzero = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        position = -1
        group_end = min(group_size, count - start)
        for _ in range(nonzero):
            encoded, offset = _get_varint(payload, offset)
            position += (encoded >> 1) + 1
            if position >= group_end:
                raise ValueError("sparse05 position outside group")
            output[start + position] = float(scales[group_index]) if encoded & 1 else -float(scales[group_index])
    if offset != len(payload):
        raise ValueError("trailing sparse05 payload bytes")
    return output
