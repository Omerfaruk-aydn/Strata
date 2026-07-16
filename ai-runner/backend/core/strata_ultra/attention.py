"""Reference scaled dot-product attention backed by UltraKVCache."""

from __future__ import annotations

import math

from .kv_runtime import UltraKVCache


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


class LowBitAttention:
    """Single-head reference attention with quantized rolling K/V history."""

    def __init__(self, width: int, capacity_tokens: int, mode: str = "sign1", sparse_threshold: float = 0.125, backend: str = "auto"):
        self.width = width
        self.keys = UltraKVCache(width, capacity_tokens, mode, sparse_threshold=sparse_threshold, backend=backend)
        self.values = UltraKVCache(width, capacity_tokens, mode, sparse_threshold=sparse_threshold, backend=backend)

    def step(self, query: list[float], key: list[float], value: list[float]) -> list[float]:
        if len(query) != self.width or len(key) != self.width or len(value) != self.width:
            raise ValueError("query, key, and value must match attention width")
        self.keys.append(key)
        self.values.append(value)
        key_values = self.keys.values()
        value_values = self.values.values()
        token_count = self.keys.token_count
        scores = [
            sum(query[index] * key_values[token * self.width + index] for index in range(self.width))
            / math.sqrt(self.width)
            for token in range(token_count)
        ]
        weights = _softmax(scores)
        return [
            sum(weights[token] * value_values[token * self.width + index] for token in range(token_count))
            for index in range(self.width)
        ]
