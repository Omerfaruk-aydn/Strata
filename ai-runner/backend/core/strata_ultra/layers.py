"""Reference low-bit neural layers built on the Strata pager runtime."""

from __future__ import annotations

import math

from .executor import StrataRuntime, matvec


def rms_norm(values: list[float], epsilon: float = 1e-5) -> list[float]:
    if not values:
        raise ValueError("RMSNorm input cannot be empty")
    mean_square = sum(value * value for value in values) / len(values)
    scale = 1.0 / math.sqrt(mean_square + epsilon)
    return [value * scale for value in values]


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


class LowBitMLP:
    """SwiGLU-style MLP using three pager-backed Q0.5 projections."""

    def __init__(self, runtime: StrataRuntime, gate: str, up: str, down: str):
        self.runtime = runtime
        self.gate = gate
        self.up = up
        self.down = down

    def forward(self, values: list[float]) -> list[float]:
        normalized = rms_norm(values)
        gate = matvec(self.runtime.pager.get(self.gate), normalized)
        up = matvec(self.runtime.pager.get(self.up), normalized)
        hidden = [silu(gate[index]) * up[index] for index in range(len(gate))]
        return matvec(self.runtime.pager.get(self.down), hidden)
