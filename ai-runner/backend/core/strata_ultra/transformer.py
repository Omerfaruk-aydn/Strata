"""Reference low-bit transformer block composed from Strata primitives."""

from __future__ import annotations

from .attention import LowBitAttention
from .executor import StrataRuntime, matvec
from .layers import LowBitMLP, rms_norm


class LowBitTransformerBlock:
    """Single-head transformer block with pager-backed Q0.5 projections."""

    def __init__(
        self,
        runtime: StrataRuntime,
        *,
        q_proj: str,
        k_proj: str,
        v_proj: str,
        o_proj: str,
        gate_proj: str,
        up_proj: str,
        down_proj: str,
        width: int,
        context_capacity: int,
        kv_mode: str = "sign1",
    ):
        self.runtime = runtime
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.attention = LowBitAttention(width, context_capacity, kv_mode)
        self.mlp = LowBitMLP(runtime, gate_proj, up_proj, down_proj)

    def step(self, hidden: list[float]) -> list[float]:
        normalized = rms_norm(hidden)
        query = matvec(self.runtime.pager.get(self.q_proj), normalized)
        key = matvec(self.runtime.pager.get(self.k_proj), normalized)
        value = matvec(self.runtime.pager.get(self.v_proj), normalized)
        attended = self.attention.step(query, key, value)
        projected = matvec(self.runtime.pager.get(self.o_proj), attended)
        residual = [hidden[index] + projected[index] for index in range(len(hidden))]
        mlp_output = self.mlp.forward(residual)
        return [residual[index] + mlp_output[index] for index in range(len(residual))]
