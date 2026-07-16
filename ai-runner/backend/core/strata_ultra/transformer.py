"""Reference low-bit transformer block composed from Strata primitives."""

from __future__ import annotations

from .attention import LowBitAttention
from .executor import StrataRuntime
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

    @classmethod
    def from_layout(cls, runtime: StrataRuntime, layout: dict[str, str], *, width: int, context_capacity: int, kv_mode: str = "sign1") -> "LowBitTransformerBlock":
        required = {"q", "k", "v", "o", "gate", "up", "down"}
        missing = sorted(required - set(layout))
        if missing:
            raise ValueError(f"incomplete transformer layout; missing: {', '.join(missing)}")
        return cls(
            runtime, q_proj=layout["q"], k_proj=layout["k"], v_proj=layout["v"], o_proj=layout["o"],
            gate_proj=layout["gate"], up_proj=layout["up"], down_proj=layout["down"], width=width,
            context_capacity=context_capacity, kv_mode=kv_mode,
        )

    def step(self, hidden: list[float]) -> list[float]:
        normalized = rms_norm(hidden)
        query = self.runtime.tensor_matvec(self.q_proj, normalized)
        key = self.runtime.tensor_matvec(self.k_proj, normalized)
        value = self.runtime.tensor_matvec(self.v_proj, normalized)
        attended = self.attention.step(query, key, value)
        projected = self.runtime.tensor_matvec(self.o_proj, attended)
        residual = [hidden[index] + projected[index] for index in range(len(hidden))]
        mlp_output = self.mlp.forward(residual)
        return [residual[index] + mlp_output[index] for index in range(len(residual))]


class LowBitTransformer:
    """Sequential multi-block runtime with a shared bounded pager."""

    def __init__(self, blocks: list[LowBitTransformerBlock]):
        if not blocks:
            raise ValueError("transformer must contain at least one block")
        self.blocks = blocks

    def step(self, hidden: list[float]) -> list[float]:
        output = list(hidden)
        for block in self.blocks:
            output = block.step(output)
        return output
