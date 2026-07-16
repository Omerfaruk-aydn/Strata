import struct
from pathlib import Path

from backend.core.strata_ultra import LowBitTransformerBlock, StrataContainerWriter, StrataRuntime, TensorRecord


def _identity(name: str):
    return TensorRecord(name, 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0))


def test_low_bit_transformer_block_runs_attention_mlp_and_residual(tmp_path: Path):
    target = tmp_path / "block.strata"
    writer = StrataContainerWriter()
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        writer.add_tensor(_identity(name))
    writer.write(target)
    with StrataRuntime(target, 4096, resident_window=2) as runtime:
        block = LowBitTransformerBlock(
            runtime, q_proj="q", k_proj="k", v_proj="v", o_proj="o",
            gate_proj="gate", up_proj="up", down_proj="down", width=2,
            context_capacity=4,
        )
        output = block.step([1.0, 0.5])
    assert len(output) == 2
    assert all(value == value for value in output)
