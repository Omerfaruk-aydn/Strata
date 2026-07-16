import struct
from pathlib import Path

from backend.core.strata_ultra import LowBitTransformer, LowBitTransformerBlock, StrataContainerWriter, StrataRuntime, TensorRecord


def _identity(name: str):
    return TensorRecord(name, 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0))


def _block(runtime, prefix):
    names = {part: f"{prefix}.{part}" for part in ("q", "k", "v", "o", "gate", "up", "down")}
    return LowBitTransformerBlock(
        runtime, q_proj=names["q"], k_proj=names["k"], v_proj=names["v"], o_proj=names["o"],
        gate_proj=names["gate"], up_proj=names["up"], down_proj=names["down"], width=2,
        context_capacity=4,
    )


def test_multi_block_transformer_runs_with_shared_pager(tmp_path: Path):
    target = tmp_path / "model.strata"
    writer = StrataContainerWriter()
    for prefix in ("block0", "block1"):
        for part in ("q", "k", "v", "o", "gate", "up", "down"):
            writer.add_tensor(_identity(f"{prefix}.{part}"))
    writer.write(target)
    with StrataRuntime(target, 4096, resident_window=2) as runtime:
        model = LowBitTransformer([_block(runtime, "block0"), _block(runtime, "block1")])
        output = model.step([1.0, 0.5])
        assert len(output) == 2
        assert runtime.pager.resident_pages <= 2
