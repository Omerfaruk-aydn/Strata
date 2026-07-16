import struct
from pathlib import Path

from backend.core.strata_ultra import LowBitMLP, StrataContainerWriter, StrataRuntime, TensorRecord, rms_norm


def _identity(name: str, size: int = 2):
    # Diagonal +1 matrix for a 2x2 tensor.
    return TensorRecord(name, size, size, size * size, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0))


def test_rms_norm_is_finite_and_unit_rms():
    result = rms_norm([3.0, 4.0])
    assert abs(sum(value * value for value in result) / 2 - 1.0) < 1e-4


def test_low_bit_mlp_runs_three_paged_projections(tmp_path: Path):
    target = tmp_path / "mlp.strata"
    writer = StrataContainerWriter()
    for name in ("gate", "up", "down"):
        writer.add_tensor(_identity(name))
    writer.write(target)
    with StrataRuntime(target, 2048, resident_window=1) as runtime:
        output = LowBitMLP(runtime, "gate", "up", "down").forward([1.0, 2.0])
    assert len(output) == 2
    assert all(value >= 0 for value in output)
