import struct
from pathlib import Path

from backend.core.strata_ultra import StrataContainerWriter, StrataRuntime, TensorRecord, matmul, matvec


def _record():
    # 2x4 tensor: first row [-1, 0, +1, -1], second row [0, +1, 0, +1].
    payload = bytes([0b01_10_00_01, 0b10_00_10_00])
    return TensorRecord("dense.weight", 2, 4, 4, "ternary-q05", payload, struct.pack("<2f", 1.0, 1.0))


def test_q05_matvec_runs_without_external_runtime():
    result = matvec(_record(), [1.0, 2.0, 3.0, 4.0])
    assert result == [-2.0, 6.0]


def test_strata_runtime_uses_pager(tmp_path: Path):
    target = tmp_path / "runtime.strata"
    writer = StrataContainerWriter({"profile": "STRATA-Q0.5"})
    writer.add_tensor(_record())
    writer.write(target)
    with StrataRuntime(target, memory_budget_bytes=1024, resident_window=1) as runtime:
        assert runtime.tensor_matvec("dense.weight", [1.0, 2.0, 3.0, 4.0]) == [-2.0, 6.0]
        assert runtime.pager.resident_pages == 1


def test_matmul_decodes_tensor_for_a_batch():
    result = matmul(_record(), [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]])
    assert result == [[-2.0, 6.0], [-1.0, 2.0]]
