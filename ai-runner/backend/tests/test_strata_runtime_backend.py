import struct
from pathlib import Path

from backend.core.strata_ultra import StrataContainerWriter, StrataRuntime, TensorRecord, matmul


def test_runtime_auto_backend_matches_reference(tmp_path: Path):
    target = tmp_path / "backend.strata"
    writer = StrataContainerWriter()
    writer.add_tensor(TensorRecord("x", 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0)))
    writer.write(target)
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    with StrataRuntime(target, 2048, backend="auto") as runtime:
        assert runtime.tensor_matmul("x", matrix) == matmul(runtime._records["x"], matrix)
