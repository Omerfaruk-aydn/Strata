import struct
from pathlib import Path

from backend.core.strata_ultra import StrataContainerWriter, StrataRuntime, TensorRecord, encode_sparse05


def test_sparse05_tensor_runs_through_container_and_runtime(tmp_path: Path):
    values = [0.0] * 4
    values[1] = 2.0
    payload, scales = encode_sparse05(values, group_size=4)
    target = tmp_path / "sparse.strata"
    writer = StrataContainerWriter()
    writer.add_tensor(TensorRecord("sparse", 1, 4, 4, "sparse05", payload, struct.pack(f"<{len(scales)}f", *scales)))
    writer.write(target)
    with StrataRuntime(target, 1024) as runtime:
        assert runtime.tensor_matvec("sparse", [1.0, 3.0, 1.0, 1.0]) == [6.0]
