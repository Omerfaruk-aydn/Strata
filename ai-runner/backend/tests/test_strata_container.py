from pathlib import Path

import pytest

from backend.core.strata_ultra import StrataContainerReader, StrataContainerWriter, TensorRecord


def test_strata_container_round_trip(tmp_path: Path):
    target = tmp_path / "demo.strata"
    writer = StrataContainerWriter({"model": "demo", "profile": "STRATA-Q0.5"})
    writer.add_tensor(TensorRecord("layer.0.weight", 2, 4, 128, "ternary-q05", b"payload", b"scales"))
    writer.write(target)
    with StrataContainerReader(target) as reader:
        assert reader.manifest["metadata"]["profile"] == "STRATA-Q0.5"
        assert reader.tensor_names() == ["layer.0.weight"]
        records = list(reader.read_tensors())
    assert records[0].payload == b"payload"


def test_strata_container_rejects_checksum_corruption(tmp_path: Path):
    target = tmp_path / "broken.strata"
    writer = StrataContainerWriter()
    writer.add_tensor(TensorRecord("x", 1, 1, 1, "ternary-q05", b"a", b"b"))
    writer.write(target)
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(data)
    with StrataContainerReader(target) as reader:
        with pytest.raises(ValueError, match="checksum"):
            list(reader.read_tensors())
