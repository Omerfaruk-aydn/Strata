import struct
from pathlib import Path

from backend.core.strata_ultra import StrataContainerReader, convert_gguf_to_strata


def _write_float_gguf(path: Path, tensor_type: int, values):
    name = b"test.weight"
    key = b"general.architecture"
    value = b"llama"
    # GGUF v3, one tensor, one string metadata entry.
    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1))
    data += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
    data += struct.pack("<Q", len(value)) + value
    data += struct.pack("<Q", len(name)) + name
    data += struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<IQ", tensor_type, 0)
    data += b"\x00" * ((32 - len(data) % 32) % 32)
    data += struct.pack(f"<4{'f' if tensor_type == 0 else 'e'}", *values)
    path.write_bytes(data)


def test_f32_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny.gguf"
    target = tmp_path / "tiny.strata"
    _write_float_gguf(source, 0, (-1.0, 0.0, 2.0, -3.0))
    result = convert_gguf_to_strata(source, target, group_size=4)
    assert result["tensor_count"] == 1
    with StrataContainerReader(target) as reader:
        assert reader.tensor_names() == ["test.weight"]


def test_f16_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-f16.gguf"
    target = tmp_path / "tiny-f16.strata"
    _write_float_gguf(source, 1, (-1.0, 0.0, 2.0, -3.0))
    result = convert_gguf_to_strata(source, target, group_size=4)
    assert result["tensor_count"] == 1
