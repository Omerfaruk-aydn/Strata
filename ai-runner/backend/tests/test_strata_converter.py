import struct
from pathlib import Path

from backend.core.strata_ultra import StrataContainerReader, convert_gguf_to_strata


def _write_float_gguf(path: Path, tensor_type: int, values, raw_override=None):
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
    data += raw_override if raw_override is not None else struct.pack(f"<4{'f' if tensor_type == 0 else 'e'}", *values)
    path.write_bytes(data)


def test_f32_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny.gguf"
    target = tmp_path / "tiny.strata"
    _write_float_gguf(source, 0, (-1.0, 0.0, 2.0, -3.0))
    result = convert_gguf_to_strata(source, target, group_size=4)
    assert result["tensor_count"] == 1
    assert 0.0 <= result["quality"]["mse"]
    assert -1.0 <= result["quality"]["cosine_similarity"] <= 1.0
    with StrataContainerReader(target) as reader:
        assert reader.tensor_names() == ["test.weight"]
        assert "conversion_quality" in reader.manifest["metadata"]
        assert "tokenizer_metadata" in reader.manifest["metadata"]


def test_f16_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-f16.gguf"
    target = tmp_path / "tiny-f16.strata"
    _write_float_gguf(source, 1, (-1.0, 0.0, 2.0, -3.0))
    result = convert_gguf_to_strata(source, target, group_size=4)
    assert result["tensor_count"] == 1


def test_q4_0_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-q4.gguf"
    target = tmp_path / "tiny-q4.strata"
    # One Q4_0 block: d=1, all quantized values are zero-point 8 => zero.
    block = struct.pack("<e", 1.0) + bytes([0x88] * 16)
    _write_float_gguf(source, 2, [0] * 32, raw_override=block)
    # The helper declares four dimensions; replace its tensor dimension with 32.
    data = bytearray(source.read_bytes())
    dim_offset = data.index(struct.pack("<Q", 4))
    data[dim_offset:dim_offset + 8] = struct.pack("<Q", 32)
    source.write_bytes(data)
    result = convert_gguf_to_strata(source, target, group_size=32)
    assert result["tensor_count"] == 1


def test_q4_k_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-q4-k.gguf"
    target = tmp_path / "tiny-q4-k.strata"
    # Q4_K type 12, one 256-value super-block. Zero scale/min codes produce zeros.
    block = struct.pack("<ee", 1.0, 0.0) + bytes(12) + bytes([0x00] * 128)
    _write_float_gguf(source, 12, [0] * 32, raw_override=block)
    data = bytearray(source.read_bytes())
    dim_offset = data.index(struct.pack("<Q", 4))
    data[dim_offset:dim_offset + 8] = struct.pack("<Q", 256)
    source.write_bytes(data)
    result = convert_gguf_to_strata(source, target, group_size=256)
    assert result["tensor_count"] == 1


def test_q2_k_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-q2-k.gguf"
    target = tmp_path / "tiny-q2-k.strata"
    # Q2_K type 10: d/dmin + 16 packed scale/min bytes + 64 packed 2-bit bytes.
    block = struct.pack("<ee", 1.0, 0.0) + bytes(16) + bytes(64)
    _write_float_gguf(source, 10, [0] * 32, raw_override=block)
    data = bytearray(source.read_bytes())
    dim_offset = data.index(struct.pack("<Q", 4))
    data[dim_offset:dim_offset + 8] = struct.pack("<Q", 256)
    source.write_bytes(data)
    result = convert_gguf_to_strata(source, target, group_size=256)
    assert result["tensor_count"] == 1


def test_q3_k_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-q3-k.gguf"
    target = tmp_path / "tiny-q3-k.strata"
    # Q3_K type 11: hmask[32] + qs[64] + packed scales[12] + fp16 d.
    block = bytes(32 + 64 + 12) + struct.pack("<e", 0.0)
    _write_float_gguf(source, 11, [0] * 32, raw_override=block)
    data = bytearray(source.read_bytes())
    dim_offset = data.index(struct.pack("<Q", 4))
    data[dim_offset:dim_offset + 8] = struct.pack("<Q", 256)
    source.write_bytes(data)
    result = convert_gguf_to_strata(source, target, group_size=256)
    assert result["tensor_count"] == 1


def test_q5_k_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-q5-k.gguf"
    target = tmp_path / "tiny-q5-k.strata"
    # Q5_K type 13 with zero high plane and zero scale/min codes.
    block = struct.pack("<ee", 1.0, 0.0) + bytes(12) + bytes(128) + bytes(32)
    _write_float_gguf(source, 13, [0] * 32, raw_override=block)
    data = bytearray(source.read_bytes())
    dim_offset = data.index(struct.pack("<Q", 4))
    data[dim_offset:dim_offset + 8] = struct.pack("<Q", 256)
    source.write_bytes(data)
    result = convert_gguf_to_strata(source, target, group_size=256)
    assert result["tensor_count"] == 1


def test_q6_k_gguf_converts_to_strata(tmp_path: Path):
    source = tmp_path / "tiny-q6-k.gguf"
    target = tmp_path / "tiny-q6-k.strata"
    # Q6_K type 14: fp16 d + ql[128] + qh[64] + scales[16].
    block = struct.pack("<e", 1.0) + bytes(128) + bytes(64) + bytes(16)
    _write_float_gguf(source, 14, [0] * 32, raw_override=block)
    data = bytearray(source.read_bytes())
    dim_offset = data.index(struct.pack("<Q", 4))
    data[dim_offset:dim_offset + 8] = struct.pack("<Q", 256)
    source.write_bytes(data)
    result = convert_gguf_to_strata(source, target, group_size=256)
    assert result["tensor_count"] == 1


def test_f32_gguf_can_target_sparse05(tmp_path: Path):
    source = tmp_path / "tiny-sparse.gguf"
    target = tmp_path / "tiny-sparse.strata"
    _write_float_gguf(source, 0, (0.0, 0.0, 2.0, 0.0))
    result = convert_gguf_to_strata(source, target, group_size=4, target_codec="sparse05", sparse_threshold=0.25)
    assert result["codec"] == "sparse05"
    assert result["sparse_threshold"] == 0.25
