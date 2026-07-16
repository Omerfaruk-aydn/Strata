"""GGUF validation and quantization-matrix tests."""

from __future__ import annotations

import hashlib
import io
import struct

import pytest

from backend.core.model_loader import (
    _read_gguf_value,
    compute_file_checksum,
    validate_gguf_file,
)
from backend.core.quant_matrix import (
    QUANT_MATRIX,
    estimate_file_size_gb,
    get_all_quants,
    get_quant_info,
    get_quant_recommendation_text,
)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def gguf_metadata_entry(key: str, value_type: int, value: bytes) -> bytes:
    return gguf_string(key) + struct.pack("<I", value_type) + value


def test_validate_gguf_extracts_common_metadata(tmp_path):
    entries = [
        gguf_metadata_entry("general.architecture", 8, gguf_string("llama")),
        gguf_metadata_entry("llama.context_length", 4, struct.pack("<I", 8192)),
        gguf_metadata_entry("llama.embedding_length", 4, struct.pack("<I", 4096)),
        gguf_metadata_entry("llama.block_count", 4, struct.pack("<I", 32)),
        gguf_metadata_entry("llama.attention.head_count", 4, struct.pack("<I", 32)),
        gguf_metadata_entry("llama.attention.head_count_kv", 4, struct.pack("<I", 8)),
    ]
    path = tmp_path / "valid.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 291)
        + struct.pack("<Q", len(entries))
        + b"".join(entries)
    )

    metadata = validate_gguf_file(str(path))
    assert metadata.is_valid is True
    assert metadata.magic == "GGUF"
    assert metadata.version == 3
    assert metadata.tensor_count == 291
    assert metadata.architecture == "llama"
    assert metadata.context_length == 8192
    assert metadata.embedding_length == 4096
    assert metadata.block_count == 32
    assert metadata.head_count == 32
    assert metadata.head_count_kv == 8


def test_validate_gguf_uses_architecture_specific_keys_after_large_array(tmp_path):
    token_ids = list(range(1500))
    array_value = (
        struct.pack("<I", 4)
        + struct.pack("<Q", len(token_ids))
        + b"".join(struct.pack("<I", item) for item in token_ids)
    )
    entries = [
        gguf_metadata_entry("general.architecture", 8, gguf_string("qwen2")),
        gguf_metadata_entry("tokenizer.test.ids", 9, array_value),
        gguf_metadata_entry("general.name", 8, gguf_string("Qwen Test")),
        gguf_metadata_entry("general.size_label", 8, gguf_string("100B")),
        gguf_metadata_entry("general.parameter_count", 10, struct.pack("<Q", 100_000_000_000)),
        gguf_metadata_entry("general.file_type", 4, struct.pack("<I", 15)),
        gguf_metadata_entry("general.quantization_version", 4, struct.pack("<I", 2)),
        gguf_metadata_entry("qwen2.context_length", 4, struct.pack("<I", 32768)),
        gguf_metadata_entry("qwen2.embedding_length", 4, struct.pack("<I", 8192)),
        gguf_metadata_entry("qwen2.block_count", 4, struct.pack("<I", 96)),
        gguf_metadata_entry("qwen2.attention.head_count", 4, struct.pack("<I", 64)),
        gguf_metadata_entry("qwen2.attention.head_count_kv", 4, struct.pack("<I", 8)),
    ]
    path = tmp_path / "qwen-100b.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1000)
        + struct.pack("<Q", len(entries))
        + b"".join(entries)
    )

    metadata = validate_gguf_file(str(path))
    assert metadata.is_valid is True
    assert metadata.architecture == "qwen2"
    assert metadata.model_name == "Qwen Test"
    assert metadata.size_label == "100B"
    assert metadata.parameter_count == 100_000_000_000
    assert metadata.context_length == 32768
    assert metadata.embedding_length == 8192
    assert metadata.block_count == 96
    assert metadata.head_count == 64
    assert metadata.head_count_kv == 8
    assert metadata.file_type == 15
    assert metadata.quantization_version == 2


def test_validate_gguf_error_paths(tmp_path):
    missing = validate_gguf_file(str(tmp_path / "missing.gguf"))
    assert missing.is_valid is False
    assert "bulunamadı" in missing.error

    tiny_path = tmp_path / "tiny.gguf"
    tiny_path.write_bytes(b"GGUF")
    assert "çok küçük" in validate_gguf_file(str(tiny_path)).error

    invalid_path = tmp_path / "invalid.gguf"
    invalid_path.write_bytes(b"HTML" + b"x" * 20)
    invalid = validate_gguf_file(str(invalid_path))
    assert invalid.is_valid is False
    assert invalid.magic == "48544d4c"

    truncated_path = tmp_path / "truncated.gguf"
    truncated_path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"x" * 8)
    assert "okunamadı" in validate_gguf_file(str(truncated_path)).error

    oversized_array = tmp_path / "oversized-array.gguf"
    broken_entry = gguf_metadata_entry(
        "tokenizer.ids",
        9,
        struct.pack("<I", 4) + struct.pack("<Q", 100) + struct.pack("<I", 1),
    )
    oversized_array.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
        + struct.pack("<Q", 1) + broken_entry
    )
    broken = validate_gguf_file(str(oversized_array))
    assert broken.is_valid is False
    assert "dosya sınırını" in (broken.error or "")


@pytest.mark.parametrize(
    ("value_type", "encoded", "expected"),
    [
        (0, struct.pack("<B", 255), 255),
        (1, struct.pack("<b", -5), -5),
        (2, struct.pack("<H", 65530), 65530),
        (3, struct.pack("<h", -123), -123),
        (4, struct.pack("<I", 123_456), 123_456),
        (5, struct.pack("<i", -123_456), -123_456),
        (6, struct.pack("<f", 1.5), 1.5),
        (7, struct.pack("<B", 1), True),
        (8, gguf_string("hello"), "hello"),
        (10, struct.pack("<Q", 2**40), 2**40),
        (11, struct.pack("<q", -(2**40)), -(2**40)),
        (12, struct.pack("<d", 3.25), 3.25),
    ],
)
def test_read_gguf_scalar_types(value_type, encoded, expected):
    assert _read_gguf_value(io.BytesIO(encoded), value_type) == expected


def test_read_gguf_array_unknown_and_truncated_values():
    array = struct.pack("<I", 4) + struct.pack("<Q", 3) + struct.pack("<III", 1, 2, 3)
    assert _read_gguf_value(io.BytesIO(array), 9) is None
    assert _read_gguf_value(io.BytesIO(b"anything"), 99) is None
    assert _read_gguf_value(io.BytesIO(b""), 4) is None


def test_checksum_success_missing_and_invalid_algorithm(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"checksum payload")
    assert compute_file_checksum(str(path), chunk_size=3) == hashlib.sha256(
        b"checksum payload"
    ).hexdigest()
    assert compute_file_checksum(str(tmp_path / "missing.bin")) is None
    assert compute_file_checksum(str(path), algorithm="not-a-hash") is None


def test_quant_matrix_queries_and_recommendations():
    assert get_all_quants() is QUANT_MATRIX
    assert get_quant_info("Q4_K_M").bits_per_weight == 4.8
    assert get_quant_info("unknown") is None

    turkish = get_quant_recommendation_text("Q4_K_M", "Dengeli seçim.", "tr")
    english = get_quant_recommendation_text("Q4_K_M", "Balanced choice.", "en")
    assert "kalite kaybı" in turkish
    assert "quality loss" in english
    assert get_quant_recommendation_text("unknown", "fallback") == "fallback"

    assert estimate_file_size_gb(7_000_000_000, "Q4_K_M") == 3.9
    assert estimate_file_size_gb(7_000_000_000, "unknown") == 3.9

