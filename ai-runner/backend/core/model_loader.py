"""
AI Runner — Model Loader
GGUF file validation, metadata extraction, and integrity checking.
"""

import os
import struct
import hashlib
from typing import Optional, Dict, Any, BinaryIO
from pydantic import BaseModel


class GGUFMetadata(BaseModel):
    """Extracted metadata from a GGUF file."""
    filename: str
    file_size_bytes: int
    file_size_mb: float
    magic: str = ""
    version: int = 0
    tensor_count: int = 0
    metadata_kv_count: int = 0
    architecture: str = ""
    model_name: str = ""
    size_label: str = ""
    parameter_count: int = 0
    context_length: int = 4096
    embedding_length: int = 0
    block_count: int = 0  # This is the layer count
    head_count: int = 0
    head_count_kv: int = 0
    quantization_version: Optional[int] = None
    file_type: Optional[int] = None
    is_valid: bool = False
    error: Optional[str] = None


# GGUF Magic number
GGUF_MAGIC = b'GGUF'
GGUF_MAGIC_INT = 0x46475547  # "GGUF" in little-endian


def validate_gguf_file(filepath: str) -> GGUFMetadata:
    """
    Validate a GGUF file and extract basic metadata.
    Checks magic number, version, and basic structural integrity.
    """
    filename = os.path.basename(filepath)

    if not os.path.exists(filepath):
        return GGUFMetadata(
            filename=filename,
            file_size_bytes=0,
            file_size_mb=0.0,
            error="Dosya bulunamadı",
            is_valid=False,
        )

    file_size = os.path.getsize(filepath)
    file_size_mb = file_size / (1024 * 1024)

    if file_size < 16:
        return GGUFMetadata(
            filename=filename,
            file_size_bytes=file_size,
            file_size_mb=file_size_mb,
            error="Dosya çok küçük, geçerli bir GGUF dosyası değil",
            is_valid=False,
        )

    try:
        with open(filepath, 'rb') as f:
            # Read magic (4 bytes)
            magic_bytes = f.read(4)
            if magic_bytes != GGUF_MAGIC:
                return GGUFMetadata(
                    filename=filename,
                    file_size_bytes=file_size,
                    file_size_mb=file_size_mb,
                    magic=magic_bytes.hex(),
                    error="Geçersiz GGUF magic number",
                    is_valid=False,
                )

            # Read version (4 bytes, uint32 LE)
            version = struct.unpack('<I', _read_exact(f, 4))[0]

            # Read tensor count (8 bytes, uint64 LE)
            tensor_count = struct.unpack('<Q', _read_exact(f, 8))[0]

            # Read metadata KV count (8 bytes, uint64 LE)
            metadata_kv_count = struct.unpack('<Q', _read_exact(f, 8))[0]

            # Try to extract common metadata keys
            metadata = _extract_metadata(f, metadata_kv_count, version)
            architecture = str(metadata.get("general.architecture", "") or "")

            def arch_value(suffix: str, default=0):
                if architecture:
                    value = metadata.get(f"{architecture}.{suffix}")
                    if value is not None:
                        return value
                return metadata.get(f"llama.{suffix}", default)

            return GGUFMetadata(
                filename=filename,
                file_size_bytes=file_size,
                file_size_mb=round(file_size_mb, 1),
                magic="GGUF",
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=metadata_kv_count,
                architecture=architecture,
                model_name=str(metadata.get("general.name", "") or ""),
                size_label=str(metadata.get("general.size_label", "") or ""),
                parameter_count=int(metadata.get("general.parameter_count", 0) or 0),
                context_length=int(arch_value("context_length", metadata.get("general.context_length", 4096)) or 4096),
                embedding_length=int(arch_value("embedding_length", 0) or 0),
                block_count=int(arch_value("block_count", 0) or 0),
                head_count=int(arch_value("attention.head_count", 0) or 0),
                head_count_kv=int(arch_value("attention.head_count_kv", 0) or 0),
                quantization_version=metadata.get("general.quantization_version"),
                file_type=metadata.get("general.file_type"),
                is_valid=True,
            )

    except Exception as e:
        return GGUFMetadata(
            filename=filename,
            file_size_bytes=file_size,
            file_size_mb=round(file_size_mb, 1),
            error=f"Dosya okunamadı: {str(e)}",
            is_valid=False,
        )


def _read_exact(f: BinaryIO, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Beklenmeyen GGUF üstbilgi sonu")
    return data


def _extract_metadata(f, kv_count: int, version: int) -> Dict[str, Any]:
    """
    Extract metadata key-value pairs from GGUF file header.
    This is a simplified parser that handles common types.
    """
    metadata = {}

    # GGUF KV types
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    if kv_count > 1_000_000:
        raise ValueError("GGUF metadata anahtar sayısı güvenli sınırı aşıyor")
    for _ in range(kv_count):
        # Read key (string: uint64 length + bytes)
        key_len = struct.unpack('<Q', _read_exact(f, 8))[0]
        if key_len > 4096:
            raise ValueError("GGUF metadata anahtarı güvenli sınırı aşıyor")
        key = _read_exact(f, key_len).decode('utf-8', errors='replace')

        # Read value type (uint32)
        value_type = struct.unpack('<I', _read_exact(f, 4))[0]

        # Strict parsing is required here so an unknown or truncated value
        # cannot silently desynchronise all subsequent metadata entries.
        value = _read_gguf_value(f, value_type, strict=True)

        if value is not None:
            metadata[key] = value

    return metadata


def _read_gguf_value(f, value_type: int, *, strict: bool = False):
    """Read a single GGUF value based on its type."""
    try:
        if value_type == 0:   # UINT8
            return struct.unpack('<B', _read_exact(f, 1))[0]
        elif value_type == 1:  # INT8
            return struct.unpack('<b', _read_exact(f, 1))[0]
        elif value_type == 2:  # UINT16
            return struct.unpack('<H', _read_exact(f, 2))[0]
        elif value_type == 3:  # INT16
            return struct.unpack('<h', _read_exact(f, 2))[0]
        elif value_type == 4:  # UINT32
            return struct.unpack('<I', _read_exact(f, 4))[0]
        elif value_type == 5:  # INT32
            return struct.unpack('<i', _read_exact(f, 4))[0]
        elif value_type == 6:  # FLOAT32
            return struct.unpack('<f', _read_exact(f, 4))[0]
        elif value_type == 7:  # BOOL
            return struct.unpack('<B', _read_exact(f, 1))[0] != 0
        elif value_type == 8:  # STRING
            str_len = struct.unpack('<Q', _read_exact(f, 8))[0]
            if str_len > 16 * 1024 * 1024:
                raise ValueError("GGUF metadata metni güvenli sınırı aşıyor")
            value = _read_exact(f, str_len)
            return value.decode('utf-8', errors='replace') if str_len <= 65536 else None
        elif value_type == 9:  # ARRAY
            arr_type = struct.unpack('<I', _read_exact(f, 4))[0]
            arr_len = struct.unpack('<Q', _read_exact(f, 8))[0]
            if arr_len > 100_000_000:
                raise ValueError("GGUF metadata dizisi güvenli sınırı aşıyor")
            _skip_gguf_array(f, arr_type, arr_len)
            return None
        elif value_type == 10:  # UINT64
            return struct.unpack('<Q', _read_exact(f, 8))[0]
        elif value_type == 11:  # INT64
            return struct.unpack('<q', _read_exact(f, 8))[0]
        elif value_type == 12:  # FLOAT64
            return struct.unpack('<d', _read_exact(f, 8))[0]
        else:
            raise ValueError(f"Bilinmeyen GGUF metadata türü: {value_type}")
    except Exception:
        if strict:
            raise
        return None


def _skip_gguf_array(f: BinaryIO, value_type: int, length: int) -> None:
    """Skip an entire GGUF array while preserving the stream position."""
    fixed_sizes = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
        7: 1, 10: 8, 11: 8, 12: 8,
    }
    current = f.tell()
    f.seek(0, os.SEEK_END)
    stream_end = f.tell()
    f.seek(current, os.SEEK_SET)

    def seek_forward(size: int) -> None:
        target = f.tell() + size
        if target > stream_end:
            raise ValueError("GGUF metadata dizisi dosya sınırını aşıyor")
        f.seek(target, os.SEEK_SET)

    if value_type in fixed_sizes:
        seek_forward(fixed_sizes[value_type] * length)
        return
    if value_type == 8:
        for _ in range(length):
            item_length = struct.unpack('<Q', _read_exact(f, 8))[0]
            if item_length > 16 * 1024 * 1024:
                raise ValueError("GGUF metadata dizi öğesi güvenli sınırı aşıyor")
            seek_forward(item_length)
        return
    if value_type == 9:
        raise ValueError("İç içe GGUF metadata dizileri desteklenmiyor")
    raise ValueError(f"Bilinmeyen GGUF metadata türü: {value_type}")


def compute_file_checksum(
    filepath: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> Optional[str]:
    """
    Compute checksum of a file for integrity verification.
    Used for model file corruption detection.
    """
    if not os.path.exists(filepath):
        return None

    try:
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None
