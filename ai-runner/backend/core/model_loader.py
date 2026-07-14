"""
AI Runner — Model Loader
GGUF file validation, metadata extraction, and integrity checking.
"""

import os
import struct
import hashlib
from typing import Optional, Dict, Any
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
    context_length: int = 4096
    embedding_length: int = 0
    block_count: int = 0  # This is the layer count
    head_count: int = 0
    head_count_kv: int = 0
    quantization_version: Optional[int] = None
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
            version = struct.unpack('<I', f.read(4))[0]

            # Read tensor count (8 bytes, uint64 LE)
            tensor_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata KV count (8 bytes, uint64 LE)
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Try to extract common metadata keys
            metadata = _extract_metadata(f, metadata_kv_count, version)

            return GGUFMetadata(
                filename=filename,
                file_size_bytes=file_size,
                file_size_mb=round(file_size_mb, 1),
                magic="GGUF",
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=metadata_kv_count,
                architecture=metadata.get("general.architecture", ""),
                context_length=metadata.get("llama.context_length",
                               metadata.get("general.context_length", 4096)),
                embedding_length=metadata.get("llama.embedding_length", 0),
                block_count=metadata.get("llama.block_count", 0),
                head_count=metadata.get("llama.attention.head_count", 0),
                head_count_kv=metadata.get("llama.attention.head_count_kv", 0),
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

    try:
        for _ in range(min(kv_count, 100)):  # Limit to prevent infinite loops
            # Read key (string: uint64 length + bytes)
            key_len = struct.unpack('<Q', f.read(8))[0]
            if key_len > 256:  # Sanity check
                break
            key = f.read(key_len).decode('utf-8', errors='replace')

            # Read value type (uint32)
            value_type = struct.unpack('<I', f.read(4))[0]

            # Read value based on type
            value = _read_gguf_value(f, value_type)

            if value is not None:
                metadata[key] = value

    except Exception:
        pass  # Best-effort metadata extraction

    return metadata


def _read_gguf_value(f, value_type: int):
    """Read a single GGUF value based on its type."""
    try:
        if value_type == 0:   # UINT8
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == 1:  # INT8
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == 2:  # UINT16
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == 3:  # INT16
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == 4:  # UINT32
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == 5:  # INT32
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == 6:  # FLOAT32
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == 7:  # BOOL
            return struct.unpack('<B', f.read(1))[0] != 0
        elif value_type == 8:  # STRING
            str_len = struct.unpack('<Q', f.read(8))[0]
            if str_len > 4096:
                f.seek(str_len, 1)
                return None
            return f.read(str_len).decode('utf-8', errors='replace')
        elif value_type == 9:  # ARRAY
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            # Skip arrays for now (they can be very large)
            for _ in range(min(arr_len, 1000)):
                _read_gguf_value(f, arr_type)
            return None
        elif value_type == 10:  # UINT64
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == 11:  # INT64
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == 12:  # FLOAT64
            return struct.unpack('<d', f.read(8))[0]
        else:
            return None
    except Exception:
        return None


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

    hasher = hashlib.new(algorithm)
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None
