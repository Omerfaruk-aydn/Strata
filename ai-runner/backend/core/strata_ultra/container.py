"""Versioned, checksummed ``.strata`` tensor container.

The container is intentionally small and stream-friendly.  A JSON manifest
describes tensors and a sequence of binary records stores their packed payload
and float32 group scales.  It is a foundation for GGUF conversion, not a
claim that every GGUF architecture is executable yet.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from .format import STRATA_MAGIC, STRATA_FORMAT_VERSION

CONTAINER_MAGIC = b"STRATA-CONT\x00"
_PREFIX = struct.Struct("<12sBI")
_RECORD = struct.Struct("<II32s")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    rows: int
    cols: int
    group_size: int
    codec: str
    payload: bytes
    scales: bytes


class StrataContainerWriter:
    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = dict(metadata or {})
        self._records: list[TensorRecord] = []

    def add_tensor(self, record: TensorRecord) -> None:
        if not record.name or record.rows <= 0 or record.cols <= 0 or record.group_size <= 0:
            raise ValueError("invalid tensor record")
        if record.codec not in {"ternary-q05", "sparse05"}:
            raise ValueError(f"unsupported Strata codec: {record.codec}")
        if any(existing.name == record.name for existing in self._records):
            raise ValueError(f"duplicate tensor: {record.name}")
        self._records.append(record)

    def _manifest(self) -> bytes:
        manifest = {
            "format": "strata-container",
            "version": STRATA_FORMAT_VERSION,
            "metadata": self.metadata,
            "tensors": [
                {
                    "name": r.name,
                    "rows": r.rows,
                    "cols": r.cols,
                    "group_size": r.group_size,
                    "codec": r.codec,
                    "payload_bytes": len(r.payload),
                    "scales_bytes": len(r.scales),
                }
                for r in self._records
            ],
        }
        return json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def write(self, target: str | Path) -> Path:
        if not self._records:
            raise ValueError("container must contain at least one tensor")
        target = Path(target)
        manifest = self._manifest()
        with target.open("wb") as stream:
            stream.write(_PREFIX.pack(CONTAINER_MAGIC, STRATA_FORMAT_VERSION, len(manifest)))
            stream.write(manifest)
            for record in self._records:
                digest = hashlib.sha256(record.payload + record.scales).digest()
                stream.write(_RECORD.pack(len(record.payload), len(record.scales), digest))
                stream.write(record.payload)
                stream.write(record.scales)
        return target


class StrataContainerReader:
    def __init__(self, source: str | Path | BinaryIO) -> None:
        self._stream_owned = not hasattr(source, "read")
        self._stream: BinaryIO = Path(source).open("rb") if self._stream_owned else source  # type: ignore[assignment]
        self.manifest = self._read_manifest()
        self._records_offset = self._stream.tell()

    def _read_manifest(self) -> dict:
        prefix = self._stream.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise ValueError("truncated Strata container header")
        magic, version, manifest_size = _PREFIX.unpack(prefix)
        if magic != CONTAINER_MAGIC or version != STRATA_FORMAT_VERSION:
            raise ValueError("unsupported Strata container")
        raw = self._stream.read(manifest_size)
        if len(raw) != manifest_size:
            raise ValueError("truncated Strata manifest")
        manifest = json.loads(raw.decode("utf-8"))
        if manifest.get("format") != "strata-container" or manifest.get("version") != version:
            raise ValueError("invalid Strata manifest")
        return manifest

    def tensor_names(self) -> list[str]:
        return [item["name"] for item in self.manifest.get("tensors", [])]

    def read_tensors(self) -> Iterable[TensorRecord]:
        self._stream.seek(self._records_offset)
        for item in self.manifest.get("tensors", []):
            raw_header = self._stream.read(_RECORD.size)
            if len(raw_header) != _RECORD.size:
                raise ValueError("truncated Strata tensor record")
            payload_size, scales_size, expected = _RECORD.unpack(raw_header)
            payload = self._stream.read(payload_size)
            scales = self._stream.read(scales_size)
            if len(payload) != payload_size or len(scales) != scales_size:
                raise ValueError("truncated Strata tensor payload")
            if hashlib.sha256(payload + scales).digest() != expected:
                raise ValueError(f"checksum mismatch for tensor {item['name']}")
            if item["payload_bytes"] != payload_size or item["scales_bytes"] != scales_size:
                raise ValueError(f"manifest mismatch for tensor {item['name']}")
            yield TensorRecord(
                item["name"], item["rows"], item["cols"], item["group_size"],
                item["codec"], payload, scales,
            )

    def close(self) -> None:
        if self._stream_owned:
            self._stream.close()

    def __enter__(self) -> "StrataContainerReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
