"""Versioned primitives for the Strata model container."""

from dataclasses import dataclass
import struct

STRATA_MAGIC = b"STRATA\x00\x01"
STRATA_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TensorHeader:
    """Self-describing metadata stored before a packed tensor payload."""

    name: str
    rows: int
    cols: int
    group_size: int
    codec: str = "ternary-q05"

    def validate(self) -> None:
        if not self.name or len(self.name.encode("utf-8")) > 255:
            raise ValueError("tensor name must be 1-255 UTF-8 bytes")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("tensor dimensions must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.codec != "ternary-q05":
            raise ValueError(f"unsupported codec: {self.codec}")

    def to_bytes(self) -> bytes:
        self.validate()
        name = self.name.encode("utf-8")
        codec = self.codec.encode("ascii")
        return (
            STRATA_MAGIC
            + struct.pack("<BHHII", STRATA_FORMAT_VERSION, len(name), len(codec), self.rows, self.cols)
            + struct.pack("<I", self.group_size)
            + name
            + codec
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["TensorHeader", int]:
        fixed = len(STRATA_MAGIC) + struct.calcsize("<BHHII") + 4
        if len(data) < fixed or data[:len(STRATA_MAGIC)] != STRATA_MAGIC:
            raise ValueError("invalid Strata tensor magic")
        offset = len(STRATA_MAGIC)
        version, name_len, codec_len, rows, cols = struct.unpack_from("<BHHII", data, offset)
        offset += struct.calcsize("<BHHII")
        (group_size,) = struct.unpack_from("<I", data, offset)
        offset += 4
        end = offset + name_len + codec_len
        if end > len(data) or version != STRATA_FORMAT_VERSION:
            raise ValueError("unsupported or truncated Strata tensor header")
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        codec = data[offset:end].decode("ascii")
        header = cls(name, rows, cols, group_size, codec)
        header.validate()
        return header, end
