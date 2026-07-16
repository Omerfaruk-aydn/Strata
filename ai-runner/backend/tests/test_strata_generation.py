import struct
from pathlib import Path

from backend.core.strata_ultra import (
    GenerationConfig,
    LowBitTransformer,
    LowBitTransformerBlock,
    StrataContainerWriter,
    StrataGenerator,
    StrataRuntime,
    TensorRecord,
)


def _identity(name: str, rows=2, cols=2):
    return TensorRecord(name, rows, cols, rows * cols, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0))


def test_tensor_row_reads_embedding_row(tmp_path: Path):
    target = tmp_path / "gen.strata"
    writer = StrataContainerWriter()
    writer.add_tensor(_identity("embedding"))
    writer.write(target)
    with StrataRuntime(target, 1024) as runtime:
        assert runtime.tensor_row("embedding", 1) == [0.0, 1.0]


def test_byte_tokenizer_is_reversible():
    from backend.core.strata_ultra import ByteTokenizer
    tokenizer = ByteTokenizer()
    text = "Strata"
    assert tokenizer.decode(tokenizer.encode(text)) == text
