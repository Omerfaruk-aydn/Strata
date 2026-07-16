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


class _GenerationRuntime:
    def tensor_row(self, name, token):
        return [float(token)]

    def tensor_matvec(self, name, hidden):
        return [0.0, 1.0]


class _GenerationTransformer:
    def step(self, hidden):
        return hidden


class _GenerationTokenizer:
    def encode(self, prompt):
        return [0]

    def decode(self, tokens):
        return "|".join(str(token) for token in tokens)


def test_generation_reports_length_finish_reason_and_token_count():
    generator = StrataGenerator(
        _GenerationRuntime(), _GenerationTransformer(), "embedding", "output", _GenerationTokenizer()
    )

    result = generator.generate_with_metadata("prompt", GenerationConfig(max_new_tokens=2))

    assert result == {"text": "prompt1|1", "generated_tokens": 2, "finish_reason": "length"}


def test_generation_stream_yields_token_events_and_terminal_reason():
    generator = StrataGenerator(
        _GenerationRuntime(), _GenerationTransformer(), "embedding", "output", _GenerationTokenizer()
    )

    events = list(generator.generate_stream("prompt", GenerationConfig(max_new_tokens=2)))

    assert events == [
        {"token_id": 1, "text": "1", "generated_tokens": 1},
        {"token_id": 1, "text": "1", "generated_tokens": 2},
        {"finish_reason": "length", "generated_tokens": 2},
    ]
