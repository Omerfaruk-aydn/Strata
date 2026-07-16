"""Deterministic fallback tokenizer for Strata runtime smoke tests."""


class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer; model-specific tokenizers can replace it."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, int(token))) for token in tokens).decode("utf-8", errors="replace")
