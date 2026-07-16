"""Tokenizer adapters for Strata generation."""


class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer; model-specific tokenizers can replace it."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, int(token))) for token in tokens).decode("utf-8", errors="replace")


class GGUFTokenizer:
    """Optional BPE adapter built from GGUF tokenizer token/merge metadata."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def from_metadata(cls, metadata: dict):
        tokens = metadata.get("tokenizer.ggml.tokens")
        merges = metadata.get("tokenizer.ggml.merges", [])
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("GGUF tokenizer token metadata is missing")
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            vocab = {str(token): index for index, token in enumerate(tokens)}
            normalized_merges = []
            for merge in merges:
                parts = str(merge).split(" ", 1)
                if len(parts) == 2:
                    normalized_merges.append((parts[0], parts[1]))
            model = BPE(vocab=vocab, merges=normalized_merges, unk_token=str(tokens[0]))
            tokenizer = Tokenizer(model)
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tokenizer.decoder = ByteLevelDecoder()
            return cls(tokenizer)
        except ImportError as exc:
            raise RuntimeError("tokenizers package is required for GGUF BPE decoding") from exc

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode([int(token) for token in tokens])
