"""Experimental autoregressive generation loop for Strata transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .executor import StrataRuntime
from .tokenizer import ByteTokenizer
from .transformer import LowBitTransformer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    eos_token: int | None = None
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1 or self.max_new_tokens > 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        if self.eos_token is not None and self.eos_token < 0:
            raise ValueError("eos_token must be non-negative")
        if any(token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must contain non-negative token IDs")


class StrataGenerator:
    """Greedy generator using embedding and output tensors from a Strata model."""

    def __init__(self, runtime: StrataRuntime, transformer: LowBitTransformer, embedding: str, output: str, tokenizer=None):
        self.runtime = runtime
        self.transformer = transformer
        self.embedding = embedding
        self.output = output
        self.tokenizer = tokenizer or ByteTokenizer()

    def _embedding_row(self, token: int) -> list[float]:
        return self.runtime.tensor_row(self.embedding, token)

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [0]
        for token in tokens[:-1]:
            self.transformer.step(self._embedding_row(token))
        generated: list[int] = []
        current = tokens[-1]
        for _ in range(config.max_new_tokens):
            hidden = self.transformer.step(self._embedding_row(current))
            logits = self.runtime.tensor_matvec(self.output, hidden)
            current = max(range(len(logits)), key=logits.__getitem__)
            generated.append(current)
            if current == config.eos_token or current in config.stop_token_ids:
                break
        return prompt + self.tokenizer.decode(generated)
