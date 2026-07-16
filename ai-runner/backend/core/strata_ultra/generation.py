"""Experimental autoregressive generation loop for Strata transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from collections.abc import Iterator

from .executor import StrataRuntime
from .tokenizer import ByteTokenizer
from .transformer import LowBitTransformer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    eos_token: int | None = None
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    cancel_event: Event | None = None

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
        return self.generate_with_metadata(prompt, config)["text"]

    def generate_with_metadata(self, prompt: str, config: GenerationConfig | None = None) -> dict[str, object]:
        """Generate text and return stable completion metadata for API callers."""
        config = config or GenerationConfig()
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [0]
        generated: list[int] = []
        for token in tokens[:-1]:
            if config.cancel_event is not None and config.cancel_event.is_set():
                return {"text": prompt + self.tokenizer.decode(generated), "generated_tokens": 0, "finish_reason": "cancelled"}
            self.transformer.step(self._embedding_row(token))
        current = tokens[-1]
        finish_reason = "length"
        for _ in range(config.max_new_tokens):
            if config.cancel_event is not None and config.cancel_event.is_set():
                finish_reason = "cancelled"
                break
            hidden = self.transformer.step(self._embedding_row(current))
            logits = self.runtime.tensor_matvec(self.output, hidden)
            current = max(range(len(logits)), key=logits.__getitem__)
            generated.append(current)
            if current == config.eos_token or current in config.stop_token_ids:
                finish_reason = "stop"
                break
        return {
            "text": prompt + self.tokenizer.decode(generated),
            "generated_tokens": len(generated),
            "finish_reason": finish_reason,
        }

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[dict[str, object]]:
        """Yield generated token events without buffering the completion."""
        config = config or GenerationConfig()
        tokens = self.tokenizer.encode(prompt) or [0]
        for token in tokens[:-1]:
            if config.cancel_event is not None and config.cancel_event.is_set():
                yield {"finish_reason": "cancelled", "generated_tokens": 0}
                return
            self.transformer.step(self._embedding_row(token))
        current = tokens[-1]
        generated = 0
        previous_text = ""
        stream_tokens: list[int] = []
        for _ in range(config.max_new_tokens):
            if config.cancel_event is not None and config.cancel_event.is_set():
                yield {"finish_reason": "cancelled", "generated_tokens": generated}
                return
            hidden = self.transformer.step(self._embedding_row(current))
            logits = self.runtime.tensor_matvec(self.output, hidden)
            current = max(range(len(logits)), key=logits.__getitem__)
            generated += 1
            stopped = current == config.eos_token or current in config.stop_token_ids
            stream_tokens.append(current)
            decoded = self.tokenizer.decode(stream_tokens)
            delta = decoded[len(previous_text):] if decoded.startswith(previous_text) else decoded
            previous_text = decoded
            yield {
                "token_id": current,
                "text": delta,
                "generated_tokens": generated,
            }
            if stopped:
                yield {"finish_reason": "stop", "generated_tokens": generated}
                return
        yield {"finish_reason": "length", "generated_tokens": generated}
