"""Deterministic chat-message adaptation for the Strata runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StrataChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant")
        if not self.content.strip():
            raise ValueError("message content must not be empty")


def format_chat_prompt(messages: Iterable[StrataChatMessage], *, add_generation_prompt: bool = True) -> str:
    """Format validated messages using Strata's explicit, tokenizer-neutral template.

    The delimiters are intentionally plain text so the adapter works with the
    byte tokenizer and remains inspectable when a model-specific tokenizer is
    unavailable.  A future model template can replace this function without
    changing the generation API.
    """
    normalized = list(messages)
    if not normalized:
        raise ValueError("at least one chat message is required")
    parts: list[str] = []
    for message in normalized:
        parts.append(f"<|{message.role}|>\n{message.content.strip()}\n<|end|>")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)
