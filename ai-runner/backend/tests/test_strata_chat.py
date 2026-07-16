import pytest

from backend.core.strata_ultra import StrataChatMessage, format_chat_prompt


def test_chat_prompt_is_deterministic_and_adds_assistant_turn():
    result = format_chat_prompt([
        StrataChatMessage("system", "Be concise."),
        StrataChatMessage("user", "Hello"),
    ])

    assert result == (
        "<|system|>\nBe concise.\n<|end|>\n"
        "<|user|>\nHello\n<|end|>\n<|assistant|>\n"
    )


def test_chat_prompt_rejects_empty_or_unknown_messages():
    with pytest.raises(ValueError, match="at least one"):
        format_chat_prompt([])
    with pytest.raises(ValueError, match="role"):
        StrataChatMessage("tool", "result")
    with pytest.raises(ValueError, match="content"):
        StrataChatMessage("user", "  ")
