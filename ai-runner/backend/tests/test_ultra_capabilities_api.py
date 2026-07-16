import importlib.util

import pytest

from backend.api import routes_ultra


@pytest.mark.asyncio
async def test_capabilities_report_active_tokenizer_backend():
    result = await routes_ultra.capabilities()
    expected = "gguf-bpe" if importlib.util.find_spec("tokenizers") else "byte-fallback"
    assert result["tokenizer_backend"] == expected
    assert set(result["execution_backends"]) == {"python", "numpy", "cuda"}
    assert result["execution_backends"]["python"]["available"] is True
    assert result["execution_backends"]["cuda"]["weight_codecs"] == ["ternary-q05"]
    assert isinstance(result["native_iq_decoder"], bool)
    assert result["readiness"]["container_io"] is True
    assert result["readiness"]["production_chat_runtime"] is False
    assert result["readiness"]["chat_completions_api"] is True
    assert result["readiness"]["sse_generation_api"] is True
