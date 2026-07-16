import importlib.util

import pytest

from backend.api import routes_ultra


@pytest.mark.asyncio
async def test_capabilities_report_active_tokenizer_backend():
    result = await routes_ultra.capabilities()
    expected = "gguf-bpe" if importlib.util.find_spec("tokenizers") else "byte-fallback"
    assert result["tokenizer_backend"] == expected
