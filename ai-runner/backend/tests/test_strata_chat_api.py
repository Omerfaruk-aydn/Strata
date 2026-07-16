import pytest

from backend.api import routes_ultra


def _request(**overrides):
    values = {
        "model_file": "model.strata",
        "embedding_tensor": "embedding",
        "output_tensor": "output",
        "width": 4,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    values.update(overrides)
    return routes_ultra.StrataChatRequest(**values)


@pytest.mark.asyncio
async def test_strata_chat_rejects_streaming_until_native_stream_contract_exists():
    with pytest.raises(routes_ultra.HTTPException) as failure:
        await routes_ultra.strata_chat_completions(_request(stream=True))
    assert failure.value.status_code == 501


@pytest.mark.asyncio
async def test_strata_chat_maps_generation_to_openai_shape(monkeypatch):
    async def fake_generate(request):
        assert "<|user|>" in request.prompt
        return {
            "text": request.prompt + "answer",
            "generated_tokens": 2,
            "finish_reason": "length",
            "tokenizer": "byte-fallback",
            "blocks": 3,
            "backend": "numpy",
        }

    monkeypatch.setattr(routes_ultra, "generate_text", fake_generate)
    response = await routes_ultra.strata_chat_completions(_request())

    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "answer"
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 2
