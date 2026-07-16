import pytest
from contextlib import contextmanager

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


@pytest.mark.asyncio
async def test_strata_generate_stream_emits_events_and_cleans_state(monkeypatch):
    class FakeGenerator:
        def generate_stream(self, prompt, config):
            yield {"token_id": 7, "text": "ok", "generated_tokens": 1}
            yield {"finish_reason": "length", "generated_tokens": 1}

    @contextmanager
    def fake_context(request, cancel_event):
        yield FakeGenerator(), "byte-fallback", 1, "python"

    monkeypatch.setattr(routes_ultra, "_strata_generator_context", fake_context)
    response = await routes_ultra.strata_generate_stream(_request())
    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)

    assert '"token_id": 7' in events[0]
    assert '"finish_reason": "length"' in events[1]
    assert routes_ultra._strata_generation_cancel is None
