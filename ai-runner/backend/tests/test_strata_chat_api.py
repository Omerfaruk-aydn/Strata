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
async def test_strata_chat_stream_maps_token_events_to_openai_deltas(monkeypatch):
    class FakeResponse:
        @property
        async def body_iterator(self):
            yield 'data: {"text":"hello"}\n\n'
            yield 'data: {"finish_reason":"length"}\n\n'

    async def fake_stream(request):
        return FakeResponse()

    monkeypatch.setattr(routes_ultra, "strata_generate_stream", fake_stream)
    response = await routes_ultra.strata_chat_completions(_request(stream=True))
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert '"role": "assistant"' in chunks[0]
    assert '"content": "hello"' in chunks[1]
    assert '"finish_reason": "length"' in chunks[2]
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_strata_chat_stream_preserves_worker_error_details(monkeypatch):
    class FakeResponse:
        @property
        async def body_iterator(self):
            yield 'data: {"error":"worker failed","finish_reason":"error","generated_tokens":0}\n\n'

    async def fake_stream(request):
        return FakeResponse()

    monkeypatch.setattr(routes_ultra, "strata_generate_stream", fake_stream)
    response = await routes_ultra.strata_chat_completions(_request(stream=True))
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert '"error": "worker failed"' in chunks[1]
    assert '"finish_reason": "error"' in chunks[1]


@pytest.mark.asyncio
async def test_strata_chat_stream_maps_invalid_messages_to_422():
    request = _request(messages=[{"role": "user", "content": "  "}], stream=True)
    with pytest.raises(routes_ultra.HTTPException) as failure:
        await routes_ultra.strata_chat_completions(request)
    assert failure.value.status_code == 422


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


@pytest.mark.asyncio
async def test_strata_generate_stream_times_out_while_worker_is_silent(monkeypatch):
    class SilentGenerator:
        def generate_stream(self, prompt, config):
            config.cancel_event.wait(2)
            yield {"finish_reason": "cancelled", "generated_tokens": 0}

    @contextmanager
    def fake_context(request, cancel_event):
        yield SilentGenerator(), "byte-fallback", 1, "python"

    monkeypatch.setattr(routes_ultra, "_strata_generator_context", fake_context)
    response = await routes_ultra.strata_generate_stream(_request(timeout_s=0.01))
    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)

    assert '"finish_reason": "timeout"' in events[0]
    assert routes_ultra._strata_generation_cancel is None


@pytest.mark.asyncio
async def test_strata_generate_stream_reports_worker_errors_as_terminal_events(monkeypatch):
    class FailingGenerator:
        def generate_stream(self, prompt, config):
            raise RuntimeError("synthetic worker failure")
            yield  # pragma: no cover

    @contextmanager
    def fake_context(request, cancel_event):
        yield FailingGenerator(), "byte-fallback", 1, "python"

    monkeypatch.setattr(routes_ultra, "_strata_generator_context", fake_context)
    response = await routes_ultra.strata_generate_stream(_request())
    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)

    assert '"finish_reason": "error"' in events[0]
    assert "synthetic worker failure" in events[0]
    assert routes_ultra._strata_generation_cancel is None
