import asyncio

import pytest
from fastapi import HTTPException

from backend.api import routes_chat
from backend.core.inference_engine import EngineConfig, InferenceParams


@pytest.mark.asyncio
async def test_non_streaming_generation_timeout_returns_gateway_timeout(monkeypatch):
    class FakeEngine:
        _config = EngineConfig(generation_timeout_s=0.001)
        model_info = None
        stopped = False

        def count_prompt_tokens(self, _messages):
            return 1

        def generate_sync(self, *_args, **_kwargs):
            import time
            time.sleep(0.05)

        def stop_generation(self):
            self.stopped = True

    fake = FakeEngine()
    monkeypatch.setattr(routes_chat, "engine", fake)
    with pytest.raises(HTTPException) as error:
        await routes_chat._non_streaming_completion([], InferenceParams(max_tokens=1), "test")
    assert error.value.status_code == 504
    assert fake.stopped is True
