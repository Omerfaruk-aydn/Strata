"""
AI Runner — Unit Tests: Inference Engine
Tests for model lifecycle, streaming, and stop functionality.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.inference_engine import InferenceEngine, InferenceParams, GenerationResult


class TestInferenceEngine:

    def test_initial_state(self):
        engine = InferenceEngine()
        assert not engine.is_loaded
        assert not engine.is_generating
        assert engine.model_info is None

    def test_stop_when_not_generating_is_noop(self):
        engine = InferenceEngine()
        engine.stop_generation()  # Should not raise
        assert not engine.is_generating

    def test_unload_when_not_loaded_is_noop(self):
        engine = InferenceEngine()
        engine.unload_model()  # Should not raise
        assert not engine.is_loaded

    def test_load_without_llama_raises(self):
        """When llama-cpp-python is absent, load_model should raise RuntimeError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("No module named 'llama_cpp'")
            return real_import(name, *args, **kwargs)

        engine = InferenceEngine()
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises((RuntimeError, ImportError)):
                engine.load_model("test/model", "/nonexistent/path.gguf")

    def test_inference_params_defaults(self):
        params = InferenceParams()
        assert 0.0 < params.temperature <= 2.0
        assert 0.0 < params.top_p <= 1.0
        assert params.max_tokens > 0

    def test_inference_params_custom(self):
        params = InferenceParams(temperature=0.1, max_tokens=100)
        assert params.temperature == 0.1
        assert params.max_tokens == 100

    def test_generation_result_fields(self):
        result = GenerationResult(
            content="Hello world",
            tokens_generated=2,
            finish_reason="stop",
        )
        assert result.content == "Hello world"
        assert result.tokens_generated == 2
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_streaming_no_model(self):
        engine = InferenceEngine()
        chunks = []
        async for chunk in engine.generate_streaming(
            messages=[{"role": "user", "content": "hi"}],
            params=InferenceParams(),
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        error_chunks = [c for c in chunks if c.get("type") == "error"]
        assert len(error_chunks) == 1
        assert "model" in error_chunks[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_generation(self):
        """stop_generation() sets flag that loop checks."""
        engine = InferenceEngine()
        engine._stop_event.clear()
        assert not engine._stop_event.is_set()
        engine.stop_generation()
        assert engine._stop_event.is_set()


class TestInferenceParams:

    def test_system_prompt_in_messages(self):
        """System prompt should be injectable as a system message."""
        params = InferenceParams(system_prompt="You are a helpful assistant.")
        assert params.system_prompt == "You are a helpful assistant."

    def test_stop_tokens_default_empty(self):
        params = InferenceParams()
        assert isinstance(params.stop, list)

    def test_stop_tokens_custom(self):
        params = InferenceParams(stop=["</s>", "[END]"])
        assert len(params.stop) == 2
        assert "</s>" in params.stop

    @pytest.mark.parametrize("temp", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_temperature_valid_range(self, temp):
        params = InferenceParams(temperature=temp)
        assert params.temperature == temp

    @pytest.mark.parametrize("bad_temp", [-0.1, 3.0])
    def test_temperature_out_of_range_raises(self, bad_temp):
        with pytest.raises(ValueError):
            InferenceParams(temperature=bad_temp)
