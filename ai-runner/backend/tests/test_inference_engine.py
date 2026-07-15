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


class TestEngineConfig:
    """Tests for the EngineConfig optimization configuration model."""

    from backend.core.inference_engine import EngineConfig, KV_CACHE_TYPE_MAP, _get_physical_cores

    def test_defaults_are_optimized(self):
        """Default config should have optimizations enabled."""
        from backend.core.inference_engine import EngineConfig
        cfg = EngineConfig()
        assert cfg.kv_cache_type == "q4_0"
        assert cfg.flash_attn is True
        assert cfg.use_mlock is True
        assert cfg.cache_context_shift is True

    def test_kv_cache_type_valid_values(self):
        """All documented KV cache types should be accepted."""
        from backend.core.inference_engine import EngineConfig
        for t in ("f16", "q8_0", "q5_1", "q5_0", "q4_0"):
            cfg = EngineConfig(kv_cache_type=t)
            assert cfg.kv_cache_type == t

    def test_kv_cache_type_invalid_raises(self):
        """Unknown KV cache types should raise ValidationError."""
        from backend.core.inference_engine import EngineConfig
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            EngineConfig(kv_cache_type="q2_0")

    def test_draft_model_defaults(self):
        """Draft model should be None by default (speculative decoding off)."""
        from backend.core.inference_engine import EngineConfig
        cfg = EngineConfig()
        assert cfg.draft_model_path is None
        assert cfg.draft_n_gpu_layers == -1

    def test_kv_type_map_contains_q4(self):
        """q4_0 must map to GGML_TYPE_Q4_0 = 2."""
        from backend.core.inference_engine import KV_CACHE_TYPE_MAP, GGML_TYPE_Q4_0
        assert KV_CACHE_TYPE_MAP["q4_0"] == GGML_TYPE_Q4_0
        assert GGML_TYPE_Q4_0 == 2

    def test_kv_type_map_f16_highest(self):
        """f16 should map to GGML_TYPE_F16 = 1."""
        from backend.core.inference_engine import KV_CACHE_TYPE_MAP, GGML_TYPE_F16
        assert KV_CACHE_TYPE_MAP["f16"] == GGML_TYPE_F16


class TestPhysicalCores:
    """Tests for the physical core auto-detection optimization."""

    def test_returns_positive_integer(self):
        from backend.core.inference_engine import _get_physical_cores
        cores = _get_physical_cores()
        assert isinstance(cores, int)
        assert cores >= 1

    def test_physical_lte_logical(self):
        """Physical cores should never exceed logical (HT) count."""
        import psutil
        from backend.core.inference_engine import _get_physical_cores
        logical = psutil.cpu_count(logical=True) or 1
        physical = _get_physical_cores()
        assert physical <= logical


class TestContextShift:
    """Tests for the Smart Context Shifting algorithm."""

    def test_no_shift_within_budget(self):
        """Short conversations should not be trimmed."""
        from backend.core.inference_engine import InferenceEngine, EngineConfig
        engine = InferenceEngine()
        engine._config = EngineConfig(cache_context_shift=True)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = engine._apply_context_shift(messages, max_tokens_estimate=3800)
        assert len(result) == len(messages)

    def test_shift_disabled_returns_all(self):
        """When context shift is disabled, no trimming should occur."""
        from backend.core.inference_engine import InferenceEngine, EngineConfig
        engine = InferenceEngine()
        engine._config = EngineConfig(auto_context_prune=False)
        # Simulate a very long conversation (>3800 tokens estimated)
        long_content = "A" * 20000  # ~5000 tokens
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": long_content},
        ]
        result = engine._apply_context_shift(messages, max_tokens_estimate=100)
        assert len(result) == len(messages)

    def test_shift_trims_old_messages(self):
        """When budget exceeded, old messages should be dropped."""
        from backend.core.inference_engine import InferenceEngine, EngineConfig
        engine = InferenceEngine()
        engine._config = EngineConfig(cache_context_shift=True)
        # Build a conversation that is very large
        many_messages = [{"role": "system", "content": "S"}]
        for i in range(50):
            many_messages.append({"role": "user",      "content": "Q" * 100})
            many_messages.append({"role": "assistant", "content": "A" * 100})

        result = engine._apply_context_shift(many_messages, max_tokens_estimate=500)
        # Should be shorter than original
        assert len(result) < len(many_messages)
        # System message must always be preserved
        assert result[0]["role"] == "system"

    def test_last_message_always_kept(self):
        """The last user message must survive context shift."""
        from backend.core.inference_engine import InferenceEngine, EngineConfig
        engine = InferenceEngine()
        engine._config = EngineConfig(cache_context_shift=True)
        messages = [{"role": "system", "content": "S"}]
        for i in range(30):
            messages.append({"role": "user",      "content": "Q" * 200})
            messages.append({"role": "assistant", "content": "A" * 200})
        messages.append({"role": "user", "content": "FINAL_QUESTION"})

        result = engine._apply_context_shift(messages, max_tokens_estimate=300)
        contents = [m["content"] for m in result]
        assert "FINAL_QUESTION" in contents


class TestOptimizationSummary:
    """Tests for the optimization summary telemetry endpoint."""

    def test_summary_empty_when_not_loaded(self):
        from backend.core.inference_engine import InferenceEngine
        engine = InferenceEngine()
        assert engine.get_optimization_summary() == {}

    def test_summary_keys_when_configured(self):
        from backend.core.inference_engine import InferenceEngine, EngineConfig
        engine = InferenceEngine()
        engine._config = EngineConfig()
        engine._draft_model = None
        summary = engine.get_optimization_summary()
        required_keys = {
            "kv_cache_type", "flash_attn", "use_mlock",
            "use_mmap", "n_threads", "speculative_decoding", "context_shift",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_summary_kv_type_default(self):
        from backend.core.inference_engine import InferenceEngine, EngineConfig
        engine = InferenceEngine()
        engine._config = EngineConfig()
        engine._draft_model = None
        assert engine.get_optimization_summary()["kv_cache_type"] == "q4_0"
