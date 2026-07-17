"""FastAPI contracts for Extreme Model Mode."""

from __future__ import annotations

import os
import struct

import httpx
import pytest
import pytest_asyncio

from backend.api import routes_extreme
from backend.core.hardware_profile import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareProfile,
    RAMInfo,
    VirtualMemoryInfo,
)
from backend.core.inference_engine import AdaptiveLoadReport, EngineConfig, ModelInfo, engine
from backend.core.runtime_capabilities import RuntimeCapabilities
from backend.db import session_store
from backend.main import app
from backend.models.model_manager import ModelMetadata


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _entry(key: str, value_type: int, value: bytes) -> bytes:
    return _gguf_string(key) + struct.pack("<I", value_type) + value


def write_model(path) -> None:
    entries = [
        _entry("general.architecture", 8, _gguf_string("qwen2")),
        _entry("general.parameter_count", 10, struct.pack("<Q", 100_000_000_000)),
        _entry("qwen2.context_length", 4, struct.pack("<I", 32768)),
        _entry("qwen2.embedding_length", 4, struct.pack("<I", 8192)),
        _entry("qwen2.block_count", 4, struct.pack("<I", 96)),
        _entry("qwen2.attention.head_count", 4, struct.pack("<I", 64)),
        _entry("qwen2.attention.head_count_kv", 4, struct.pack("<I", 8)),
    ]
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1000)
        + struct.pack("<Q", len(entries)) + b"".join(entries)
    )


def make_test_hardware(model_dir: str) -> HardwareProfile:
    gpu = GPUInfo(name="NVIDIA RTX Test", vram_total_mb=20 * 1024, vram_free_mb=19 * 1024)
    return HardwareProfile(
        gpu=gpu,
        gpus=[gpu],
        ram=RAMInfo(total_mb=64 * 1024, free_mb=58 * 1024),
        virtual_memory=VirtualMemoryInfo(total_mb=64 * 1024, free_mb=60 * 1024),
        disk=DiskInfo(type="SSD", free_gb=500, path=model_dir),
        cpu=CPUInfo(name="Test CPU", cores=16, threads=32),
        os_info="Windows Test",
        selected_gpu_index=0,
    )


@pytest_asyncio.fixture
async def extreme_client(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", str(tmp_path / "extreme-api.db"))
    await session_store.init_db()
    await session_store.ensure_default_settings()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    path = model_dir / "test-Q3_K_M.gguf"
    write_model(path)
    model = ModelMetadata(
        id="test/100B-GGUF",
        display_name="Test 100B",
        parameter_count=100_000_000_000,
        downloaded_quant="Q3_K_M",
        available_quants=["Q3_K_M"],
        file_size_bytes=path.stat().st_size,
        local_path=str(path),
        context_length=32768,
    )
    monkeypatch.setattr(routes_extreme.model_manager, "model_dir", str(model_dir))
    monkeypatch.setattr(routes_extreme.model_manager, "get_local_models", lambda: [model])
    hardware = make_test_hardware(str(model_dir))
    runtime = RuntimeCapabilities(
        llama_cpp_installed=True,
        active_backend="cuda",
        gpu_offload_supported=True,
    )
    monkeypatch.setattr(routes_extreme, "get_hardware_profile", lambda *args, **kwargs: hardware)
    monkeypatch.setattr(routes_extreme, "detect_runtime_capabilities", lambda *args, **kwargs: runtime)

    original = (
        engine._model,
        engine._model_info,
        engine._config,
        engine._is_generating,
        engine._last_load_report,
    )
    engine._model = None
    engine._model_info = None
    engine._config = None
    engine._is_generating = False
    engine._last_load_report = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, model, runtime
    (
        engine._model,
        engine._model_info,
        engine._config,
        engine._is_generating,
        engine._last_load_report,
    ) = original


@pytest.mark.asyncio
async def test_extreme_presets_simulation_and_local_analysis(extreme_client):
    client, model, _ = extreme_client
    presets = await client.get("/api/extreme/presets")
    assert presets.status_code == 200
    assert {item["name"] for item in presets.json()["presets"]} == {
        "safe", "balanced", "performance", "maximum_capacity",
    }

    simulation = await client.post(
        "/api/extreme/simulate",
        json={
            "model_id": "simulation/100B",
            "parameter_count": 100_000_000_000,
            "quant": "Q3_K_M",
            "preset": "maximum_capacity",
            "context_length": 2048,
        },
    )
    assert simulation.status_code == 200
    simulation_payload = simulation.json()
    simulated_report = simulation_payload["report"]
    assert simulated_report["model"]["parameter_count"] == 100_000_000_000
    assert simulated_report["runtime"]["n_gpu_layers"] > 0
    assert simulation_payload["quant_recommendation"]["recommended"] == "Q3_K_M"

    local = await client.post(
        f"/api/extreme/analyze/{model.id}",
        json={"quant": "Q3_K_M", "preset": "maximum_capacity", "context_length": 2048},
    )
    assert local.status_code == 200
    payload = local.json()
    assert payload["report"]["model"]["metadata_source"] == "gguf"
    assert payload["metadata"]["architecture"] == "qwen2"
    assert payload["metadata"]["block_count"] == 96


@pytest.mark.asyncio
async def test_extreme_capabilities_and_profile_contract(extreme_client):
    client, _, _ = extreme_client
    capabilities = await client.get("/api/extreme/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["active_backend"] == "cuda"

    profiles = await client.get("/api/extreme/profiles")
    assert profiles.status_code == 200
    assert profiles.json()["profiles"] == []


@pytest.mark.asyncio
async def test_extreme_benchmark_contract(extreme_client, monkeypatch):
    client, model, _ = extreme_client
    engine._model = object()
    engine._model_info = ModelInfo(
        model_id=model.id,
        model_path=model.local_path,
        n_gpu_layers=30,
        context_length=2048,
        total_layers=96,
        is_loaded=True,
        main_gpu=0,
    )
    engine._config = EngineConfig(n_gpu_layers=30, context_length=2048)

    async def fake_stream(messages, params):
        yield {"type": "token", "content": "ok"}
        yield {
            "type": "done",
            "result": {
                "content": "ok",
                "tokens_generated": 16,
                "tokens_per_sec": 1.5,
                "ttft_ms": 600.0,
                "total_time_ms": 10666.0,
                "stopped_by_user": False,
                "finish_reason": "length",
            },
        }

    monkeypatch.setattr(engine, "generate_streaming", fake_stream)
    monkeypatch.setattr(routes_extreme, "detect_gpus", lambda: [])
    response = await client.post("/api/extreme/benchmark", json={"max_tokens": 16})
    assert response.status_code == 200
    result = response.json()["benchmark"]
    assert result["tokens_per_second"] == 1.5
    assert result["n_gpu_layers"] == 30


@pytest.mark.asyncio
async def test_extreme_quantization_reports_missing_tool(extreme_client):
    client, _, runtime = extreme_client
    runtime.llama_quantize_path = None
    status = await client.get("/api/extreme/quantization")
    assert status.status_code == 200
    assert status.json()["available"] is False


@pytest.mark.asyncio
async def test_rebalance_restores_previous_configuration_on_failure(extreme_client, monkeypatch):
    client, model, _ = extreme_client
    previous = EngineConfig(n_gpu_layers=30, context_length=2048)
    engine._model = object()
    engine._model_info = ModelInfo(
        model_id=model.id,
        model_path=model.local_path,
        n_gpu_layers=30,
        context_length=2048,
        total_layers=96,
        is_loaded=True,
        main_gpu=0,
    )
    engine._config = previous
    calls = []

    def fake_adaptive_load(model_id, model_path, config, **kwargs):
        calls.append(config.model_copy(deep=True))
        if len(calls) == 1:
            engine._last_load_report = AdaptiveLoadReport(succeeded=False)
            raise RuntimeError("simulated rebalance OOM")
        engine._config = config
        restored = AdaptiveLoadReport(
            succeeded=True,
            final_config=config,
            recovered_from_oom=False,
        )
        engine._last_load_report = restored
        return engine._model_info, restored

    monkeypatch.setattr(engine, "load_model_adaptive", fake_adaptive_load)
    response = await client.post(
        "/api/extreme/rebalance",
        json={"preset": "maximum_capacity", "force": True},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "previous configuration was restored" in detail["message"]
    assert detail["rollback_report"]["succeeded"] is True
    assert len(calls) == 2
    assert calls[1].n_gpu_layers == previous.n_gpu_layers
