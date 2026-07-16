"""Integration tests for the FastAPI boundary and its security contract."""

from __future__ import annotations

import json
import base64
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from backend.core.extreme_model import estimated_specification
from backend.core.hardware_profile import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareProfile,
    RAMInfo,
    VirtualMemoryInfo,
)
from backend.core.inference_engine import GenerationResult, ModelInfo, engine
from backend.core.runtime_capabilities import RuntimeCapabilities
from backend.db import session_store
from backend.main import app
from backend.api import routes_models, routes_optimizer
from backend.api.auth import websocket_access_allowed
from backend.models.model_manager import DownloadProgress, ModelMetadata


@pytest_asyncio.fixture
async def api_client(tmp_path, monkeypatch):
    """Run each API test against an isolated SQLite database."""
    monkeypatch.delenv("AI_RUNNER_API_KEY", raising=False)
    monkeypatch.setattr(session_store, "DB_PATH", str(tmp_path / "api-test.db"))
    await session_store.init_db()
    await session_store.ensure_default_settings()

    original_state = (
        engine._model,
        engine._model_info,
        engine._config,
        engine._is_generating,
        engine._should_stop,
        engine._last_load_report,
    )
    engine._model = None
    engine._model_info = None
    engine._config = None
    engine._is_generating = False
    engine._should_stop = False
    engine._last_load_report = None
    engine._stop_event.clear()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client

    (
        engine._model,
        engine._model_info,
        engine._config,
        engine._is_generating,
        engine._should_stop,
        engine._last_load_report,
    ) = original_state
    engine._stop_event.clear()


@pytest.mark.asyncio
async def test_root_status_and_request_validation(api_client):
    root = await api_client.get("/")
    assert root.status_code == 200
    assert root.json()["status"] == "running"

    status = await api_client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["model_loaded"] is False

    invalid_chat = await api_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0},
    )
    assert invalid_chat.status_code == 422

    invalid_plan = await api_client.post(
        "/api/models/example/plan",
        json={"context_length": 1},
    )
    assert invalid_plan.status_code == 422


@pytest.mark.asyncio
async def test_session_crud_and_exports(api_client):
    created = await api_client.post(
        "/api/sessions",
        json={"title": "API session", "params": {"temperature": 0.2}},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]

    listed = await api_client.get("/api/sessions")
    assert [item["id"] for item in listed.json()["sessions"]] == [session_id]

    empty_update = await api_client.put(f"/api/sessions/{session_id}", json={})
    assert empty_update.status_code == 400

    updated = await api_client.put(
        f"/api/sessions/{session_id}",
        json={"title": "Renamed", "pinned": True},
    )
    assert updated.status_code == 200

    missing_update = await api_client.put(
        "/api/sessions/missing",
        json={"title": "Nope"},
    )
    assert missing_update.status_code == 404

    markdown = await api_client.get(f"/api/sessions/{session_id}/export/markdown")
    assert markdown.status_code == 200
    assert "Renamed" in markdown.text
    assert "attachment" in markdown.headers["content-disposition"]

    exported_json = await api_client.get(f"/api/sessions/{session_id}/export/json")
    assert exported_json.status_code == 200
    assert json.loads(exported_json.text)["id"] == session_id

    assert (await api_client.delete(f"/api/sessions/{session_id}")).status_code == 200
    assert (await api_client.get(f"/api/sessions/{session_id}")).status_code == 404
    assert (await api_client.delete(f"/api/sessions/{session_id}")).status_code == 404
    assert (await api_client.get("/api/sessions/missing/export/json")).status_code == 404


@pytest.mark.asyncio
async def test_api_key_and_origin_protection(api_client):
    configured = await api_client.put(
        "/api/settings",
        json={"settings": {"api_key": "  secret-value  "}},
    )
    assert configured.status_code == 200

    unauthorized = await api_client.get("/api/settings")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"

    bearer = {"Authorization": "Bearer secret-value"}
    authorized = await api_client.get("/api/settings", headers=bearer)
    assert authorized.status_code == 200
    assert authorized.json()["settings"]["api_key"] is None
    assert authorized.json()["api_key_configured"] is True

    exported = await api_client.get("/api/settings/export", headers=bearer)
    assert exported.status_code == 200
    assert "api_key" not in exported.json()

    x_api_key = await api_client.get(
        "/api/settings",
        headers={"X-API-Key": "secret-value"},
    )
    assert x_api_key.status_code == 200

    untrusted_origin = await api_client.get(
        "/api/settings",
        headers={**bearer, "Origin": "https://attacker.example"},
    )
    assert untrusted_origin.status_code == 403

    trusted_origin = await api_client.get(
        "/api/settings",
        headers={**bearer, "Origin": "http://localhost:1420"},
    )
    assert trusted_origin.status_code == 200


@pytest.mark.asyncio
async def test_environment_api_key_override(api_client, monkeypatch):
    monkeypatch.setenv("AI_RUNNER_API_KEY", "environment-secret")
    assert (await api_client.get("/api/settings")).status_code == 401

    response = await api_client.get(
        "/api/settings",
        headers={"Authorization": "Bearer environment-secret"},
    )
    assert response.status_code == 200
    assert response.json()["api_key_source"] == "environment"
    assert response.json()["settings"]["api_key"] is None


@pytest.mark.asyncio
async def test_websocket_key_uses_subprotocol_instead_of_url(api_client):
    await api_client.put(
        "/api/settings",
        json={"settings": {"api_key": "websocket-secret"}},
    )
    encoded = base64.urlsafe_b64encode(b"websocket-secret").decode().rstrip("=")

    websocket = SimpleNamespace(
        headers={
            "origin": "http://localhost:1420",
            "sec-websocket-protocol": f"ai-runner, ai-runner-key.{encoded}",
        }
    )
    assert await websocket_access_allowed(websocket) == (True, 1000)

    wrong_key = SimpleNamespace(
        headers={
            "origin": "http://localhost:1420",
            "sec-websocket-protocol": "ai-runner, ai-runner-key.invalid!",
        }
    )
    assert await websocket_access_allowed(wrong_key) == (False, 4401)

    untrusted = SimpleNamespace(
        headers={
            "origin": "https://attacker.example",
            "sec-websocket-protocol": f"ai-runner, ai-runner-key.{encoded}",
        }
    )
    assert await websocket_access_allowed(untrusted) == (False, 4403)


@pytest.mark.asyncio
async def test_settings_allowlist_network_consent_and_normalization(api_client):
    rejected_network = await api_client.put(
        "/api/settings",
        json={"settings": {"api_host": "0.0.0.0"}},
    )
    assert rejected_network.status_code == 422

    rejected_without_key = await api_client.put(
        "/api/settings",
        json={
            "settings": {
                "api_host": "0.0.0.0",
                "allow_network_access": True,
            }
        },
    )
    assert rejected_without_key.status_code == 422

    accepted = await api_client.put(
        "/api/settings",
        json={
            "settings": {
                "api_host": "0.0.0.0",
                "allow_network_access": True,
                "api_key": "network-secret",
                "tensor_split": [3, 1],
                "extreme_mode_enabled": True,
                "extreme_preset": "maximum_capacity",
                "adaptive_load": True,
                "adaptive_max_attempts": 7,
                "backend_preference": "auto",
                "context_compaction_mode": "extractive_summary",
            }
        },
    )
    assert accepted.status_code == 200

    auth = {"Authorization": "Bearer network-secret"}
    settings = (await api_client.get("/api/settings", headers=auth)).json()["settings"]
    assert settings["allow_network_access"] is True
    assert settings["tensor_split"] == [0.75, 0.25]
    assert settings["extreme_preset"] == "maximum_capacity"
    assert settings["adaptive_max_attempts"] == 7
    assert settings["context_compaction_mode"] == "extractive_summary"

    forbidden_key = await api_client.put(
        "/api/settings",
        headers=auth,
        json={"settings": {"unknown_setting": True}},
    )
    assert forbidden_key.status_code == 422

    malformed_host = await api_client.put(
        "/api/settings",
        headers=auth,
        json={"settings": {"api_host": "bad host!"}},
    )
    assert malformed_host.status_code == 422


@pytest.mark.asyncio
async def test_environment_key_allows_network_settings(api_client, monkeypatch):
    monkeypatch.setenv("AI_RUNNER_API_KEY", "environment-network-secret")
    response = await api_client.put(
        "/api/settings",
        headers={"Authorization": "Bearer environment-network-secret"},
        json={
            "settings": {
                "api_host": "0.0.0.0",
                "allow_network_access": True,
            }
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_openai_non_streaming_response_has_real_usage(api_client, monkeypatch):
    engine._model = object()
    engine._model_info = ModelInfo(
        model_id="local-model",
        model_path="model.gguf",
        n_gpu_layers=12,
        context_length=4096,
        total_layers=32,
        is_loaded=True,
    )
    monkeypatch.setattr(engine, "count_prompt_tokens", lambda messages: 7)
    monkeypatch.setattr(
        engine,
        "generate_sync",
        lambda messages, params: GenerationResult(
            content="Hello from local inference",
            tokens_generated=5,
            finish_reason="length",
        ),
    )

    response = await api_client.post(
        "/v1/chat/completions",
        json={
            "model": "ignored-alias",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "local-model"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
    }


@pytest.mark.asyncio
async def test_streaming_chat_persists_complete_assistant_message(api_client, monkeypatch):
    session = (await api_client.post("/api/sessions", json={"title": "Streaming"})).json()
    engine._model = object()

    async def fake_stream(messages, params):
        yield {"type": "token", "content": "Mer", "tokens_generated": 1}
        yield {"type": "token", "content": "haba", "tokens_generated": 2}
        yield {
            "type": "done",
            "result": {"tokens_generated": 2, "finish_reason": "stop"},
        }

    monkeypatch.setattr(engine, "generate_streaming", fake_stream)
    response = await api_client.post(
        "/api/chat",
        json={"session_id": session["id"], "content": "Selam"},
    )
    assert response.status_code == 200
    assert '"type": "done"' in response.text

    stored = (await api_client.get(f"/api/sessions/{session['id']}")).json()
    assert [(message["role"], message["content"]) for message in stored["messages"]] == [
        ("user", "Selam"),
        ("assistant", "Merhaba"),
    ]
    assert stored["messages"][1]["tokens_generated"] == 2


@pytest.mark.asyncio
async def test_chat_and_download_conflict_paths(api_client):
    no_model = await api_client.post(
        "/api/chat",
        json={"session_id": "missing", "content": "Hello"},
    )
    assert no_model.status_code == 503

    engine._model = object()
    missing_session = await api_client.post(
        "/api/chat",
        json={"session_id": "missing", "content": "Hello"},
    )
    assert missing_session.status_code == 404

    idle_stop = await api_client.post("/api/chat/stop")
    assert idle_stop.json() == {"status": "idle"}

    no_active_download = await api_client.post(
        "/api/models/org/model/download/cancel"
    )
    assert no_active_download.status_code == 409


@pytest.mark.asyncio
async def test_optimizer_api_contracts_without_system_mutation(api_client, monkeypatch):
    class Dumpable:
        def __init__(self, **payload):
            self.payload = payload

        def model_dump(self):
            return self.payload

    monkeypatch.setattr(
        routes_optimizer,
        "get_optimizer_status",
        lambda: Dumpable(optimization_score=88, recommendations=["ready"]),
    )
    monkeypatch.setattr(
        routes_optimizer,
        "analyze_pagefile",
        lambda model_size_mb: Dumpable(status="ok", model_size_mb=model_size_mb),
    )
    monkeypatch.setattr(
        routes_optimizer,
        "audit_services",
        lambda: [Dumpable(name="service-a", running=True)],
    )
    monkeypatch.setattr(
        routes_optimizer,
        "get_top_processes",
        lambda limit: [Dumpable(pid=123, name="worker")][:limit],
    )
    monkeypatch.setattr(
        routes_optimizer,
        "analyze_ramdisk",
        lambda model_size_mb: Dumpable(status="recommended", model_size_mb=model_size_mb),
    )
    monkeypatch.setattr(
        routes_optimizer,
        "calculate_prompt_budget",
        lambda **kwargs: {"remaining": 123, **kwargs},
    )
    monkeypatch.setattr(
        routes_optimizer,
        "get_gpu_profiles",
        lambda: Dumpable(gpus=[], tensor_split_recommended=[]),
    )

    action_names = [
        "lock_cpu_affinity_and_priority",
        "flush_vram_cache",
        "apply_windows_performance_mode",
        "create_zero_vram_launcher",
        "apply_nvidia_sysmem_fallback_tweak",
    ]
    for action_name in action_names:
        monkeypatch.setattr(
            routes_optimizer,
            action_name,
            lambda name=action_name: {"status": "mocked", "action": name},
        )

    assert (await api_client.get("/api/optimizer/status")).json()["optimization_score"] == 88
    assert (
        await api_client.get("/api/optimizer/pagefile", params={"model_size_mb": 4096})
    ).json()["model_size_mb"] == 4096
    assert len((await api_client.get("/api/optimizer/services")).json()["services"]) == 1
    assert (await api_client.get("/api/optimizer/processes", params={"limit": 5})).json()[
        "total_shown"
    ] == 1
    assert (await api_client.get("/api/optimizer/processes", params={"limit": 0})).status_code == 422
    assert (
        await api_client.get("/api/optimizer/ramdisk", params={"model_size_mb": 2048})
    ).json()["status"] == "recommended"

    budget = await api_client.post(
        "/api/optimizer/prompt-budget",
        params={"context_length": 4096, "system_prompt": "Be concise"},
        json=[{"role": "user", "content": "Hello"}],
    )
    assert budget.status_code == 200
    assert budget.json()["remaining"] == 123
    assert (await api_client.get("/api/optimizer/gpu-profile")).json()["gpus"] == []

    endpoints = [
        "/api/optimizer/affinity",
        "/api/optimizer/vram-flush",
        "/api/optimizer/apply-windows-performance",
        "/api/optimizer/create-launcher",
        "/api/optimizer/apply-nvidia-tweak",
    ]
    for endpoint in endpoints:
        response = await api_client.post(endpoint)
        assert response.status_code == 200
        assert response.json()["status"] == "mocked"

    monkeypatch.setattr(
        routes_optimizer,
        "get_optimizer_status",
        lambda: (_ for _ in ()).throw(RuntimeError("optimizer unavailable")),
    )
    fallback = await api_client.get("/api/optimizer/status")
    assert fallback.json()["optimization_score"] == 0


@pytest.mark.asyncio
async def test_model_library_search_plan_and_delete_contracts(api_client, monkeypatch):
    model = ModelMetadata(
        id="org/model-7B-GGUF",
        display_name="Model 7B",
        parameter_count=7_000_000_000,
        available_quants=["Q4_K_M"],
        downloaded_quant="Q4_K_M",
        file_size_bytes=4 * 1024**3,
        local_path="C:/models/model.gguf",
        author="org",
    )

    async def fake_search(query, limit):
        assert query == "model"
        assert limit == 20
        return [model]

    monkeypatch.setattr(routes_models.model_manager, "search_models", fake_search)
    monkeypatch.setattr(routes_models.model_manager, "get_local_models", lambda: [model])
    monkeypatch.setattr(
        routes_models.model_manager,
        "get_compatibility_badge",
        lambda *args, **kwargs: "compatible",
    )
    monkeypatch.setattr(
        routes_models,
        "get_hardware_profile",
        lambda **kwargs: SimpleNamespace(
            gpu=SimpleNamespace(vram_free_mb=12_000),
            gpus=[SimpleNamespace()],
            ram=SimpleNamespace(free_mb=32_000),
        ),
    )

    search = await api_client.get("/api/models/search", params={"q": "model"})
    assert search.status_code == 200
    assert search.json()["models"][0]["compatibility"] == "compatible"

    local = await api_client.get("/api/models/local")
    assert local.status_code == 200
    assert local.json()["models"][0]["id"] == model.id

    openai_models = await api_client.get("/v1/models")
    assert openai_models.json()["data"][0]["owned_by"] == "org"

    class DumpablePlan:
        def model_dump(self):
            return {"gpu_layers": 20, "ram_layers": 12, "disk_layers": 0}

    monkeypatch.setattr(routes_models, "calculate_offload_plan", lambda **kwargs: DumpablePlan())
    plan = await api_client.post(
        f"/api/models/{model.id}/plan",
        json={"quant": "Q4_K_M", "context_length": 4096},
    )
    assert plan.status_code == 200
    assert plan.json()["gpu_layers"] == 20

    monkeypatch.setattr(routes_models.model_manager, "delete_model", lambda model_id, quant=None: True)
    assert (await api_client.delete(f"/api/models/local/{model.id}")).status_code == 200
    monkeypatch.setattr(routes_models.model_manager, "delete_model", lambda model_id, quant=None: False)
    assert (await api_client.delete(f"/api/models/local/{model.id}")).status_code == 404


@pytest.mark.asyncio
async def test_model_load_unload_and_optimization_contract(api_client, monkeypatch):
    model = ModelMetadata(
        id="org/loadable-7B-GGUF",
        display_name="Loadable",
        parameter_count=7_000_000_000,
        downloaded_quant="Q4_K_M",
        file_size_bytes=4 * 1024**3,
        local_path="C:/models/loadable.gguf",
    )
    monkeypatch.setattr(routes_models.model_manager, "get_local_models", lambda: [model])
    monkeypatch.setattr(
        routes_models,
        "get_hardware_profile",
        lambda **kwargs: SimpleNamespace(gpus=[SimpleNamespace(), SimpleNamespace()]),
    )
    updated = []
    monkeypatch.setattr(routes_models.model_manager, "update_last_used", updated.append)

    def fake_load(model_id, model_path, config):
        assert model_id == model.id
        assert model_path == model.local_path
        assert config.n_gpu_layers == 12
        assert config.main_gpu == 1
        assert config.tensor_split == [0.75, 0.25]
        return ModelInfo(
            model_id=model_id,
            model_path=model_path,
            n_gpu_layers=config.n_gpu_layers,
            context_length=config.context_length,
            total_layers=32,
            is_loaded=True,
            main_gpu=config.main_gpu,
            tensor_split=config.tensor_split,
        )

    monkeypatch.setattr(engine, "load_model", fake_load)
    monkeypatch.setattr(
        engine,
        "get_optimization_summary",
        lambda: {"flash_attention": True, "main_gpu": 1},
    )

    loaded = await api_client.post(
        f"/api/models/{model.id}/load",
        json={
            "n_gpu_layers": 12,
            "context_length": 8192,
            "selected_gpu_index": 1,
            "tensor_split": [3, 1],
            "speculative_decoding": True,
            "draft_num_pred_tokens": 8,
        },
    )
    assert loaded.status_code == 200
    assert loaded.json()["model"]["main_gpu"] == 1
    assert updated == [model.id]

    engine._model = object()
    optimizations = await api_client.get("/api/models/optimizations")
    assert optimizations.json()["loaded"] is True
    assert optimizations.json()["optimizations"]["main_gpu"] == 1

    unloaded = []
    monkeypatch.setattr(engine, "unload_model", lambda: unloaded.append(True))
    assert (await api_client.post("/api/models/unload")).status_code == 200
    assert unloaded == [True]

    monkeypatch.setattr(routes_models.model_manager, "get_local_models", lambda: [])
    missing = await api_client.post(f"/api/models/{model.id}/load", json={})
    assert missing.status_code == 404

    invalid_split = await api_client.post(
        f"/api/models/{model.id}/load",
        json={"tensor_split": [1, 0]},
    )
    assert invalid_split.status_code == 422


@pytest.mark.asyncio
async def test_cpu_backend_uses_cpu_plan_and_model_volume(api_client, monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "cpu-model.gguf"
    model_path.write_bytes(b"GGUF" + b"x" * 32)
    model = ModelMetadata(
        id="org/cpu-7B-GGUF",
        display_name="CPU Model",
        parameter_count=7_000_000_000,
        downloaded_quant="Q4_K_M",
        file_size_bytes=4 * 1024**3,
        local_path=str(model_path),
    )
    gpu = GPUInfo(name="NVIDIA RTX", vram_total_mb=20_000, vram_free_mb=18_000)
    hardware = HardwareProfile(
        gpu=gpu,
        gpus=[gpu],
        ram=RAMInfo(total_mb=64_000, free_mb=56_000),
        virtual_memory=VirtualMemoryInfo(total_mb=32_000, free_mb=30_000),
        disk=DiskInfo(type="SSD", free_gb=500, path=str(model_dir)),
        cpu=CPUInfo(name="Test CPU", cores=8, threads=16),
    )
    runtime = RuntimeCapabilities(
        llama_cpp_installed=True,
        active_backend="cuda",
        gpu_offload_supported=True,
    )
    captured = {}

    def fake_hardware_profile(**kwargs):
        captured["model_dir"] = kwargs.get("model_dir")
        return hardware

    def fake_load(model_id, local_path, config):
        captured["config"] = config
        engine._config = config
        return ModelInfo(
            model_id=model_id,
            model_path=local_path,
            n_gpu_layers=config.n_gpu_layers,
            context_length=config.context_length,
            total_layers=32,
            is_loaded=True,
        )

    monkeypatch.setattr(routes_models.model_manager, "model_dir", str(model_dir))
    monkeypatch.setattr(routes_models.model_manager, "get_local_models", lambda: [model])
    monkeypatch.setattr(routes_models.model_manager, "update_last_used", lambda model_id: None)
    monkeypatch.setattr(routes_models, "get_hardware_profile", fake_hardware_profile)
    monkeypatch.setattr(routes_models, "detect_runtime_capabilities", lambda hardware: runtime)
    monkeypatch.setattr(
        routes_models,
        "specification_from_gguf",
        lambda *args, **kwargs: (
            estimated_specification(model.id, "Q4_K_M", model.parameter_count),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(engine, "load_model", fake_load)

    response = await api_client.post(
        f"/api/models/{model.id}/load",
        json={
            "backend_preference": "cpu",
            "adaptive_load": False,
            "context_length": 32768,
            "n_batch": 4096,
            "kv_cache_type": "f16",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["model_dir"] == str(model_dir)
    assert captured["config"].n_gpu_layers == 0
    assert captured["config"].flash_attn is False
    assert captured["config"].context_length == 2048
    assert captured["config"].n_batch == 64
    assert captured["config"].kv_cache_type == "q4_0"
    assert payload["feasibility"]["runtime"]["backend"] == "cpu"
    assert payload["runtime_profile"]["backend"] == "cpu"


@pytest.mark.asyncio
async def test_model_download_stream_and_cancel_contract(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes_models.model_manager, "model_dir", str(tmp_path))

    async def fake_download(model_id, quant, progress_callback, max_file_bytes):
        assert model_id == "org/model"
        assert quant == "Q4_K_M"
        assert max_file_bytes > 0
        progress_callback(
            DownloadProgress(
                model_id=model_id,
                quant=quant,
                progress=0.5,
                downloaded_bytes=50,
                total_bytes=100,
            )
        )
        progress_callback(
            DownloadProgress(
                model_id=model_id,
                quant=quant,
                progress=1.0,
                downloaded_bytes=100,
                total_bytes=100,
                status="completed",
            )
        )
        return str(tmp_path / "model.gguf")

    monkeypatch.setattr(routes_models.model_manager, "download_model", fake_download)
    response = await api_client.post(
        "/api/models/org/model/download",
        json={"quant": "Q4_K_M"},
    )
    assert response.status_code == 200
    assert '"status":"completed"' in response.text

    monkeypatch.setattr(routes_models.model_manager, "cancel_download", lambda model_id: True)
    cancelled = await api_client.post("/api/models/org/model/download/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["model_id"] == "org/model"
