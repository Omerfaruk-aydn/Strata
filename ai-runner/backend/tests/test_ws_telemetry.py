"""WebSocket authorization, commands, and telemetry-manager tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect

from backend.api import ws_telemetry
from backend.api.ws_telemetry import TelemetryManager
from backend.core import hardware_profile
from backend.core.inference_engine import ModelInfo, engine


class FakeWebSocket:
    def __init__(self, incoming=None, fail_send=False, headers=None):
        self.incoming = list(incoming or [])
        self.fail_send = fail_send
        self.headers = headers or {}
        self.accepted = False
        self.accepted_subprotocol = None
        self.closed_with = None
        self.sent = []

    async def accept(self, subprotocol=None):
        self.accepted = True
        self.accepted_subprotocol = subprotocol

    async def close(self, code):
        self.closed_with = code

    async def send_text(self, message):
        if self.fail_send:
            raise RuntimeError("socket closed")
        self.sent.append(json.loads(message))

    async def receive_text(self):
        if not self.incoming:
            raise WebSocketDisconnect()
        item = self.incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.asyncio
async def test_telemetry_manager_connect_broadcast_send_and_disconnect(monkeypatch):
    manager = TelemetryManager()

    async def idle_loop():
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_broadcast_loop", idle_loop)
    good = FakeWebSocket(headers={"sec-websocket-protocol": "ai-runner, ai-runner-key.test"})
    bad = FakeWebSocket(fail_send=True)

    await manager.connect(good)
    assert good.accepted is True
    assert good.accepted_subprotocol == "ai-runner"
    assert manager._running is True
    assert manager._task is not None

    manager._connections.add(bad)
    await manager.broadcast({"type": "telemetry", "data": {"ready": True}})
    assert good.sent[-1]["data"]["ready"] is True
    assert bad not in manager._connections

    await manager.send_to(good, {"type": "pong"})
    assert good.sent[-1]["type"] == "pong"
    await manager.send_to(bad, {"type": "ignored"})

    manager.disconnect(good)
    assert manager._running is False
    assert manager._task.cancelled() or manager._task.cancelling()


def test_collect_telemetry_uses_selected_gpu(monkeypatch):
    manager = TelemetryManager()
    monkeypatch.setattr(
        hardware_profile,
        "detect_gpus",
        lambda: [
            SimpleNamespace(
                vram_used_mb=100,
                vram_total_mb=1000,
                vram_free_mb=900,
                temperature=45,
            ),
            SimpleNamespace(
                vram_used_mb=200,
                vram_total_mb=2000,
                vram_free_mb=1800,
                temperature=55,
            ),
        ],
    )
    monkeypatch.setattr(
        hardware_profile,
        "detect_ram",
        lambda: SimpleNamespace(
            used_mb=4000,
            total_mb=16000,
            free_mb=12000,
            percent_used=25.0,
        ),
    )
    monkeypatch.setattr(engine, "_model", object())
    monkeypatch.setattr(engine, "_model_info", ModelInfo(
        model_id="telemetry-model",
        model_path="model.gguf",
        n_gpu_layers=24,
        context_length=8192,
        total_layers=32,
        is_loaded=True,
        main_gpu=1,
    ))

    snapshot = manager._collect_telemetry()
    assert snapshot["gpu"]["vram_total_mb"] == 2000
    assert snapshot["ram"]["percent"] == 25.0
    assert snapshot["engine"]["model_id"] == "telemetry-model"
    assert snapshot["engine"]["n_gpu_layers"] == 24
    assert snapshot["timestamp"].endswith("Z")


@pytest.mark.asyncio
async def test_websocket_endpoint_rejects_unauthorized_connection(monkeypatch):
    async def denied(websocket):
        return False, 4401

    monkeypatch.setattr(ws_telemetry, "websocket_access_allowed", denied)
    websocket = FakeWebSocket()
    await ws_telemetry.websocket_endpoint(websocket)
    assert websocket.closed_with == 4401
    assert websocket.accepted is False


@pytest.mark.asyncio
async def test_websocket_endpoint_handles_ping_stop_and_invalid_json(monkeypatch):
    async def allowed(websocket):
        return True, 1000

    class FakeManager:
        def __init__(self):
            self.connected = []
            self.disconnected = []
            self.sent = []

        async def connect(self, websocket):
            self.connected.append(websocket)
            await websocket.accept()

        def disconnect(self, websocket):
            self.disconnected.append(websocket)

        async def send_to(self, websocket, data):
            self.sent.append(data)

    fake_manager = FakeManager()
    stopped = []
    monkeypatch.setattr(ws_telemetry, "websocket_access_allowed", allowed)
    monkeypatch.setattr(ws_telemetry, "telemetry_manager", fake_manager)
    monkeypatch.setattr(engine, "stop_generation", lambda: stopped.append(True))
    websocket = FakeWebSocket(
        incoming=[
            "not-json",
            json.dumps({"type": "ping"}),
            json.dumps({"type": "stop"}),
        ]
    )

    await ws_telemetry.websocket_endpoint(websocket)
    assert websocket.accepted is True
    assert [message["type"] for message in fake_manager.sent] == [
        "pong",
        "generation_stopped",
    ]
    assert stopped == [True]
    assert fake_manager.disconnected == [websocket]
