"""
AI Runner — WebSocket Telemetry
Real-time telemetry streaming via WebSocket.
Implements FR-402, FR-403 live data feeds.
"""

import asyncio
import json
import time
from typing import Set, Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from .auth import websocket_access_allowed

logger = logging.getLogger(__name__)
router = APIRouter(tags=["telemetry"])


class TelemetryManager:
    """Manages WebSocket connections and telemetry broadcasting."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._interval = 1.0  # seconds between telemetry updates

    async def connect(self, websocket: WebSocket):
        requested_protocols = websocket.headers.get("sec-websocket-protocol", "")
        subprotocol = "ai-runner" if "ai-runner" in {
            part.strip() for part in requested_protocols.split(",")
        } else None
        await websocket.accept(subprotocol=subprotocol)
        self._connections.add(websocket)
        logger.info(f"WebSocket connected. Active: {len(self._connections)}")

        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket):
        self._connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Active: {len(self._connections)}")

        if not self._connections and self._running:
            self._running = False
            if self._task:
                self._task.cancel()

    async def broadcast(self, data: Dict[str, Any]):
        """Send data to all connected clients."""
        if not self._connections:
            return

        message = json.dumps(data)
        disconnected = set()

        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self._connections.discard(ws)

    async def send_to(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to a specific client."""
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            self._connections.discard(websocket)

    async def _broadcast_loop(self):
        """Periodically broadcast telemetry snapshots."""
        while self._running and self._connections:
            try:
                snapshot = self._collect_telemetry()
                await self.broadcast({
                    "type": "telemetry",
                    "data": snapshot,
                })
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry broadcast error: {e}")
                await asyncio.sleep(self._interval)

    def _collect_telemetry(self) -> Dict[str, Any]:
        """Collect current telemetry snapshot (Section 9: TelemetrySnapshot)."""
        from ..core.hardware_profile import detect_gpus, detect_ram
        from ..core.inference_engine import engine

        # GPU/RAM stats
        gpus = detect_gpus()
        ram = detect_ram()

        gpu_data = {}
        if gpus:
            selected_gpu = engine.model_info.main_gpu if engine.model_info else 0
            gpu = gpus[selected_gpu] if 0 <= selected_gpu < len(gpus) else gpus[0]
            gpu_data = {
                "vram_used_mb": gpu.vram_used_mb,
                "vram_total_mb": gpu.vram_total_mb,
                "vram_free_mb": gpu.vram_free_mb,
                "temperature": gpu.temperature,
            }

        # Engine stats
        engine_data = {
            "model_loaded": engine.is_loaded,
            "is_generating": engine.is_generating,
        }

        if engine.model_info:
            engine_data["model_id"] = engine.model_info.model_id
            engine_data["n_gpu_layers"] = engine.model_info.n_gpu_layers
            engine_data["context_length"] = engine.model_info.context_length

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": gpu_data,
            "ram": {
                "used_mb": ram.used_mb,
                "total_mb": ram.total_mb,
                "free_mb": ram.free_mb,
                "percent": ram.percent_used,
            },
            "engine": engine_data,
        }


# Singleton
telemetry_manager = TelemetryManager()


# ── WebSocket Endpoint ──

@router.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry and token streaming."""
    allowed, close_code = await websocket_access_allowed(websocket)
    if not allowed:
        await websocket.close(code=close_code)
        return
    await telemetry_manager.connect(websocket)

    try:
        while True:
            # Listen for client messages (e.g., commands)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await telemetry_manager.send_to(websocket, {
                        "type": "pong",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })

                elif msg_type == "stop":
                    from ..core.inference_engine import engine
                    engine.stop_generation()
                    await telemetry_manager.send_to(websocket, {
                        "type": "generation_stopped",
                    })

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        telemetry_manager.disconnect(websocket)
