"""
AI Runner — Model API Routes
Implements endpoints from Section 10 for model management.
"""

import asyncio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import logging

from ..core.hardware_profile import get_hardware_profile
from ..core.memory_manager import (
    calculate_offload_plan,
    suggest_best_quant,
    estimate_model_size_mb,
    estimate_total_layers,
)
from ..core.inference_engine import engine, InferenceParams
from ..models.model_manager import model_manager, DownloadProgress

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


# ── Request/Response Models ──

class PlanRequest(BaseModel):
    quant: str = "Q4_K_M"
    context_length: int = 4096
    n_gpu_layers: Optional[int] = None


class LoadRequest(BaseModel):
    quant: str = "Q4_K_M"
    n_gpu_layers: Optional[int] = None
    context_length: int = 4096
    n_threads: Optional[int] = None
    use_mmap: bool = True
    n_batch: int = 512


class DownloadRequest(BaseModel):
    quant: str = "Q4_K_M"


# ── Endpoints ──

@router.get("/search")
async def search_models(q: str = Query("", description="Search query")):
    """FR-101: Search HuggingFace Hub for GGUF models."""
    try:
        results = await model_manager.search_models(q, limit=20)
        hardware = get_hardware_profile()

        # Add compatibility badges (FR-103)
        models_with_badges = []
        for model in results:
            badge = model_manager.get_compatibility_badge(
                model,
                vram_free_mb=hardware.gpu.vram_free_mb,
                ram_free_mb=hardware.ram.free_mb,
            )
            model_dict = model.model_dump()
            model_dict["compatibility"] = badge
            models_with_badges.append(model_dict)

        return {"models": models_with_badges}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id:path}/download")
async def download_model(model_id: str, request: DownloadRequest):
    """
    FR-102: Start a model download with SSE progress.
    Returns Server-Sent Events stream.
    """
    progress_queue = asyncio.Queue()

    def on_progress(progress: DownloadProgress):
        try:
            progress_queue.put_nowait(progress)
        except asyncio.QueueFull:
            pass

    async def event_stream():
        # Start download in background
        download_task = asyncio.create_task(
            model_manager.download_model(
                model_id=model_id,
                quant=request.quant,
                progress_callback=on_progress,
            )
        )

        try:
            while not download_task.done():
                try:
                    progress = await asyncio.wait_for(
                        progress_queue.get(), timeout=1.0
                    )
                    yield f"data: {progress.model_dump_json()}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'status': 'downloading'})}\n\n"

            # Get the result
            result = await download_task
            yield f"data: {json.dumps({'status': 'completed', 'path': result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/local")
async def list_local_models():
    """FR-105: List all locally downloaded models."""
    try:
        models = model_manager.get_local_models()
        hardware = get_hardware_profile()

        results = []
        for model in models:
            badge = model_manager.get_compatibility_badge(
                model,
                vram_free_mb=hardware.gpu.vram_free_mb,
                ram_free_mb=hardware.ram.free_mb,
            )
            model_dict = model.model_dump()
            model_dict["compatibility"] = badge
            results.append(model_dict)

        return {"models": results}
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/local/{model_id:path}")
async def delete_local_model(model_id: str):
    """FR-105: Delete a locally downloaded model."""
    success = model_manager.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model bulunamadı")
    return {"status": "deleted", "model_id": model_id}


@router.post("/{model_id:path}/plan")
async def get_offload_plan(model_id: str, request: PlanRequest):
    """
    FR-104: Calculate offload plan before download.
    Returns layer distribution and performance estimates.
    """
    try:
        hardware = get_hardware_profile()

        # Try to find model in local or search results
        local_models = model_manager.get_local_models()
        model = next((m for m in local_models if m.id == model_id), None)

        if model and model.file_size_bytes > 0:
            file_size_mb = model.file_size_bytes / (1024 * 1024)
            param_count = model.parameter_count
        else:
            # Estimate from typical model sizes
            param_count = model_manager._extract_param_count(model_id, [])
            if param_count == 0:
                param_count = 7_000_000_000  # Default 7B
            file_size_mb = estimate_model_size_mb(param_count, request.quant)

        total_layers = estimate_total_layers(param_count)

        plan = calculate_offload_plan(
            model_id=model_id,
            quant=request.quant,
            file_size_mb=file_size_mb,
            total_layers=total_layers,
            hardware=hardware,
            context_length=request.context_length,
            parameter_count=param_count,
            user_gpu_layers=request.n_gpu_layers,
        )

        return plan.model_dump()

    except Exception as e:
        logger.error(f"Plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id:path}/load")
async def load_model(model_id: str, request: LoadRequest):
    """Load a model into memory for inference."""
    try:
        # Find the local model
        local_models = model_manager.get_local_models()
        model = next((m for m in local_models if m.id == model_id), None)

        if not model or not model.local_path:
            raise HTTPException(
                status_code=404,
                detail="Model yerel olarak bulunamadı. Önce indirin."
            )

        # Calculate offload plan if no manual override
        n_gpu_layers = request.n_gpu_layers
        if n_gpu_layers is None:
            hardware = get_hardware_profile()
            file_size_mb = model.file_size_bytes / (1024 * 1024)
            total_layers = estimate_total_layers(model.parameter_count or 7_000_000_000)
            plan = calculate_offload_plan(
                model_id=model_id,
                quant=request.quant,
                file_size_mb=file_size_mb,
                total_layers=total_layers,
                hardware=hardware,
                context_length=request.context_length,
            )
            n_gpu_layers = plan.gpu_layers

        # Load the model
        info = engine.load_model(
            model_id=model_id,
            model_path=model.local_path,
            n_gpu_layers=n_gpu_layers,
            context_length=request.context_length,
            n_threads=request.n_threads,
            use_mmap=request.use_mmap,
            n_batch=request.n_batch,
        )

        model_manager.update_last_used(model_id)

        return {
            "status": "loaded",
            "model": info.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model():
    """Unload the active model from memory."""
    engine.unload_model()
    return {"status": "unloaded"}
