"""
AI Runner — Model API Routes
Implements endpoints from Section 10 for model management.
"""

import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
from ..core.inference_engine import engine, InferenceParams, EngineConfig
from ..models.model_manager import model_manager, DownloadProgress
from ..db import session_store
from .auth import require_api_access

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(require_api_access)],
)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _validate_gpu_selection(hardware, selected_gpu: int, tensor_split=None) -> None:
    gpu_count = len(hardware.gpus)
    if selected_gpu < 0 or (gpu_count and selected_gpu >= gpu_count) or (not gpu_count and selected_gpu != 0):
        raise HTTPException(status_code=422, detail="Seçilen GPU indeksi sistemde bulunmuyor.")
    if tensor_split is not None and len(tensor_split) != gpu_count:
        raise HTTPException(
            status_code=422,
            detail="tensor_split uzunluğu algılanan GPU sayısıyla aynı olmalıdır.",
        )


# ── Request/Response Models ──

class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quant: str = "Q4_K_M"
    context_length: int = Field(default=4096, ge=512, le=1_048_576)
    n_gpu_layers: Optional[int] = Field(default=None, ge=-1, le=10_000)
    selected_gpu_index: int = Field(default=0, ge=0, le=128)


class LoadRequest(BaseModel):
    """Model yükleme isteği — tüm performans optimizasyon parametrelerini içerir."""
    model_config = ConfigDict(extra="forbid")

    quant: str = "Q4_K_M"
    n_gpu_layers: Optional[int] = Field(default=None, ge=-1, le=10_000)
    context_length: int = Field(default=4096, ge=512, le=1_048_576)
    n_threads: Optional[int] = Field(default=None, ge=1, le=1_024)
    use_mmap: bool = True
    use_mlock: bool = True
    n_batch: int = Field(default=512, ge=1, le=65_536)
    # KV Cache quantization: "q4_0" | "q5_0" | "q5_1" | "q8_0" | "f16"
    kv_cache_type: str = "q4_0"
    # Flash Attention (requires compatible llama-cpp build)
    flash_attn: bool = True
    # Deprecated compatibility fields; prompt lookup decoding uses no draft model.
    draft_model_path: Optional[str] = None
    draft_n_gpu_layers: int = -1
    speculative_decoding: bool = False
    draft_num_pred_tokens: int = Field(default=10, ge=1, le=64)
    # Smart Context Shifting
    cache_context_shift: bool = True
    selected_gpu_index: int = Field(default=0, ge=0, le=128)
    tensor_split: Optional[List[float]] = Field(default=None, max_length=128)

    @field_validator("tensor_split")
    @classmethod
    def validate_tensor_split(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return None
        if not value or any(part <= 0 for part in value):
            raise ValueError("tensor_split pozitif oranlardan oluşmalıdır")
        total = sum(value)
        return [round(part / total, 6) for part in value]


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quant: str = Field(default="Q4_K_M", pattern=r"^[A-Za-z0-9_.-]{1,40}$")


# ── Endpoints ──

@router.get("/search")
async def search_models(q: str = Query("", description="Search query")):
    """FR-101: Search HuggingFace Hub for GGUF models."""
    try:
        results = await model_manager.search_models(q, limit=20)
        selected_gpu = int(await session_store.get_setting("selected_gpu_index", 0))
        hardware = get_hardware_profile(selected_gpu=selected_gpu)

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
    progress_queue = asyncio.Queue(maxsize=64)

    def on_progress(progress: DownloadProgress):
        try:
            progress_queue.put_nowait(progress)
        except asyncio.QueueFull:
            pass

    async def event_stream():
        cache_limit_gb = await session_store.get_setting("cache_size_limit_gb", 50)
        used_bytes = sum(
            os.path.getsize(os.path.join(model_manager.model_dir, name))
            for name in os.listdir(model_manager.model_dir)
            if name.lower().endswith(".gguf")
        )
        max_file_bytes = max(int(cache_limit_gb * 1024 ** 3) - used_bytes, 0)

        # Start download in background
        download_task = asyncio.create_task(
            model_manager.download_model(
                model_id=model_id,
                quant=request.quant,
                progress_callback=on_progress,
                max_file_bytes=max_file_bytes,
            )
        )
        terminal_status = None

        try:
            while not download_task.done():
                try:
                    progress = await asyncio.wait_for(
                        progress_queue.get(), timeout=1.0
                    )
                    terminal_status = progress.status if progress.status in {"completed", "paused", "error"} else terminal_status
                    yield f"data: {progress.model_dump_json()}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'status': 'downloading'})}\n\n"

            result = await download_task
            while not progress_queue.empty():
                progress = progress_queue.get_nowait()
                terminal_status = progress.status if progress.status in {"completed", "paused", "error"} else terminal_status
                yield f"data: {progress.model_dump_json()}\n\n"

            if result and terminal_status != "completed":
                yield f"data: {json.dumps({'status': 'completed', 'progress': 1.0, 'path': result})}\n\n"
            elif not result and terminal_status != "paused":
                yield f"data: {json.dumps({'status': 'paused'})}\n\n"

        except asyncio.CancelledError:
            model_manager.cancel_download(model_id)
            raise
        except Exception as e:
            if terminal_status != "error":
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/{model_id:path}/download/cancel")
async def cancel_model_download(model_id: str):
    """Pause an active download while preserving its partial file."""
    if not model_manager.cancel_download(model_id):
        raise HTTPException(status_code=409, detail="Bu model için etkin bir indirme yok.")
    return {"status": "cancelling", "model_id": model_id}


@router.get("/local")
async def list_local_models():
    """FR-105: List all locally downloaded models."""
    try:
        models = model_manager.get_local_models()
        selected_gpu = int(await session_store.get_setting("selected_gpu_index", 0))
        hardware = get_hardware_profile(selected_gpu=selected_gpu)

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
async def delete_local_model(
    model_id: str,
    quant: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9_.-]{1,40}$"),
):
    """FR-105: Delete a locally downloaded model."""
    candidates = [
        model for model in model_manager.get_local_models()
        if model.id == model_id and (quant is None or model.downloaded_quant == quant)
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail="Model bulunamadı")

    active_path = os.path.abspath(engine.model_info.model_path) if engine.model_info else None
    if any(active_path and os.path.abspath(model.local_path or "") == active_path for model in candidates):
        raise HTTPException(status_code=409, detail="Yüklü model silinmeden önce bellekten çıkarılmalıdır.")

    success = model_manager.delete_model(model_id, quant=quant)
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
        hardware = get_hardware_profile(selected_gpu=request.selected_gpu_index)
        _validate_gpu_selection(hardware, request.selected_gpu_index)

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id:path}/load")
async def load_model(model_id: str, request: LoadRequest):
    """
    Load a model into memory for inference.
    All performance optimizations (KV cache, Flash Attention, mlock,
    speculative decoding, context shifting) are forwarded via EngineConfig.
    """
    try:
        # Find the local model
        local_models = model_manager.get_local_models()
        model = next((m for m in local_models if m.id == model_id), None)

        if not model or not model.local_path:
            raise HTTPException(
                status_code=404,
                detail="Model yerel olarak bulunamadı. Önce indirin."
            )

        async def effective(request_field: str, setting_key: str):
            request_value = getattr(request, request_field)
            if request_field in request.model_fields_set:
                return request_value
            return await session_store.get_setting(setting_key, request_value)

        context_length = int(await effective("context_length", "max_context_length"))
        n_threads_value = await effective("n_threads", "n_threads")
        n_threads = int(n_threads_value) if n_threads_value is not None else None
        use_mlock = _as_bool(await effective("use_mlock", "use_mlock"))
        use_mmap = _as_bool(await effective("use_mmap", "use_mmap"))
        n_batch = int(await effective("n_batch", "n_batch"))
        kv_cache_type = str(await effective("kv_cache_type", "kv_cache_type"))
        flash_attn = _as_bool(await effective("flash_attn", "flash_attn"))
        cache_context_shift = _as_bool(await effective("cache_context_shift", "cache_context_shift"))
        auto_context_prune = _as_bool(await session_store.get_setting("auto_context_prune", True))
        speculative_decoding = _as_bool(await effective("speculative_decoding", "speculative_decoding"))
        draft_num_pred_tokens = int(await effective("draft_num_pred_tokens", "draft_num_pred_tokens"))
        selected_gpu_index = int(await effective("selected_gpu_index", "selected_gpu_index"))
        tensor_split = await effective("tensor_split", "tensor_split")

        hardware = get_hardware_profile(selected_gpu=selected_gpu_index)
        _validate_gpu_selection(hardware, selected_gpu_index, tensor_split)

        # Calculate offload plan if no manual override
        n_gpu_layers = request.n_gpu_layers
        if n_gpu_layers is None:
            file_size_mb = model.file_size_bytes / (1024 * 1024)
            total_layers = estimate_total_layers(model.parameter_count or 7_000_000_000)
            plan = calculate_offload_plan(
                model_id=model_id,
                quant=request.quant,
                file_size_mb=file_size_mb,
                total_layers=total_layers,
                hardware=hardware,
                context_length=context_length,
            )
            n_gpu_layers = plan.gpu_layers

        # Build EngineConfig with all optimizations
        config = EngineConfig(
            n_gpu_layers=n_gpu_layers,
            context_length=context_length,
            n_batch=n_batch,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            n_threads=n_threads,
            kv_cache_type=kv_cache_type,
            flash_attn=flash_attn,
            draft_model_path=request.draft_model_path,
            draft_n_gpu_layers=request.draft_n_gpu_layers,
            cache_context_shift=cache_context_shift,
            auto_context_prune=auto_context_prune,
            speculative_decoding=speculative_decoding,
            draft_num_pred_tokens=draft_num_pred_tokens,
            main_gpu=selected_gpu_index,
            tensor_split=tensor_split,
        )

        # Load the model with full optimization config
        info = await asyncio.to_thread(
            engine.load_model,
            model_id,
            model.local_path,
            config,
        )

        model_manager.update_last_used(model_id)

        return {
            "status": "loaded",
            "model": info.model_dump(),
            "optimizations": engine.get_optimization_summary(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimizations")
async def get_optimizations():
    """Return active optimization flags for the loaded model."""
    return {
        "loaded": engine.is_loaded,
        "optimizations": engine.get_optimization_summary() if engine.is_loaded else {},
    }


@router.post("/unload")
async def unload_model():
    """Unload the active model from memory."""
    await asyncio.to_thread(engine.unload_model)
    return {"status": "unloaded"}
