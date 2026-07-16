"""Extreme Model Mode APIs for feasibility, benchmarking, and quantization."""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..core.extreme_model import (
    PRESETS,
    PresetName,
    analyze_feasibility,
    estimated_specification,
    hardware_fingerprint,
    model_path_fingerprint,
    specification_from_gguf,
)
from ..core.hardware_profile import detect_gpus, get_hardware_profile
from ..core.inference_engine import InferenceParams, engine
from ..core.quantization_service import SUPPORTED_OUTPUT_QUANTS, quantization_manager
from ..core.runtime_capabilities import detect_runtime_capabilities, validate_backend_preference
from ..db import session_store
from ..models.model_manager import model_manager
from .auth import require_api_access


router = APIRouter(
    prefix="/api/extreme",
    tags=["extreme-model"],
    dependencies=[Depends(require_api_access)],
)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quant: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,40}$")
    preset: PresetName = "maximum_capacity"
    context_length: int = Field(default=2048, ge=512, le=1_048_576)
    selected_gpu_index: int = Field(default=0, ge=0, le=128)
    tensor_split: Optional[List[float]] = Field(default=None, max_length=128)


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=500)
    parameter_count: int = Field(default=0, ge=0, le=10_000_000_000_000)
    quant: str = Field(default="Q4_K_M", pattern=r"^[A-Za-z0-9_.-]{1,40}$")
    file_size_mb: Optional[float] = Field(default=None, gt=0, le=10_000_000)
    total_layers: Optional[int] = Field(default=None, ge=1, le=10_000)
    native_context_length: int = Field(default=4096, ge=512, le=1_048_576)
    preset: PresetName = "maximum_capacity"
    context_length: int = Field(default=2048, ge=512, le=1_048_576)
    selected_gpu_index: int = Field(default=0, ge=0, le=128)


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tokens: int = Field(default=32, ge=8, le=256)
    prompt: str = Field(
        default="Explain in three concise sentences why local language models are useful.",
        min_length=1,
        max_length=4000,
    )


class RebalanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: PresetName = "maximum_capacity"
    context_length: Optional[int] = Field(default=None, ge=512, le=1_048_576)
    force: bool = False


class QuantizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quant: str
    source_quant: Optional[str] = None
    threads: Optional[int] = Field(default=None, ge=1, le=1024)
    allow_requantize: bool = False


def _local_model(model_id: str, quant: Optional[str] = None):
    candidates = [model for model in model_manager.get_local_models() if model.id == model_id]
    if quant:
        candidates = [model for model in candidates if (model.downloaded_quant or "").upper() == quant.upper()]
    if not candidates:
        raise HTTPException(status_code=404, detail="Yerel model bulunamadı.")
    if len(candidates) > 1 and not quant:
        raise HTTPException(status_code=409, detail="Birden fazla quant bulundu; analiz için quant seçin.")
    return candidates[0]


async def _effective_runtime(hardware, *, refresh: bool = False):
    runtime = await asyncio.to_thread(
        detect_runtime_capabilities,
        hardware,
        refresh=refresh,
    )
    preference = str(await session_store.get_setting("backend_preference", "auto"))
    validate_backend_preference(preference, runtime)
    force_cpu = preference == "cpu"
    if force_cpu:
        runtime = runtime.model_copy(update={
            "active_backend": "cpu",
            "gpu_offload_supported": False,
        })
    return runtime, force_cpu, preference


@router.get("/capabilities")
async def capabilities(refresh: bool = Query(False)):
    selected_gpu = int(await session_store.get_setting("selected_gpu_index", 0))
    hardware = await asyncio.to_thread(
        get_hardware_profile,
        model_manager.model_dir,
        selected_gpu,
    )
    report = await asyncio.to_thread(
        detect_runtime_capabilities,
        hardware,
        refresh=refresh,
    )
    return report.model_dump()


@router.get("/presets")
async def presets():
    return {"presets": [preset.model_dump() for preset in PRESETS.values()]}


@router.post("/analyze/{model_id:path}")
async def analyze_local_model(model_id: str, request: AnalyzeRequest):
    model = _local_model(model_id, request.quant)
    try:
        spec, metadata = await asyncio.to_thread(
            specification_from_gguf,
            model.id,
            model.local_path,
            model.downloaded_quant or request.quant or "Q4_K_M",
            model.parameter_count,
        )
        hardware = await asyncio.to_thread(
            get_hardware_profile,
            model_manager.model_dir,
            request.selected_gpu_index,
        )
        runtime, force_cpu, _ = await _effective_runtime(hardware)
        report = analyze_feasibility(
            spec,
            hardware,
            runtime,
            preset_name=request.preset,
            requested_context_length=request.context_length,
            selected_gpu_index=request.selected_gpu_index,
            tensor_split=request.tensor_split,
            force_cpu=force_cpu,
        )
        return {"report": report.model_dump(), "metadata": metadata.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/simulate")
async def simulate_model(request: SimulateRequest):
    try:
        spec = estimated_specification(
            request.model_id,
            request.quant,
            request.parameter_count,
            file_size_mb=request.file_size_mb,
            total_layers=request.total_layers,
            context_length=request.native_context_length,
        )
        hardware = await asyncio.to_thread(
            get_hardware_profile,
            model_manager.model_dir,
            request.selected_gpu_index,
        )
        runtime, force_cpu, _ = await _effective_runtime(hardware)
        report = analyze_feasibility(
            spec,
            hardware,
            runtime,
            preset_name=request.preset,
            requested_context_length=request.context_length,
            selected_gpu_index=request.selected_gpu_index,
            force_cpu=force_cpu,
        )
        return {"report": report.model_dump()}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _sample_benchmark_usage(stop: asyncio.Event, samples: dict) -> None:
    process = psutil.Process(os.getpid())
    while not stop.is_set():
        try:
            samples["process_ram_peak_mb"] = max(
                samples.get("process_ram_peak_mb", 0.0),
                process.memory_info().rss / (1024 * 1024),
            )
            gpus = await asyncio.to_thread(detect_gpus)
            if gpus:
                samples["system_vram_peak_mb"] = max(
                    samples.get("system_vram_peak_mb", 0.0),
                    float(sum(gpu.vram_used_mb for gpu in gpus)),
                )
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass


@router.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    if not engine.is_loaded or not engine.model_info:
        raise HTTPException(status_code=409, detail="Benchmark için önce bir model yükleyin.")
    if engine.is_generating:
        raise HTTPException(status_code=409, detail="Başka bir üretim devam ederken benchmark çalıştırılamaz.")

    stop = asyncio.Event()
    samples: dict = {}
    sampler = asyncio.create_task(_sample_benchmark_usage(stop, samples))
    started = time.perf_counter()
    done_result = None
    try:
        async for event in engine.generate_streaming(
            [{"role": "user", "content": request.prompt}],
            InferenceParams(temperature=0.0, top_p=1.0, max_tokens=request.max_tokens),
        ):
            if event.get("type") == "error":
                raise RuntimeError(event.get("error", "Benchmark failed"))
            if event.get("type") == "done":
                done_result = event.get("result")
    finally:
        stop.set()
        await sampler

    if not done_result:
        raise HTTPException(status_code=500, detail="Benchmark sonuç üretmedi.")
    benchmark_result = {
        "model_id": engine.model_info.model_id,
        "measured_at": int(time.time()),
        "tokens_generated": done_result["tokens_generated"],
        "tokens_per_second": done_result["tokens_per_sec"],
        "ttft_ms": done_result["ttft_ms"],
        "total_time_ms": done_result["total_time_ms"],
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 1),
        "process_ram_peak_mb": round(samples.get("process_ram_peak_mb", 0.0), 1),
        "system_vram_peak_mb": round(samples.get("system_vram_peak_mb", 0.0), 1),
        "n_gpu_layers": engine.model_info.n_gpu_layers,
        "context_length": engine.model_info.context_length,
        "note": "VRAM is system-wide usage; RAM is measured for the AI Runner backend process.",
    }

    hardware = await asyncio.to_thread(
        get_hardware_profile,
        model_manager.model_dir,
        engine.model_info.main_gpu,
    )
    runtime = await asyncio.to_thread(detect_runtime_capabilities, hardware)
    profile_backend = (
        "cpu"
        if engine.config and engine.config.backend_preference == "cpu"
        else runtime.active_backend
    )
    fingerprint = hardware_fingerprint(hardware, profile_backend)
    path_hash = model_path_fingerprint(engine.model_info.model_path)
    profile = await session_store.get_runtime_profile(
        engine.model_info.model_id,
        path_hash,
        fingerprint,
    )
    if profile:
        await session_store.save_runtime_benchmark(profile["id"], benchmark_result)
    return {"benchmark": benchmark_result}


@router.post("/rebalance")
async def rebalance(request: RebalanceRequest):
    """Re-plan and safely reload between generations when memory availability changes."""
    if not engine.is_loaded or not engine.model_info or not engine.config:
        raise HTTPException(status_code=409, detail="Yeniden dengeleme için önce bir model yükleyin.")
    if engine.is_generating:
        raise HTTPException(status_code=409, detail="Üretim sürerken katman dağılımı değiştirilemez.")

    info = engine.model_info
    model = next(
        (
            item for item in model_manager.get_local_models()
            if os.path.abspath(item.local_path or "") == os.path.abspath(info.model_path)
        ),
        None,
    )
    if not model:
        raise HTTPException(status_code=404, detail="Yüklü model yerel kütüphanede bulunamadı.")

    hardware = await asyncio.to_thread(
        get_hardware_profile,
        model_manager.model_dir,
        info.main_gpu,
    )
    capabilities, force_cpu, backend_preference = await _effective_runtime(hardware, refresh=True)
    spec, _ = await asyncio.to_thread(
        specification_from_gguf,
        model.id,
        model.local_path,
        model.downloaded_quant or "Q4_K_M",
        model.parameter_count,
    )
    report = analyze_feasibility(
        spec,
        hardware,
        capabilities,
        preset_name=request.preset,
        requested_context_length=request.context_length or engine.config.context_length,
        selected_gpu_index=info.main_gpu,
        tensor_split=engine.config.tensor_split,
        force_cpu=force_cpu,
    )
    if report.status == "blocked":
        raise HTTPException(status_code=422, detail={"blockers": report.blockers, "report": report.model_dump()})
    if report.runtime.n_gpu_layers == info.n_gpu_layers and not request.force:
        return {"status": "unchanged", "report": report.model_dump(), "load_report": None}

    previous_config = engine.config.model_copy(deep=True)
    updated = previous_config.model_copy(deep=True)
    updated.n_gpu_layers = report.runtime.n_gpu_layers
    updated.context_length = report.runtime.context_length
    updated.n_batch = report.runtime.n_batch
    updated.use_mmap = report.runtime.use_mmap
    updated.use_mlock = report.runtime.use_mlock
    updated.kv_cache_type = report.runtime.kv_cache_type
    updated.flash_attn = report.runtime.flash_attn
    updated.tensor_split = report.runtime.tensor_split
    updated.backend_preference = backend_preference

    try:
        loaded_info, load_report = await asyncio.to_thread(
            engine.load_model_adaptive,
            model.id,
            model.local_path,
            updated,
            max_attempts=report.runtime.max_load_attempts,
        )
    except Exception as exc:
        failed_report = engine.last_load_report.model_dump() if engine.last_load_report else None
        rollback_report = None
        rollback_error = None
        try:
            _, restored = await asyncio.to_thread(
                engine.load_model_adaptive,
                model.id,
                model.local_path,
                previous_config,
                max_attempts=3,
            )
            rollback_report = restored.model_dump()
        except Exception as restore_exc:
            rollback_error = str(restore_exc)
        raise HTTPException(status_code=500, detail={
            "message": (
                "Rebalance failed; the previous configuration was restored."
                if rollback_report
                else "Rebalance failed and the previous configuration could not be restored."
            ),
            "error": str(exc),
            "load_report": failed_report,
            "rollback_report": rollback_report,
            "rollback_error": rollback_error,
        }) from exc

    path_hash = model_path_fingerprint(model.local_path)
    fingerprint = hardware_fingerprint(hardware, capabilities.active_backend)
    profile = await session_store.save_runtime_profile(
        model_id=model.id,
        model_path_hash=path_hash,
        hardware_fingerprint=fingerprint,
        backend=capabilities.active_backend,
        preset=request.preset,
        config=(engine.config or updated).model_dump(),
        load_report=load_report.model_dump(),
    )
    return {
        "status": "rebalanced",
        "model": loaded_info.model_dump(),
        "report": report.model_dump(),
        "load_report": load_report.model_dump(),
        "runtime_profile": profile,
    }


@router.get("/profiles")
async def profiles(model_id: Optional[str] = Query(default=None, max_length=500)):
    return {"profiles": await session_store.list_runtime_profiles(model_id)}


@router.get("/quantization")
async def quantization_status():
    selected_gpu = int(await session_store.get_setting("selected_gpu_index", 0))
    hardware = await asyncio.to_thread(get_hardware_profile, model_manager.model_dir, selected_gpu)
    runtime = await asyncio.to_thread(detect_runtime_capabilities, hardware)
    jobs = quantization_manager.list_jobs()
    for job in jobs:
        if job.status == "completed" and not model_manager._load_model_cache(os.path.basename(job.output_path)):
            try:
                await asyncio.to_thread(
                    model_manager.register_local_model,
                    job.model_id,
                    job.output_quant,
                    job.output_path,
                    source="quantized",
                )
            except Exception as exc:
                job.status = "failed"
                job.error = f"Output registration failed: {str(exc)[:1500]}"
                job.message = "Quantized output could not be registered"
    return {
        "available": bool(runtime.llama_quantize_path),
        "executable": runtime.llama_quantize_path,
        "supported_quants": SUPPORTED_OUTPUT_QUANTS,
        "jobs": [job.model_dump() for job in jobs],
    }


@router.post("/quantization/start/{model_id:path}")
async def start_quantization(model_id: str, request: QuantizeRequest):
    model = _local_model(model_id, request.source_quant)
    if engine.model_info and os.path.abspath(engine.model_info.model_path) == os.path.abspath(model.local_path):
        raise HTTPException(status_code=409, detail="Quantize etmeden önce kaynak modeli bellekten çıkarın.")
    hardware = await asyncio.to_thread(get_hardware_profile, model_manager.model_dir, 0)
    runtime = await asyncio.to_thread(detect_runtime_capabilities, hardware)
    if not runtime.llama_quantize_path:
        raise HTTPException(
            status_code=409,
            detail="llama-quantize bulunamadı. AI_RUNNER_LLAMA_QUANTIZE ile güvenilir yürütülebilir dosyayı tanımlayın.",
        )
    try:
        job = await quantization_manager.start_job(
            executable=runtime.llama_quantize_path,
            model_id=model.id,
            source_path=model.local_path,
            model_dir=model_manager.model_dir,
            output_quant=request.quant,
            threads=request.threads or max(1, hardware.cpu.cores),
            allow_requantize=request.allow_requantize,
        )
        return {"job": job.model_dump()}
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/quantization/jobs/{job_id}/cancel")
async def cancel_quantization(job_id: str):
    try:
        job = await quantization_manager.cancel_job(job_id)
        return {"job": job.model_dump()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Quantization job not found") from exc
