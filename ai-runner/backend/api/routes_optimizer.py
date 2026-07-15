"""
AI Runner — System Optimizer API Routes
Provides endpoints for system analysis, pagefile recommendations,
service audit, RAM disk setup, and prompt pruning budget.
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional, List, Dict
import logging

from ..core.system_optimizer import (
    analyze_pagefile,
    audit_services,
    get_top_processes,
    analyze_ramdisk,
    calculate_prompt_budget,
    get_optimizer_status,
    get_gpu_profiles,
    lock_cpu_affinity_and_priority,
    flush_vram_cache,
    apply_windows_performance_mode,
    create_zero_vram_launcher,
    apply_nvidia_sysmem_fallback_tweak,
)
from ..core.inference_engine import engine
from .auth import require_api_access

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/optimizer",
    tags=["optimizer"],
    dependencies=[Depends(require_api_access)],
)


@router.get("/status")
async def optimizer_status():
    """
    Overall system optimization scorecard.
    Returns score (0-100), OS info, RAM usage, and top recommendations.
    """
    try:
        status = get_optimizer_status()
        return status.model_dump()
    except Exception as e:
        logger.error(f"Optimizer status error: {e}")
        return {"error": str(e), "optimization_score": 0, "recommendations": []}


@router.get("/pagefile")
async def pagefile_info(model_size_mb: int = Query(0, description="Model file size in MB for optimal recommendation")):
    """
    ① Pagefile analysis and optimization recommendations.
    Returns current size, recommended size, PowerShell command, and status.
    """
    try:
        # Auto-detect active model size if not provided
        if model_size_mb == 0 and engine.model_info:
            import os
            try:
                model_size_mb = os.path.getsize(engine.model_info.model_path) // (1024 * 1024)
            except Exception:
                pass

        info = analyze_pagefile(model_size_mb=model_size_mb)
        return info.model_dump()
    except Exception as e:
        logger.error(f"Pagefile analysis error: {e}")
        return {"error": str(e), "status": "unavailable"}


@router.get("/services")
async def service_audit():
    """
    ② Background service audit.
    Returns list of known heavy services with status, RAM usage,
    and PowerShell stop/disable commands (NOT executed — user must run them).
    """
    try:
        services = audit_services()
        return {
            "services": [s.model_dump() for s in services],
            "note": "Bu komutlar uygulamada çalıştırılmaz. Yönetici PowerShell'de manuel çalıştırın.",
        }
    except Exception as e:
        logger.error(f"Service audit error: {e}")
        return {"services": [], "error": str(e)}


@router.get("/processes")
async def top_processes(limit: int = Query(10, ge=1, le=50)):
    """
    ② Top RAM-consuming processes.
    Returns sorted list of processes for manual analysis.
    """
    try:
        procs = get_top_processes(limit=limit)
        return {
            "processes": [p.model_dump() for p in procs],
            "total_shown": len(procs),
        }
    except Exception as e:
        logger.error(f"Process list error: {e}")
        return {"processes": [], "error": str(e)}


@router.get("/ramdisk")
async def ramdisk_info(model_size_mb: int = Query(0, description="Model file size in MB")):
    """
    ③ RAM Disk feasibility analysis and setup guide.
    Returns available space, recommended disk size, and step-by-step commands.
    """
    try:
        if model_size_mb == 0 and engine.model_info:
            import os
            try:
                model_size_mb = os.path.getsize(engine.model_info.model_path) // (1024 * 1024)
            except Exception:
                pass

        info = analyze_ramdisk(model_size_mb=model_size_mb)
        return info.model_dump()
    except Exception as e:
        logger.error(f"RAM disk analysis error: {e}")
        return {"error": str(e), "status": "unavailable"}


@router.post("/prompt-budget")
async def prompt_budget(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    context_length: int = Query(4096, ge=512, le=131072),
):
    """
    ⑤ Calculate token budget for the current conversation.
    Returns utilization percentage, remaining tokens, and warning flags.
    """
    try:
        budget = calculate_prompt_budget(
            context_length=context_length,
            history_messages=messages,
            system_prompt=system_prompt,
        )
        return budget
    except Exception as e:
        logger.error(f"Prompt budget error: {e}")
        return {"error": str(e)}


@router.get("/gpu-profile")
async def gpu_profile():
    """
    ④ GPU profiling.
    Detects multiple GPUs, recommends tensor_split, and builds nvidia-smi power tuning commands.
    """
    try:
        profile = get_gpu_profiles()
        return profile.model_dump()
    except Exception as e:
        logger.error(f"GPU profiling error: {e}")
        return {"error": str(e)}


@router.post("/affinity")
async def lock_affinity():
    """
    ② CPU affinity and Process priority locker.
    Forces physical core execution and HIGH process priority.
    """
    try:
        res = lock_cpu_affinity_and_priority()
        return res
    except Exception as e:
        logger.error(f"Affinity lock error: {e}")
        return {"error": str(e)}


@router.post("/vram-flush")
async def vram_flush():
    """
    Trim AI Runner/WebView helper-process working sets.
    """
    try:
        res = flush_vram_cache()
        return res
    except Exception as e:
        logger.error(f"VRAM flush error: {e}")
        return {"error": str(e)}


@router.post("/apply-windows-performance")
async def windows_performance():
    """
    Adjust Windows visual settings to Best Performance (Registry).
    """
    try:
        res = apply_windows_performance_mode()
        return res
    except Exception as e:
        logger.error(f"Windows performance tweak error: {e}")
        return {"error": str(e)}


@router.post("/create-launcher")
async def create_launcher():
    """
    Write a baslat_0_vram.bat shortcut script to current folder.
    """
    try:
        res = create_zero_vram_launcher()
        return res
    except Exception as e:
        logger.error(f"Bat script creation error: {e}")
        return {"error": str(e)}


@router.post("/apply-nvidia-tweak")
async def apply_nvidia_tweak():
    """
    Return safe manual guidance for NVIDIA sysmem fallback configuration.
    """
    try:
        res = apply_nvidia_sysmem_fallback_tweak()
        return res
    except Exception as e:
        logger.error(f"Nvidia tweak error: {e}")
        return {"error": str(e)}
