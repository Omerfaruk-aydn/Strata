"""Strata Ultra runtime APIs.

These endpoints expose the independent low-bit primitives without pretending
that a standard GGUF file has already been converted to the experimental
format. Conversion and native execution are deliberately separate milestones.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from ..core.strata_ultra import convert_gguf_to_strata, kv_memory_report, run_codec_benchmark
from ..models.model_manager import model_manager
from .auth import require_api_access

router = APIRouter(
    prefix="/api/ultra",
    tags=["strata-ultra"],
    dependencies=[Depends(require_api_access)],
)


class MemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_count: int = Field(default=4096, ge=1, le=10_000_000_000)
    group_size: int = Field(default=128, ge=8, le=4096)


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_count: int = Field(default=16_384, ge=128, le=10_000_000)
    group_size: int = Field(default=128, ge=8, le=4096)


class PagingPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer_sizes_bytes: List[int] = Field(min_length=1, max_length=100_000)
    memory_budget_bytes: int = Field(ge=1)
    resident_window: int = Field(default=2, ge=1, le=100_000)


class ConvertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_name: Optional[str] = Field(default=None, max_length=255, pattern=r"^[A-Za-z0-9._-]+$")
    group_size: int = Field(default=128, ge=8, le=4096)


@router.get("/capabilities")
async def capabilities():
    return {
        "runtime": "strata-ultra",
        "format_version": 1,
        "weight_codecs": ["ternary-q05"],
        "kv_cache_modes": ["sign1", "ternary05"],
        "features": ["bit-packing", "group-scales", "layer-paging", "benchmark"],
        "status": "experimental",
    }


@router.post("/memory")
async def memory(request: MemoryRequest):
    return {"report": kv_memory_report(request.value_count, request.group_size)}


@router.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    return {"benchmark": run_codec_benchmark(request.value_count, request.group_size)}


@router.post("/paging-plan")
async def paging_plan(request: PagingPlanRequest):
    if any(size <= 0 for size in request.layer_sizes_bytes):
        return {"feasible": False, "reason": "Katman boyutları pozitif olmalıdır."}
    largest = max(request.layer_sizes_bytes)
    if largest > request.memory_budget_bytes:
        return {
            "feasible": False,
            "reason": "En büyük katman bellek bütçesini aşıyor.",
            "largest_layer_bytes": largest,
        }
    resident = []
    used = 0
    for index, size in enumerate(request.layer_sizes_bytes):
        if len(resident) >= request.resident_window or used + size > request.memory_budget_bytes:
            break
        resident.append(index)
        used += size
    return {
        "feasible": True,
        "total_layers": len(request.layer_sizes_bytes),
        "resident_window": request.resident_window,
        "initial_resident_layers": resident,
        "initial_resident_bytes": used,
        "memory_budget_bytes": request.memory_budget_bytes,
        "paging_required": len(resident) < len(request.layer_sizes_bytes),
    }


@router.post("/convert/{model_id:path}")
async def convert_model(model_id: str, request: ConvertRequest):
    """Convert a local GGUF model into a sibling .strata file."""
    candidates = [model for model in model_manager.get_local_models() if model.id == model_id]
    if not candidates:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Yerel model bulunamadı.")
    model = candidates[0]
    source = Path(model.local_path).resolve()
    root = Path(model_manager.model_dir).resolve()
    if root not in source.parents or source.suffix.lower() != ".gguf":
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Model yolu güvenli bir GGUF dosyası değil.")
    target_name = request.target_name or f"{source.stem}-STRATA-Q0.5.strata"
    target = (root / target_name).resolve()
    if target.parent != root or target.suffix.lower() != ".strata":
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Çıktı yalnızca model klasöründeki .strata dosyası olabilir.")
    if target.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Strata çıktı dosyası zaten mevcut.")
    try:
        result = await __import__("asyncio").to_thread(
            convert_gguf_to_strata, source, target, group_size=request.group_size
        )
        return {"conversion": result}
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc)) from exc
