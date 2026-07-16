"""Strata Ultra runtime APIs.

These endpoints expose the independent low-bit primitives without pretending
that a standard GGUF file has already been converted to the experimental
format. Conversion and native execution are deliberately separate milestones.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..core.strata_ultra import (
    StrataContainerReader,
    StrataGraph,
    StrataRuntime,
    LinearNode,
    LowBitAttention,
    LowBitTransformer,
    LowBitTransformerBlock,
    GenerationConfig,
    StrataGenerator,
    convert_gguf_to_strata,
    kv_memory_report,
    run_codec_benchmark,
)
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


class MatvecRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    tensor_name: str = Field(min_length=1, max_length=1024)
    vector: List[float] = Field(min_length=1, max_length=1_000_000)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy)$")


class RuntimeBenchmarkRequest(MatvecRequest):
    iterations: int = Field(default=10, ge=1, le=10_000)


class GraphNodeRequest(BaseModel):
    tensor_name: str = Field(min_length=1, max_length=1024)
    activation: str = Field(default="none", pattern=r"^(none|relu)$")


class GraphRunRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    nodes: List[GraphNodeRequest] = Field(min_length=1, max_length=10_000)
    vector: List[float] = Field(min_length=1, max_length=1_000_000)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy)$")


class AttentionStepRequest(BaseModel):
    width: int = Field(ge=1, le=16_384)
    capacity_tokens: int = Field(default=2048, ge=1, le=1_000_000)
    mode: str = Field(default="sign1", pattern=r"^(sign1|ternary05)$")
    query: List[float] = Field(min_length=1, max_length=16_384)
    key: List[float] = Field(min_length=1, max_length=16_384)
    value: List[float] = Field(min_length=1, max_length=16_384)


class TransformerStepRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    block_prefixes: List[str] = Field(min_length=1, max_length=1024)
    width: int = Field(ge=1, le=16_384)
    context_capacity: int = Field(default=2048, ge=1, le=1_000_000)
    kv_mode: str = Field(default="sign1", pattern=r"^(sign1|ternary05)$")
    hidden: List[float] = Field(min_length=1, max_length=16_384)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy)$")


class GenerateRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    block_prefixes: List[str] = Field(min_length=1, max_length=1024)
    embedding_tensor: str = Field(min_length=1, max_length=1024)
    output_tensor: str = Field(min_length=1, max_length=1024)
    width: int = Field(ge=1, le=16_384)
    context_capacity: int = Field(default=2048, ge=1, le=1_000_000)
    kv_mode: str = Field(default="sign1", pattern=r"^(sign1|ternary05)$")
    prompt: str = Field(default="", max_length=100_000)
    max_new_tokens: int = Field(default=16, ge=1, le=1024)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy)$")


@router.get("/capabilities")
async def capabilities():
    return {
        "runtime": "strata-ultra",
        "format_version": 1,
        "weight_codecs": ["ternary-q05"],
        "source_gguf_codecs": ["F32", "F16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"],
        "unsupported_source_codecs": ["IQ1", "IQ2", "IQ3", "Q2_K", "Q3_K"],
        "kv_cache_modes": ["sign1", "ternary05"],
        "features": ["bit-packing", "group-scales", "layer-paging", "benchmark"],
        "status": "experimental",
    }


@router.get("/models")
async def ultra_models():
    """List validated Strata containers in the configured model directory."""
    root = Path(model_manager.model_dir).resolve()
    models = []
    for path in sorted(root.glob("*.strata")):
        try:
            with StrataContainerReader(path) as reader:
                models.append({
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "tensor_count": len(reader.tensor_names()),
                    "metadata": reader.manifest.get("metadata", {}),
                    "valid": True,
                })
        except (OSError, ValueError, KeyError) as exc:
            models.append({"file": path.name, "valid": False, "error": str(exc)[:500]})
    return {"models": models, "directory": str(root)}


@router.post("/memory")
async def memory(request: MemoryRequest):
    return {"report": kv_memory_report(request.value_count, request.group_size)}


@router.post("/matvec")
async def runtime_matvec(request: MatvecRequest):
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        def execute():
            with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
                result = runtime.tensor_matvec(request.tensor_name, request.vector)
                return {"values": result, "pager": {
                    "resident_pages": runtime.pager.resident_pages,
                    "resident_bytes": runtime.pager.resident_bytes,
                }}
        return await asyncio.to_thread(execute)
    except (ValueError, MemoryError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/runtime-benchmark")
async def runtime_benchmark(request: RuntimeBenchmarkRequest):
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        def execute():
            with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
                started = time.perf_counter()
                result = None
                for _ in range(request.iterations):
                    result = runtime.tensor_matvec(request.tensor_name, request.vector)
                elapsed_ms = (time.perf_counter() - started) * 1000
                return {
                    "iterations": request.iterations,
                    "output_length": len(result or []),
                    "total_time_ms": round(elapsed_ms, 3),
                    "average_time_ms": round(elapsed_ms / request.iterations, 3),
                    "pager": {
                        "resident_pages": runtime.pager.resident_pages,
                        "resident_bytes": runtime.pager.resident_bytes,
                        "events": len(runtime.pager.events),
                    },
                }
        return {"benchmark": await asyncio.to_thread(execute)}
    except (ValueError, MemoryError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/graph/run")
async def run_graph(request: GraphRunRequest):
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        def execute():
            with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
                graph = StrataGraph(runtime, [LinearNode(node.tensor_name, node.activation) for node in request.nodes])
                result = graph.run(request.vector)
                return {"values": result, "nodes": len(request.nodes), "pager": {
                    "resident_pages": runtime.pager.resident_pages,
                    "resident_bytes": runtime.pager.resident_bytes,
                    "events": len(runtime.pager.events),
                }}
        return await asyncio.to_thread(execute)
    except (ValueError, MemoryError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/attention/step")
async def attention_step(request: AttentionStepRequest):
    if not (len(request.query) == len(request.key) == len(request.value) == request.width):
        raise HTTPException(status_code=422, detail="query, key ve value width ile aynı uzunlukta olmalıdır.")
    try:
        attention = LowBitAttention(request.width, request.capacity_tokens, request.mode)
        output = attention.step(request.query, request.key, request.value)
        return {"output": output, "keys": attention.keys.snapshot().__dict__, "values": attention.values.snapshot().__dict__}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/transformer/step")
async def transformer_step(request: TransformerStepRequest):
    if len(request.hidden) != request.width:
        raise HTTPException(status_code=422, detail="hidden width ile aynı uzunlukta olmalıdır.")
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        def execute():
            with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
                blocks = []
                for prefix in request.block_prefixes:
                    names = {part: f"{prefix}.{part}" for part in ("q", "k", "v", "o", "gate", "up", "down")}
                    blocks.append(LowBitTransformerBlock(
                        runtime, q_proj=names["q"], k_proj=names["k"], v_proj=names["v"], o_proj=names["o"],
                        gate_proj=names["gate"], up_proj=names["up"], down_proj=names["down"], width=request.width,
                        context_capacity=request.context_capacity, kv_mode=request.kv_mode,
                    ))
                result = LowBitTransformer(blocks).step(request.hidden)
                return {"hidden": result, "blocks": len(blocks), "pager": {
                    "resident_pages": runtime.pager.resident_pages,
                    "resident_bytes": runtime.pager.resident_bytes,
                    "events": len(runtime.pager.events),
                }}
        return await asyncio.to_thread(execute)
    except (ValueError, MemoryError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/generate")
async def generate_text(request: GenerateRequest):
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        def execute():
            with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
                blocks = []
                for prefix in request.block_prefixes:
                    names = {part: f"{prefix}.{part}" for part in ("q", "k", "v", "o", "gate", "up", "down")}
                    blocks.append(LowBitTransformerBlock(
                        runtime, q_proj=names["q"], k_proj=names["k"], v_proj=names["v"], o_proj=names["o"],
                        gate_proj=names["gate"], up_proj=names["up"], down_proj=names["down"], width=request.width,
                        context_capacity=request.context_capacity, kv_mode=request.kv_mode,
                    ))
                generator = StrataGenerator(
                    runtime, LowBitTransformer(blocks), request.embedding_tensor, request.output_tensor,
                )
                text = generator.generate(request.prompt, GenerationConfig(request.max_new_tokens))
                return {"text": text, "tokenizer": "byte-fallback", "blocks": len(blocks), "backend": runtime.backend}
        return await asyncio.to_thread(execute)
    except (ValueError, MemoryError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
        raise HTTPException(status_code=404, detail="Yerel model bulunamadı.")
    model = candidates[0]
    source = Path(model.local_path).resolve()
    root = Path(model_manager.model_dir).resolve()
    if root not in source.parents or source.suffix.lower() != ".gguf":
        raise HTTPException(status_code=422, detail="Model yolu güvenli bir GGUF dosyası değil.")
    target_name = request.target_name or f"{source.stem}-STRATA-Q0.5.strata"
    target = (root / target_name).resolve()
    if target.parent != root or target.suffix.lower() != ".strata":
        raise HTTPException(status_code=422, detail="Çıktı yalnızca model klasöründeki .strata dosyası olabilir.")
    if target.exists():
        raise HTTPException(status_code=409, detail="Strata çıktı dosyası zaten mevcut.")
    try:
        result = await __import__("asyncio").to_thread(
            convert_gguf_to_strata, source, target, group_size=request.group_size
        )
        return {"conversion": result}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
