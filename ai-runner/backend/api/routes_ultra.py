"""Strata Ultra runtime APIs.

These endpoints expose the independent low-bit primitives without pretending
that a standard GGUF file has already been converted to the experimental
format. Conversion and native execution are deliberately separate milestones.
"""

import asyncio
import importlib.util
import json
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..core.strata_ultra import (
    StrataContainerReader,
    StrataGraph,
    StrataRuntime,
    LinearNode,
    LowBitAttention,
    LowBitTransformer,
    LowBitTransformerBlock,
    GGUFTokenizer,
    GenerationConfig,
    StrataGenerator,
    StrataChatMessage,
    format_chat_prompt,
    discover_layout,
    tensor_quality,
    convert_gguf_to_strata,
    kv_memory_report,
    run_codec_benchmark,
)
from ..core.strata_ultra.cuda_backend import cuda_available
from ..core.strata_ultra.iq_registry import capability_report as iq_capability_report, source_codec_names
from ..core.strata_ultra.iq_native import native_iq_available
from ..models.model_manager import model_manager
from .auth import require_api_access

router = APIRouter(
    prefix="/api/ultra",
    tags=["strata-ultra"],
    dependencies=[Depends(require_api_access)],
)

_strata_generation_state_lock = threading.Lock()
_strata_generation_cancel: threading.Event | None = None


@contextmanager
def _strata_generator_context(request: "GenerateRequest", cancel_event: threading.Event):
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise FileNotFoundError("Strata model dosyası bulunamadı.")
    with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
        with StrataContainerReader(model_path) as reader:
            tensor_names = set(reader.tensor_names())
            discovered = discover_layout(list(tensor_names))
            tokenizer_metadata = reader.manifest.get("metadata", {}).get("tokenizer_metadata", {})
        if request.embedding_tensor not in tensor_names or request.output_tensor not in tensor_names:
            raise ValueError("embedding veya output tensor bulunamadı")
        prefixes = list(request.block_prefixes) or [item["prefix"] for item in discovered["blocks"] if item["complete"]]
        blocks = []
        for prefix in prefixes:
            names = {part: f"{prefix}.{part}" for part in ("q", "k", "v", "o", "gate", "up", "down")}
            blocks.append(LowBitTransformerBlock(
                runtime, q_proj=names["q"], k_proj=names["k"], v_proj=names["v"], o_proj=names["o"],
                gate_proj=names["gate"], up_proj=names["up"], down_proj=names["down"], width=request.width,
                context_capacity=request.context_capacity, kv_mode=request.kv_mode,
            ))
        if not blocks:
            raise ValueError("no complete transformer blocks discovered")
        tokenizer = None
        tokenizer_name = "byte-fallback"
        if tokenizer_metadata:
            try:
                tokenizer = GGUFTokenizer.from_metadata(tokenizer_metadata)
                tokenizer_name = "gguf-bpe"
            except (ValueError, RuntimeError):
                pass
        yield StrataGenerator(
            runtime, LowBitTransformer(blocks), request.embedding_tensor, request.output_tensor, tokenizer=tokenizer,
        ), tokenizer_name, len(blocks), runtime.backend


class MemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_count: int = Field(default=4096, ge=1, le=10_000_000_000)
    group_size: int = Field(default=128, ge=8, le=4096)
    sparse_nonzero_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_count: int = Field(default=16_384, ge=128, le=10_000_000)
    group_size: int = Field(default=128, ge=8, le=4096)
    sparse_threshold: float = Field(default=0.125, ge=0.0, le=10.0)


class PagingPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer_sizes_bytes: List[int] = Field(min_length=1, max_length=100_000)
    memory_budget_bytes: int = Field(ge=1)
    resident_window: int = Field(default=2, ge=1, le=100_000)


class ConvertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_name: Optional[str] = Field(default=None, max_length=255, pattern=r"^[A-Za-z0-9._-]+$")
    group_size: int = Field(default=128, ge=8, le=4096)
    target_codec: str = Field(default="ternary-q05", pattern=r"^(ternary-q05|sparse05)$")
    sparse_threshold: float = Field(default=0.125, ge=0.0, le=10.0)


class MatvecRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    tensor_name: str = Field(min_length=1, max_length=1024)
    vector: List[float] = Field(min_length=1, max_length=1_000_000)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy|cuda)$")


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
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy|cuda)$")
    prefetch: bool = True


class AttentionStepRequest(BaseModel):
    width: int = Field(ge=1, le=16_384)
    capacity_tokens: int = Field(default=2048, ge=1, le=1_000_000)
    mode: str = Field(default="sign1", pattern=r"^(sign1|ternary05|sparse05)$")
    sparse_threshold: float = Field(default=0.125, ge=0.0, le=10.0)
    query: List[float] = Field(min_length=1, max_length=16_384)
    key: List[float] = Field(min_length=1, max_length=16_384)
    value: List[float] = Field(min_length=1, max_length=16_384)


class QualityRequest(BaseModel):
    reference: List[float] = Field(min_length=1, max_length=10_000_000)
    reconstructed: List[float] = Field(min_length=1, max_length=10_000_000)


class TransformerStepRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    block_prefixes: List[str] = Field(min_length=1, max_length=1024)
    width: int = Field(ge=1, le=16_384)
    context_capacity: int = Field(default=2048, ge=1, le=1_000_000)
    kv_mode: str = Field(default="sign1", pattern=r"^(sign1|ternary05|sparse05)$")
    hidden: List[float] = Field(min_length=1, max_length=16_384)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy|cuda)$")


class GenerateRequest(BaseModel):
    model_file: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+\.strata$")
    block_prefixes: List[str] = Field(default_factory=list, max_length=1024)
    embedding_tensor: str = Field(min_length=1, max_length=1024)
    output_tensor: str = Field(min_length=1, max_length=1024)
    width: int = Field(ge=1, le=16_384)
    context_capacity: int = Field(default=2048, ge=1, le=1_000_000)
    kv_mode: str = Field(default="sign1", pattern=r"^(sign1|ternary05|sparse05)$")
    prompt: str = Field(default="", max_length=100_000)
    max_new_tokens: int = Field(default=16, ge=1, le=1024)
    stop_token_ids: List[int] = Field(default_factory=list, max_length=64)
    timeout_s: float = Field(default=300.0, ge=0.0, le=86_400.0)
    memory_budget_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    resident_window: int = Field(default=2, ge=1, le=1024)
    backend: str = Field(default="auto", pattern=r"^(auto|python|numpy|cuda)$")


class StrataChatMessageRequest(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=100_000)


class StrataChatRequest(GenerateRequest):
    messages: List[StrataChatMessageRequest] = Field(min_length=1, max_length=100_000)
    stream: bool = False


@router.get("/capabilities")
async def capabilities():
    native_iq = native_iq_available()
    source_codecs = ["F32", "F16", "Q4_0", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
    source_codecs.extend(source_codec_names(native_bridge=native_iq))
    iq_codecs = iq_capability_report(native_bridge=native_iq)
    unsupported_iq = [item["name"] for item in iq_codecs if not item["decodable"]]
    cuda = cuda_available()
    return {
        "runtime": "strata-ultra",
        "format_version": 1,
        "weight_codecs": ["ternary-q05", "sparse05"],
        "source_gguf_codecs": source_codecs,
        "unsupported_source_codecs": unsupported_iq,
        "iq_codecs": iq_codecs,
        "native_iq_decoder": native_iq,
        "kv_cache_modes": ["sign1", "ternary05", "sparse05"],
        "features": ["bit-packing", "group-scales", "layer-paging", "benchmark", "chat-completions", "sse-generation"],
        "execution_backends": {
            "python": {"available": True, "active": True, "weight_codecs": ["ternary-q05", "sparse05"]},
            "numpy": {"available": True, "active": False, "weight_codecs": ["ternary-q05", "sparse05"]},
            "cuda": {"available": cuda, "active": False, "weight_codecs": ["ternary-q05"]},
        },
        "tokenizer_backend": "gguf-bpe" if importlib.util.find_spec("tokenizers") else "byte-fallback",
        "readiness": {
            "container_io": True,
            "python_executor": True,
            "native_cuda_executor": cuda,
            "native_iq_conversion": native_iq,
            "experimental_generation": True,
            "chat_completions_api": True,
            "sse_generation_api": True,
            "production_chat_runtime": False,
        },
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


@router.get("/layout/{model_file}")
async def ultra_layout(model_file: str):
    root = Path(model_manager.model_dir).resolve()
    path = (root / model_file).resolve()
    if path.parent != root or path.suffix.lower() != ".strata" or not path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        with StrataContainerReader(path) as reader:
            return discover_layout(reader.tensor_names())
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/inspect/{model_file}")
async def ultra_inspect(model_file: str):
    """Return a preflight summary for a Strata container before execution."""
    root = Path(model_manager.model_dir).resolve()
    path = (root / model_file).resolve()
    if path.parent != root or path.suffix.lower() != ".strata" or not path.is_file():
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        with StrataContainerReader(path) as reader:
            codec_counts: dict[str, int] = {}
            packed_bytes = 0
            scales_bytes = 0
            tensor_count = 0
            for record in reader.read_tensors():
                tensor_count += 1
                codec_counts[record.codec] = codec_counts.get(record.codec, 0) + 1
                packed_bytes += len(record.payload)
                scales_bytes += len(record.scales)
            tensor_names = reader.tensor_names()
            metadata = reader.manifest.get("metadata", {})
            layout = discover_layout(tensor_names)
            has_embedding = any("embed" in name and "weight" in name for name in tensor_names)
            has_output = any(name.endswith("output.weight") or name.endswith("lm_head.weight") for name in tensor_names)
            tokenizer_metadata = metadata.get("tokenizer_metadata", {})
            return {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "tensor_count": tensor_count,
                "codec_counts": codec_counts,
                "packed_bytes": packed_bytes,
                "scales_bytes": scales_bytes,
                "metadata": metadata,
                "layout": layout,
                "has_embedding": has_embedding,
                "has_output": has_output,
                "tokenizer_metadata_present": bool(tokenizer_metadata),
                "ready_for_experimental_generation": bool(layout["complete_blocks"] and has_embedding and has_output),
            }
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/memory")
async def memory(request: MemoryRequest):
    return {"report": kv_memory_report(request.value_count, request.group_size, request.sparse_nonzero_ratio)}


@router.post("/quality")
async def quality(request: QualityRequest):
    try:
        return {"quality": tensor_quality(request.reference, request.reconstructed)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
                    "execution_path": "python-streaming" if request.backend == "python" else "numpy-or-fallback",
                    "tensor_codec": runtime.pager.get(request.tensor_name).codec,
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
                graph = StrataGraph(runtime, [LinearNode(node.tensor_name, node.activation) for node in request.nodes], prefetch=request.prefetch)
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
        attention = LowBitAttention(request.width, request.capacity_tokens, request.mode, request.sparse_threshold)
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
                if request.block_prefixes:
                    layouts = [
                        {part: f"{prefix}.{part}" for part in ("q", "k", "v", "o", "gate", "up", "down")}
                        for prefix in request.block_prefixes
                    ]
                else:
                    with StrataContainerReader(model_path) as reader:
                        discovered = discover_layout(reader.tensor_names())
                    layouts = [item["tensors"] for item in discovered["blocks"] if item["complete"]]
                for layout in layouts:
                    blocks.append(LowBitTransformerBlock.from_layout(
                        runtime, layout, width=request.width, context_capacity=request.context_capacity, kv_mode=request.kv_mode,
                    ))
                if not blocks:
                    raise ValueError("no complete transformer blocks discovered")
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
    global _strata_generation_cancel
    cancel_event = threading.Event()
    with _strata_generation_state_lock:
        if _strata_generation_cancel is not None:
            raise HTTPException(status_code=409, detail="Başka bir Strata generation devam ediyor")
        _strata_generation_cancel = cancel_event
    root = Path(model_manager.model_dir).resolve()
    model_path = (root / request.model_file).resolve()
    if model_path.parent != root or not model_path.is_file():
        with _strata_generation_state_lock:
            _strata_generation_cancel = None
        raise HTTPException(status_code=404, detail="Strata model dosyası bulunamadı.")
    try:
        def execute():
            with StrataRuntime(model_path, request.memory_budget_bytes, request.resident_window, request.backend) as runtime:
                with StrataContainerReader(model_path) as reader:
                    tensor_names = set(reader.tensor_names())
                    discovered = discover_layout(list(tensor_names))
                    tokenizer_metadata = reader.manifest.get("metadata", {}).get("tokenizer_metadata", {})
                if request.embedding_tensor not in tensor_names:
                    raise ValueError(f"embedding tensor bulunamadı: {request.embedding_tensor}")
                if request.output_tensor not in tensor_names:
                    raise ValueError(f"output tensor bulunamadı: {request.output_tensor}")
                blocks = []
                prefixes = list(request.block_prefixes)
                if not prefixes:
                    prefixes = [item["prefix"] for item in discovered["blocks"] if item["complete"]]
                for prefix in prefixes:
                    names = {part: f"{prefix}.{part}" for part in ("q", "k", "v", "o", "gate", "up", "down")}
                    blocks.append(LowBitTransformerBlock(
                        runtime, q_proj=names["q"], k_proj=names["k"], v_proj=names["v"], o_proj=names["o"],
                        gate_proj=names["gate"], up_proj=names["up"], down_proj=names["down"], width=request.width,
                        context_capacity=request.context_capacity, kv_mode=request.kv_mode,
                    ))
                if not blocks:
                    raise ValueError("no complete transformer blocks discovered")
                tokenizer = None
                tokenizer_name = "byte-fallback"
                if tokenizer_metadata:
                    try:
                        tokenizer = GGUFTokenizer.from_metadata(tokenizer_metadata)
                        tokenizer_name = "gguf-bpe"
                    except (ValueError, RuntimeError):
                        tokenizer = None
                generator = StrataGenerator(
                    runtime, LowBitTransformer(blocks), request.embedding_tensor, request.output_tensor, tokenizer=tokenizer,
                )
                completion = generator.generate_with_metadata(
                    request.prompt,
                    GenerationConfig(
                        request.max_new_tokens,
                        stop_token_ids=tuple(request.stop_token_ids),
                        cancel_event=cancel_event,
                    ),
                )
                return {**completion, "tokenizer": tokenizer_name, "blocks": len(blocks), "backend": runtime.backend}
        task = asyncio.create_task(asyncio.to_thread(execute))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=request.timeout_s or None)
        except asyncio.TimeoutError as exc:
            cancel_event.set()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except Exception:
                pass
            raise HTTPException(status_code=504, detail="Strata generation timed out") from exc
    except (ValueError, MemoryError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        with _strata_generation_state_lock:
            if _strata_generation_cancel is cancel_event:
                _strata_generation_cancel = None


@router.post("/generate/stop")
async def stop_generate_text():
    """Request cooperative cancellation of the active Strata generation."""
    with _strata_generation_state_lock:
        active = _strata_generation_cancel
    if active is None:
        return {"status": "idle"}
    active.set()
    return {"status": "stopping"}


@router.post("/chat/completions")
async def strata_chat_completions(request: StrataChatRequest):
    """OpenAI-shaped non-streaming chat adapter for the Strata generator."""
    if request.stream:
        try:
            prompt = format_chat_prompt([
                StrataChatMessage(message.role, message.content)
                for message in request.messages
            ])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        generation_values = request.model_dump(exclude={"messages", "stream"})
        generation_values["prompt"] = prompt
        lower_stream = await strata_generate_stream(GenerateRequest(**generation_values))

        async def openai_body():
            response_id = f"strata-{int(time.time() * 1000)}"
            role_chunk = {"id": response_id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            try:
                yield f"data: {json.dumps(role_chunk)}\n\n"
                async for raw in lower_stream.body_iterator:
                    if not raw.startswith("data: "):
                        continue
                    event = json.loads(raw[6:].strip())
                    if "text" in event:
                        payload = {"id": response_id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": event["text"]}, "finish_reason": None}]}
                    elif "finish_reason" in event:
                        payload = {"id": response_id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": event["finish_reason"]}]}
                    else:
                        payload = {"id": response_id, "object": "chat.completion.chunk", "choices": [], "error": event}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                close = getattr(lower_stream.body_iterator, "aclose", None)
                if close is not None:
                    await close()

        return StreamingResponse(openai_body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    try:
        prompt = format_chat_prompt([
            StrataChatMessage(message.role, message.content)
            for message in request.messages
        ])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    generation_values = request.model_dump(exclude={"messages", "stream"})
    generation_values["prompt"] = prompt
    generation_request = GenerateRequest(**generation_values)
    completion = await generate_text(generation_request)
    generated_text = completion["text"]
    if isinstance(generated_text, str) and generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    return {
        "id": f"strata-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "model": request.model_file,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": completion["finish_reason"],
        }],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": completion["generated_tokens"],
            "total_tokens": len(prompt) + completion["generated_tokens"],
        },
        "strata": {
            "tokenizer": completion["tokenizer"],
            "blocks": completion["blocks"],
            "backend": completion["backend"],
        },
    }


@router.post("/generate/stream")
async def strata_generate_stream(request: GenerateRequest):
    """Stream Strata token events as server-sent events."""
    global _strata_generation_cancel
    cancel_event = threading.Event()
    with _strata_generation_state_lock:
        if _strata_generation_cancel is not None:
            raise HTTPException(status_code=409, detail="Başka bir Strata generation devam ediyor")
        _strata_generation_cancel = cancel_event
    events: queue.Queue[dict | None] = queue.Queue()

    def worker() -> None:
        try:
            with _strata_generator_context(request, cancel_event) as (generator, tokenizer, blocks, backend):
                for event in generator.generate_stream(
                    request.prompt,
                    GenerationConfig(
                        request.max_new_tokens,
                        stop_token_ids=tuple(request.stop_token_ids),
                        cancel_event=cancel_event,
                    ),
                ):
                    events.put({**event, "tokenizer": tokenizer, "blocks": blocks, "backend": backend})
        except Exception as exc:
            events.put({"error": str(exc)[:500], "finish_reason": "error", "generated_tokens": 0})
        finally:
            events.put(None)

    task = asyncio.create_task(asyncio.to_thread(worker))

    async def body():
        global _strata_generation_cancel
        started = time.monotonic()
        try:
            while True:
                remaining = None
                if request.timeout_s:
                    remaining = request.timeout_s - (time.monotonic() - started)
                    if remaining <= 0:
                        cancel_event.set()
                        yield f"data: {json.dumps({'finish_reason': 'timeout'})}\n\n"
                        break
                try:
                    event = await asyncio.wait_for(asyncio.to_thread(events.get), timeout=remaining)
                except asyncio.TimeoutError:
                    cancel_event.set()
                    yield f"data: {json.dumps({'finish_reason': 'timeout'})}\n\n"
                    break
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        finally:
            cancel_event.set()
            with _strata_generation_state_lock:
                if _strata_generation_cancel is cancel_event:
                    _strata_generation_cancel = None
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # The native/CPU worker is cooperative, but cleanup must not
                # hold an HTTP connection forever if an optional backend is
                # slow to observe the cancellation event.
                pass

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    return {"benchmark": run_codec_benchmark(request.value_count, request.group_size, request.sparse_threshold)}


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
            convert_gguf_to_strata, source, target, group_size=request.group_size, target_codec=request.target_codec,
            sparse_threshold=request.sparse_threshold
        )
        return {"conversion": result}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
