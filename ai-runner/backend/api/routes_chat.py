"""
AI Runner — Chat API Routes
OpenAI-compatible chat completions + session management.
Implements FR-501, FR-502, FR-701–FR-703.
"""

import json
import time
import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List, Dict, Any, Literal, Union
import logging

from ..core.inference_engine import engine, InferenceParams
from ..db import session_store
from .auth import require_api_access

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_access)])


# ── OpenAI-Compatible Models ──

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=1_000_000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="", max_length=500)
    messages: List[ChatMessage] = Field(min_length=1, max_length=100_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=10_000)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=10.0)
    max_tokens: int = Field(default=2048, ge=1, le=1_048_576)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    stream_options: Optional[Dict[str, Any]] = None

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value):
        if value is None:
            return value
        stops = [value] if isinstance(value, str) else value
        if len(stops) > 16 or any(not stop or len(stop) > 1_000 for stop in stops):
            raise ValueError("stop en fazla 16 adet, boş olmayan dize içerebilir")
        return value


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Yeni Sohbet", min_length=1, max_length=500)
    model_id: Optional[str] = Field(default=None, max_length=500)
    params: Optional[Dict[str, Any]] = None


class SessionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    model_id: Optional[str] = Field(default=None, max_length=500)
    pinned: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=1_000_000)
    session_id: str = Field(min_length=1, max_length=100)
    system_prompt: str = Field(default="", max_length=100_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=10_000)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=10.0)
    max_tokens: int = Field(default=2048, ge=1, le=1_048_576)


# ── OpenAI-Compatible Endpoints ──

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    FR-501: OpenAI-compatible chat completions.
    Supports both streaming and non-streaming modes.
    """
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model yüklü değil. Önce bir model yükleyin."
        )

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    stops = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
    params = InferenceParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        max_tokens=request.max_tokens,
        stop=stops,
    )

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(
                messages,
                params,
                include_usage=bool((request.stream_options or {}).get("include_usage")),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        return await _non_streaming_completion(messages, params, request.model)


async def _stream_chat_completion(
    messages: List[Dict[str, str]],
    params: InferenceParams,
    include_usage: bool = False,
):
    """Stream chat completion in OpenAI SSE format."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    model_id = engine.model_info.model_id if engine.model_info else "unknown"
    prompt_tokens = engine.count_prompt_tokens(messages)

    try:
        async for chunk in engine.generate_streaming(messages, params):
            if chunk["type"] == "token":
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk["content"]},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"
            elif chunk["type"] == "done":
                result = chunk.get("result", {})
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": result.get("finish_reason", "stop"),
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"
                if include_usage:
                    usage_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": result.get("tokens_generated", 0),
                            "total_tokens": prompt_tokens + result.get("tokens_generated", 0),
                        },
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            elif chunk["type"] == "error":
                error = {
                    "error": {
                        "message": chunk["error"],
                        "type": "inference_error",
                        "code": "generation_failed",
                    }
                }
                yield f"data: {json.dumps(error)}\n\n"
    finally:
        # ASGI cancellation/network disconnect must stop the worker thread;
        # otherwise the single-model engine keeps generating after the client
        # has gone away and blocks the next request.
        if engine.is_generating:
            engine.stop_generation()


async def _non_streaming_completion(
    messages: List[Dict[str, str]],
    params: InferenceParams,
    model: str,
):
    """Non-streaming chat completion in OpenAI format."""
    try:
        prompt_tokens = engine.count_prompt_tokens(messages)
        loop = asyncio.get_running_loop()
        generation_future = loop.run_in_executor(
            None,
            lambda: engine.generate_sync(messages, params)
        )
        timeout = None
        if engine._config and engine._config.generation_timeout_s > 0:
            timeout = engine._config.generation_timeout_s
        result = await asyncio.wait_for(generation_future, timeout=timeout)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_info.model_id if engine.model_info else model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.content,
                },
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "total_tokens": prompt_tokens + result.tokens_generated,
            },
        }
    except asyncio.TimeoutError as exc:
        engine.stop_generation()
        raise HTTPException(status_code=504, detail="Model generation timed out") from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models():
    """FR-502: OpenAI-compatible model list."""
    from ..models.model_manager import model_manager

    local_models = model_manager.get_local_models()

    return {
        "object": "list",
        "data": [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": model.author or "local",
            }
            for model in local_models
        ],
    }


# ── Session Management Endpoints (FR-701–FR-703) ──

@router.get("/api/sessions")
async def get_sessions():
    """Get all chat sessions."""
    sessions = await session_store.get_sessions()
    return {"sessions": sessions}


@router.post("/api/sessions")
async def create_session(request: SessionCreateRequest):
    """Create a new chat session."""
    session = await session_store.create_session(
        title=request.title,
        model_id=request.model_id,
        params=request.params,
    )
    return session


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a single session with messages."""
    session = await session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")
    return session


@router.put("/api/sessions/{session_id}")
async def update_session(session_id: str, request: SessionUpdateRequest):
    """Update a session (rename, pin, etc.)."""
    if not request.model_fields_set:
        raise HTTPException(status_code=400, detail="Güncellenecek en az bir alan gerekli.")
    success = await session_store.update_session(
        session_id=session_id,
        title=request.title,
        model_id=request.model_id,
        pinned=request.pinned,
        params=request.params,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")
    return {"status": "updated"}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its messages."""
    success = await session_store.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")
    return {"status": "deleted"}


@router.get("/api/sessions/{session_id}/export/{format}")
async def export_session(session_id: str, format: str):
    """FR-405: Export a chat session."""
    if not await session_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")
    if format == "markdown" or format == "md":
        content = await session_store.export_session_markdown(session_id)
        return StreamingResponse(
            iter([content]),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=chat_{session_id}.md"},
        )
    elif format == "json":
        content = await session_store.export_session_json(session_id)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=chat_{session_id}.json"},
        )
    else:
        raise HTTPException(status_code=400, detail="Desteklenmeyen format. 'markdown' veya 'json' kullanın.")


# ── Chat Message Endpoint (for frontend) ──

@router.post("/api/chat")
async def send_chat_message(request: MessageRequest):
    """
    Send a message and get a streaming response.
    Saves messages to the session.
    """
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model yüklü değil"
        )
    if engine.is_generating:
        raise HTTPException(status_code=409, detail="Başka bir üretim zaten devam ediyor.")
    if not await session_store.get_session(request.session_id):
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")

    # Save user message
    await session_store.add_message(
        session_id=request.session_id,
        role="user",
        content=request.content,
    )

    # Get session messages for context
    messages = await session_store.get_messages(request.session_id)
    
    # Respect prompt pruning: Limit history messages if configured
    max_history = await session_store.get_setting("max_history_messages", 20)
    auto_prune = await session_store.get_setting("auto_context_prune", True)
    if auto_prune and max_history and max_history > 0:
        # Keep the latest N messages
        messages = messages[-max_history:]

    msg_list = [{"role": m["role"], "content": m["content"]} for m in messages]

    params = InferenceParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        max_tokens=request.max_tokens,
        system_prompt=request.system_prompt,
    )

    async def event_stream():
        full_response = ""
        tokens_generated = 0
        assistant_saved = False

        try:
            async for chunk in engine.generate_streaming(msg_list, params):
                if chunk["type"] == "token":
                    full_response += chunk["content"]
                    tokens_generated = chunk.get("tokens_generated", 0)
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif chunk["type"] == "done":
                    result = chunk.get("result") or {}
                    tokens_generated = int(result.get("tokens_generated") or tokens_generated)
                    if full_response:
                        await session_store.add_message(
                            session_id=request.session_id,
                            role="assistant",
                            content=full_response,
                            tokens_generated=tokens_generated,
                        )
                        assistant_saved = True
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif chunk["type"] == "error":
                    yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            if engine.is_generating:
                engine.stop_generation()
            # Preserve partial output when a browser/network disconnects after
            # generation has already produced useful text.
            if full_response and not assistant_saved:
                try:
                    await session_store.add_message(
                        session_id=request.session_id,
                        role="assistant",
                        content=full_response,
                        tokens_generated=tokens_generated,
                    )
                except Exception:
                    logger.exception("Partial assistant response could not be saved")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/stop")
async def stop_generation():
    """FR-303: Stop the current generation."""
    if not engine.is_generating:
        return {"status": "idle"}
    engine.stop_generation()
    return {"status": "stopping"}
