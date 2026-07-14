"""
AI Runner — Chat API Routes
OpenAI-compatible chat completions + session management.
Implements FR-501, FR-502, FR-701–FR-703.
"""

import json
import time
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from ..core.inference_engine import engine, InferenceParams
from ..db import session_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


# ── OpenAI-Compatible Models ──

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False
    stop: Optional[List[str]] = None


class SessionCreateRequest(BaseModel):
    title: str = "Yeni Sohbet"
    model_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    pinned: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None


class MessageRequest(BaseModel):
    content: str
    session_id: str
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048


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
    params = InferenceParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop or [],
    )

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(messages, params),
            media_type="text/event-stream",
        )
    else:
        return await _non_streaming_completion(messages, params, request.model)


async def _stream_chat_completion(
    messages: List[Dict[str, str]],
    params: InferenceParams,
):
    """Stream chat completion in OpenAI SSE format."""
    completion_id = f"chatcmpl-{int(time.time())}"
    model_id = engine.model_info.model_id if engine.model_info else "unknown"

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
            yield "data: [DONE]\n\n"

        elif chunk["type"] == "error":
            yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
            yield "data: [DONE]\n\n"


async def _non_streaming_completion(
    messages: List[Dict[str, str]],
    params: InferenceParams,
    model: str,
):
    """Non-streaming chat completion in OpenAI format."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.generate_sync(messages, params)
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
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
                "prompt_tokens": 0,
                "completion_tokens": result.tokens_generated,
                "total_tokens": result.tokens_generated,
            },
        }
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

    # Save user message
    await session_store.add_message(
        session_id=request.session_id,
        role="user",
        content=request.content,
    )

    # Get session messages for context
    messages = await session_store.get_messages(request.session_id)
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

        async for chunk in engine.generate_streaming(msg_list, params):
            if chunk["type"] == "token":
                full_response += chunk["content"]
                tokens_generated = chunk.get("tokens_generated", 0)
                yield f"data: {json.dumps(chunk)}\n\n"

            elif chunk["type"] == "done":
                # Save assistant message
                await session_store.add_message(
                    session_id=request.session_id,
                    role="assistant",
                    content=full_response,
                    tokens_generated=tokens_generated,
                )
                yield f"data: {json.dumps(chunk)}\n\n"

            elif chunk["type"] == "error":
                yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.post("/api/chat/stop")
async def stop_generation():
    """FR-303: Stop the current generation."""
    engine.stop_generation()
    return {"status": "stopping"}
