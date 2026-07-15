"""Shared authentication and browser-origin protection for local API routes."""

from __future__ import annotations

import secrets
import os
import base64
import binascii
from typing import Optional

from fastapi import HTTPException, Request, WebSocket, status

from ..db import session_store


TRUSTED_BROWSER_ORIGINS = {
    "http://localhost:1420",
    "http://localhost:5173",
    "http://127.0.0.1:1420",
    "http://127.0.0.1:5173",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
}


def _normalise_key(value: object) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip()
    return key or None


def _extract_http_key(request: Request) -> Optional[str]:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return _normalise_key(request.headers.get("x-api-key"))


def _origin_is_trusted(origin: Optional[str]) -> bool:
    # Requests made by native/CLI clients normally have no Origin header.
    return not origin or origin in TRUSTED_BROWSER_ORIGINS


async def require_api_access(request: Request) -> None:
    """Protect API routes with origin checks and the optional configured key."""
    if not _origin_is_trusted(request.headers.get("origin")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu tarayıcı kaynağından yerel API erişimine izin verilmiyor.",
        )

    expected = _normalise_key(os.environ.get("AI_RUNNER_API_KEY"))
    if expected is None:
        expected = _normalise_key(await session_store.get_setting("api_key"))
    if expected is None:
        return

    provided = _extract_http_key(request)
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçerli bir AI Runner API anahtarı gerekli.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def websocket_access_allowed(websocket: WebSocket) -> tuple[bool, int]:
    """Return authorization state and a private WebSocket close code."""
    if not _origin_is_trusted(websocket.headers.get("origin")):
        return False, 4403

    expected = _normalise_key(os.environ.get("AI_RUNNER_API_KEY"))
    if expected is None:
        expected = _normalise_key(await session_store.get_setting("api_key"))
    if expected is None:
        return True, 1000

    provided = None
    requested_protocols = websocket.headers.get("sec-websocket-protocol", "")
    for protocol in (part.strip() for part in requested_protocols.split(",")):
        if not protocol.startswith("ai-runner-key."):
            continue
        encoded = protocol.removeprefix("ai-runner-key.")
        try:
            padding = "=" * (-len(encoded) % 4)
            provided = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            provided = None
        break
    if provided is None or not secrets.compare_digest(provided, expected):
        return False, 4401
    return True, 1000
