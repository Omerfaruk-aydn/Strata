"""Production-safe command-line launcher for the AI Runner API."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Any

import uvicorn

from .db.session_store import DB_PATH


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _stored_runtime_settings() -> dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {}
    try:
        with sqlite3.connect(DB_PATH) as database:
            rows = database.execute(
                "SELECT key, value FROM settings WHERE key IN (?, ?, ?, ?)",
                ("api_host", "api_port", "api_key", "allow_network_access"),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}

    settings: dict[str, Any] = {}
    for key, value in rows:
        try:
            settings[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            settings[key] = value
    return settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Runner yerel API sunucusu")
    parser.add_argument("--host", help="Dinlenecek IP/host (varsayılan: kayıtlı ayar veya 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Dinlenecek TCP portu")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Loopback dışındaki bir adrese bağlanmaya açıkça izin ver",
    )
    parser.add_argument("--reload", action="store_true", help="Yalnızca geliştirme için auto-reload")
    return parser


def resolve_bind(args: argparse.Namespace) -> tuple[str, int]:
    stored = _stored_runtime_settings()
    host = (
        args.host
        or os.environ.get("AI_RUNNER_API_HOST")
        or stored.get("api_host")
        or "127.0.0.1"
    )
    port_source = args.port if args.port is not None else (
        os.environ.get("AI_RUNNER_API_PORT")
        or stored.get("api_port")
        or 8420
    )
    port = int(port_source)
    if not 1 <= port <= 65_535:
        raise ValueError("API portu 1 ile 65535 arasında olmalıdır.")

    network_allowed = bool(args.allow_network or stored.get("allow_network_access"))
    if str(host).lower() not in LOOPBACK_HOSTS:
        if not network_allowed:
            raise ValueError(
                "Loopback dışı API erişimi için ayarlardan izin verin veya --allow-network kullanın."
            )
        api_key = os.environ.get("AI_RUNNER_API_KEY") or stored.get("api_key")
        if not str(api_key or "").strip():
            raise ValueError("Loopback dışı API erişimi için bir API anahtarı gereklidir.")
    return str(host), port


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        host, port = resolve_bind(args)
    except ValueError as exc:
        parser.error(str(exc))

    from .main import app

    uvicorn.run(
        app if not args.reload else "backend.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
