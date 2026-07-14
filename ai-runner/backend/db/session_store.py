"""
AI Runner — Session Store
Async SQLite operations for chat sessions, messages, and settings.
Implements FR-404, FR-405, FR-701–FR-703.
"""

import os
import json
import uuid
import aiosqlite
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.expanduser("~"), ".ai-runner")
DB_PATH = os.path.join(DB_DIR, "ai_runner.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


async def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the database with schema."""
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    async with aiosqlite.connect(path) as db:
        with open(SCHEMA_PATH) as f:
            await db.executescript(f.read())
        await db.commit()
    logger.info(f"Database initialized at {path}")


async def get_db(db_path: Optional[str] = None) -> aiosqlite.Connection:
    """Get a database connection."""
    path = db_path or DB_PATH
    db = await aiosqlite.connect(path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


# ── Session Operations (FR-701, FR-703) ──

async def create_session(
    title: str = "Yeni Sohbet",
    model_id: Optional[str] = None,
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a new chat session."""
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            """INSERT INTO chat_sessions (id, title, model_id, created_at, updated_at, params_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, title, model_id, now, now, json.dumps(params or {}))
        )
        await db.commit()

    return {
        "id": session_id,
        "title": title,
        "model_id": model_id,
        "created_at": now,
        "updated_at": now,
        "pinned": False,
        "params": params or {},
        "messages": [],
    }


async def get_sessions() -> List[Dict[str, Any]]:
    """Get all chat sessions, ordered by pinned then updated."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM chat_sessions
               ORDER BY pinned DESC, updated_at DESC"""
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "model_id": row["model_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "pinned": bool(row["pinned"]),
            "params": json.loads(row["params_json"] or "{}"),
        }
        for row in rows
    ]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get a single session with its messages."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        msg_cursor = await db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        messages = await msg_cursor.fetchall()

    return {
        "id": row["id"],
        "title": row["title"],
        "model_id": row["model_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "pinned": bool(row["pinned"]),
        "params": json.loads(row["params_json"] or "{}"),
        "messages": [
            {
                "id": msg["id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "tokens_generated": msg["tokens_generated"],
            }
            for msg in messages
        ],
    }


async def update_session(
    session_id: str,
    title: Optional[str] = None,
    model_id: Optional[str] = None,
    pinned: Optional[bool] = None,
    params: Optional[Dict] = None,
) -> bool:
    """Update a chat session (FR-703: rename, pin)."""
    updates = []
    values = []

    if title is not None:
        updates.append("title = ?")
        values.append(title)
    if model_id is not None:
        updates.append("model_id = ?")
        values.append(model_id)
    if pinned is not None:
        updates.append("pinned = ?")
        values.append(1 if pinned else 0)
    if params is not None:
        updates.append("params_json = ?")
        values.append(json.dumps(params))

    if not updates:
        return False

    updates.append("updated_at = ?")
    values.append(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    values.append(session_id)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE chat_sessions SET {', '.join(updates)} WHERE id = ?",
            values
        )
        await db.commit()
    return True


async def delete_session(session_id: str) -> bool:
    """Delete a chat session and its messages (FR-703)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor = await db.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        await db.commit()
        return cursor.rowcount > 0


# ── Message Operations (FR-404) ──

async def add_message(
    session_id: str,
    role: str,
    content: str,
    tokens_generated: int = 0,
) -> Dict[str, Any]:
    """Add a message to a session."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO messages (session_id, role, content, timestamp, tokens_generated)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, role, content, now, tokens_generated)
        )
        msg_id = cursor.lastrowid

        # Update session's updated_at
        await db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        await db.commit()

    return {
        "id": msg_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": now,
        "tokens_generated": tokens_generated,
    }


async def get_messages(session_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a session."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "tokens_generated": row["tokens_generated"],
        }
        for row in rows
    ]


# ── Chat Export (FR-405) ──

async def export_session_markdown(session_id: str) -> str:
    """Export a chat session as Markdown."""
    session = await get_session(session_id)
    if not session:
        return ""

    lines = [
        f"# {session['title']}",
        f"**Model:** {session.get('model_id', 'N/A')}",
        f"**Tarih:** {session['created_at']}",
        "",
        "---",
        "",
    ]

    for msg in session.get("messages", []):
        role_label = "🧑 Kullanıcı" if msg["role"] == "user" else "🤖 Asistan"
        lines.append(f"### {role_label}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")

    return "\n".join(lines)


async def export_session_json(session_id: str) -> str:
    """Export a chat session as JSON."""
    session = await get_session(session_id)
    if not session:
        return "{}"
    return json.dumps(session, indent=2, ensure_ascii=False)


# ── Settings Operations (FR-601–FR-604) ──

async def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()

    settings = {}
    for row in rows:
        try:
            settings[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            settings[row["key"]] = row["value"]

    return settings


async def update_settings(settings: Dict[str, Any]) -> None:
    """Update settings (upsert)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with aiosqlite.connect(DB_PATH) as db:
        for key, value in settings.items():
            value_str = json.dumps(value) if not isinstance(value, str) else value
            await db.execute(
                """INSERT INTO settings (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
                (key, value_str, now, value_str, now)
            )
        await db.commit()


async def get_setting(key: str, default: Any = None) -> Any:
    """Get a single setting value."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()

    if not row:
        return default

    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


# Default settings
DEFAULT_SETTINGS = {
    "theme": "dark",
    "language": "tr",
    "default_model": None,
    "default_system_prompt": "",
    "model_dir": os.path.join(os.path.expanduser("~"), ".ai-runner", "models"),
    "cache_size_limit_gb": 50,
    "n_threads": None,
    "use_mmap": True,
    "n_batch": 512,
    "api_host": "127.0.0.1",
    "api_port": 8420,
    "api_key": None,
    "advanced_mode": False,
}


async def ensure_default_settings() -> None:
    """Initialize default settings if not present."""
    current = await get_settings()
    missing = {k: v for k, v in DEFAULT_SETTINGS.items() if k not in current}
    if missing:
        await update_settings(missing)
