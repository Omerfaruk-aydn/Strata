-- AI Runner — Database Schema
-- Tables for chat sessions, messages, settings, and model cache.

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'Yeni Sohbet',
    model_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    pinned INTEGER NOT NULL DEFAULT 0,
    params_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    tokens_generated INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_cache (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    parameter_count INTEGER DEFAULT 0,
    available_quants TEXT DEFAULT '[]',
    license TEXT DEFAULT '',
    context_length INTEGER DEFAULT 4096,
    downloaded_quant TEXT,
    file_size_bytes INTEGER DEFAULT 0,
    local_path TEXT,
    last_used TEXT,
    downloads INTEGER DEFAULT 0,
    author TEXT DEFAULT '',
    description TEXT DEFAULT '',
    cached_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_pinned ON chat_sessions(pinned DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_category ON prompt_templates(category);
