"""
AI Runner — Unit Tests: Session Store (DB layer)
Async tests for CRUD operations on sessions and messages.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.db import session_store


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_db(tmp_path):
    """Create a fresh in-memory DB for each test."""
    db_path = str(tmp_path / "test.db")
    # Patch the DB path
    original = session_store.DB_PATH
    session_store.DB_PATH = db_path
    await session_store.init_db(db_path)
    yield db_path
    session_store.DB_PATH = original
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.mark.asyncio
class TestSessionCRUD:

    async def test_create_session(self, test_db):
        session = await session_store.create_session("Test Session")
        assert session["id"].startswith("sess_")
        assert session["title"] == "Test Session"
        assert not session["pinned"]

    async def test_create_session_with_model(self, test_db):
        session = await session_store.create_session(
            title="With Model",
            model_id="TheBloke/Llama-3-8B-GGUF",
        )
        assert session["model_id"] == "TheBloke/Llama-3-8B-GGUF"

    async def test_get_sessions_empty(self, test_db):
        sessions = await session_store.get_sessions()
        assert sessions == []

    async def test_get_sessions_returns_all(self, test_db):
        await session_store.create_session("Session 1")
        await session_store.create_session("Session 2")
        await session_store.create_session("Session 3")
        sessions = await session_store.get_sessions()
        assert len(sessions) == 3

    async def test_get_single_session(self, test_db):
        created = await session_store.create_session("Single Test")
        fetched = await session_store.get_session(created["id"])
        assert fetched is not None
        assert fetched["title"] == "Single Test"
        assert "messages" in fetched
        assert fetched["messages"] == []

    async def test_get_session_not_found(self, test_db):
        result = await session_store.get_session("nonexistent_id")
        assert result is None

    async def test_update_session_title(self, test_db):
        session = await session_store.create_session("Old Title")
        success = await session_store.update_session(session["id"], title="New Title")
        assert success

        updated = await session_store.get_session(session["id"])
        assert updated["title"] == "New Title"

    async def test_pin_session(self, test_db):
        session = await session_store.create_session("To Pin")
        await session_store.update_session(session["id"], pinned=True)

        updated = await session_store.get_session(session["id"])
        assert updated["pinned"]

    async def test_delete_session(self, test_db):
        session = await session_store.create_session("To Delete")
        success = await session_store.delete_session(session["id"])
        assert success

        result = await session_store.get_session(session["id"])
        assert result is None

    async def test_delete_nonexistent_session(self, test_db):
        success = await session_store.delete_session("does_not_exist")
        assert not success

    async def test_pinned_sessions_ordered_first(self, test_db):
        s1 = await session_store.create_session("Unpinned")
        s2 = await session_store.create_session("Pinned")
        await session_store.update_session(s2["id"], pinned=True)

        sessions = await session_store.get_sessions()
        assert sessions[0]["id"] == s2["id"]  # Pinned first


@pytest.mark.asyncio
class TestMessageCRUD:

    async def test_add_user_message(self, test_db):
        session = await session_store.create_session("Chat")
        msg = await session_store.add_message(session["id"], "user", "Hello!")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello!"

    async def test_add_assistant_message(self, test_db):
        session = await session_store.create_session("Chat")
        msg = await session_store.add_message(session["id"], "assistant", "Hi there!")
        assert msg["role"] == "assistant"

    async def test_messages_ordered_by_timestamp(self, test_db):
        session = await session_store.create_session("Chat")
        await session_store.add_message(session["id"], "user", "First")
        await session_store.add_message(session["id"], "assistant", "Second")
        await session_store.add_message(session["id"], "user", "Third")

        messages = await session_store.get_messages(session["id"])
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"

    async def test_session_includes_messages(self, test_db):
        session = await session_store.create_session("Chat With Msgs")
        await session_store.add_message(session["id"], "user", "Merhaba")
        await session_store.add_message(session["id"], "assistant", "Selam!")

        fetched = await session_store.get_session(session["id"])
        assert len(fetched["messages"]) == 2

    async def test_delete_session_cascades_messages(self, test_db):
        session = await session_store.create_session("Cascade Test")
        await session_store.add_message(session["id"], "user", "Test")
        await session_store.delete_session(session["id"])

        messages = await session_store.get_messages(session["id"])
        assert messages == []

    async def test_tokens_generated_stored(self, test_db):
        session = await session_store.create_session("Token Test")
        msg = await session_store.add_message(
            session["id"], "assistant", "Response", tokens_generated=42
        )
        assert msg["tokens_generated"] == 42


@pytest.mark.asyncio
class TestSettings:

    async def test_default_settings(self, test_db):
        await session_store.ensure_default_settings()
        settings = await session_store.get_settings()
        assert "theme" in settings
        assert "language" in settings

    async def test_update_setting(self, test_db):
        await session_store.update_settings({"theme": "light"})
        theme = await session_store.get_setting("theme")
        assert theme == "light"

    async def test_upsert_setting(self, test_db):
        await session_store.update_settings({"custom_key": "value1"})
        await session_store.update_settings({"custom_key": "value2"})
        result = await session_store.get_setting("custom_key")
        assert result == "value2"

    async def test_get_missing_setting_returns_default(self, test_db):
        result = await session_store.get_setting("nonexistent", default="fallback")
        assert result == "fallback"

    async def test_json_settings_roundtrip(self, test_db):
        """Complex values should survive JSON round-trip"""
        await session_store.update_settings({"complex": {"nested": True, "count": 42}})
        result = await session_store.get_setting("complex")
        assert result["nested"] is True
        assert result["count"] == 42


@pytest.mark.asyncio
class TestExport:

    async def test_export_markdown(self, test_db):
        session = await session_store.create_session("MD Export Test")
        await session_store.add_message(session["id"], "user", "Soru?")
        await session_store.add_message(session["id"], "assistant", "Cevap!")

        md = await session_store.export_session_markdown(session["id"])
        assert "# MD Export Test" in md
        assert "Soru?" in md
        assert "Cevap!" in md

    async def test_export_json(self, test_db):
        import json
        session = await session_store.create_session("JSON Export Test")
        await session_store.add_message(session["id"], "user", "Test")

        json_str = await session_store.export_session_json(session["id"])
        data = json.loads(json_str)
        assert data["title"] == "JSON Export Test"
        assert len(data["messages"]) == 1

    async def test_export_nonexistent_session(self, test_db):
        md = await session_store.export_session_markdown("ghost_id")
        assert md == ""
