"""Tests for the production-safe backend launcher."""

from __future__ import annotations

import argparse
import json
import sqlite3

import pytest

from backend import cli


def args(**overrides):
    values = {"host": None, "port": None, "allow_network": False, "reload": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def write_settings(path, **settings):
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        database.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in settings.items()],
        )


def test_resolve_bind_defaults_without_database(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.delenv("AI_RUNNER_API_HOST", raising=False)
    monkeypatch.delenv("AI_RUNNER_API_PORT", raising=False)
    monkeypatch.delenv("AI_RUNNER_API_KEY", raising=False)
    assert cli.resolve_bind(args()) == ("127.0.0.1", 8420)


def test_external_bind_requires_consent_and_api_key(tmp_path, monkeypatch):
    db_path = tmp_path / "settings.db"
    write_settings(db_path, api_host="0.0.0.0", api_port=9000)
    monkeypatch.setattr(cli, "DB_PATH", str(db_path))
    monkeypatch.delenv("AI_RUNNER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="izin"):
        cli.resolve_bind(args())
    with pytest.raises(ValueError, match="API anahtarı"):
        cli.resolve_bind(args(allow_network=True))

    monkeypatch.setenv("AI_RUNNER_API_KEY", "secret")
    assert cli.resolve_bind(args(allow_network=True)) == ("0.0.0.0", 9000)


def test_stored_external_consent_and_key_are_honored(tmp_path, monkeypatch):
    db_path = tmp_path / "settings.db"
    write_settings(
        db_path,
        api_host="192.168.1.20",
        api_port=9444,
        allow_network_access=True,
        api_key="stored-secret",
    )
    monkeypatch.setattr(cli, "DB_PATH", str(db_path))
    monkeypatch.delenv("AI_RUNNER_API_KEY", raising=False)
    assert cli.resolve_bind(args()) == ("192.168.1.20", 9444)


def test_environment_and_arguments_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("AI_RUNNER_API_HOST", "localhost")
    monkeypatch.setenv("AI_RUNNER_API_PORT", "9001")
    assert cli.resolve_bind(args()) == ("localhost", 9001)
    assert cli.resolve_bind(args(host="127.0.0.1", port=9002)) == ("127.0.0.1", 9002)


def test_invalid_port_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DB_PATH", str(tmp_path / "missing.db"))
    with pytest.raises(ValueError, match="portu"):
        cli.resolve_bind(args(port=70_000))
    with pytest.raises(ValueError, match="portu"):
        cli.resolve_bind(args(port=0))


def test_run_forwards_resolved_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DB_PATH", str(tmp_path / "missing.db"))
    calls = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *positional, **keyword: calls.append((positional, keyword)))

    cli.run(["--host", "127.0.0.1", "--port", "8765"])
    assert len(calls) == 1
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 8765
    assert calls[0][1]["reload"] is False
