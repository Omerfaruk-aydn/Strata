"""Model library and download integrity tests."""

from __future__ import annotations

import hashlib
import json
import os
import struct

import pytest

from backend.models.model_manager import ModelManager


@pytest.fixture
def manager(tmp_path):
    instance = ModelManager(str(tmp_path / "models"))
    instance._cache_dir = str(tmp_path / "cache")
    os.makedirs(instance._cache_dir, exist_ok=True)
    return instance


def gguf_bytes(payload=b"test-model"):
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
        + payload
    )


def write_gguf(path, payload=b"test-model"):
    path.write_bytes(gguf_bytes(payload))


def test_parameter_count_parser_uses_boundaries_and_decimals(manager):
    assert manager._extract_param_count("org/CodeLlama-13B-GGUF", []) == 13_000_000_000
    assert manager._extract_param_count("org/Phi-3.8B-GGUF", []) == 3_800_000_000
    assert manager._extract_param_count("org/Qwen2.5-7B-GGUF", []) == 7_000_000_000
    assert manager._extract_param_count("org/model", ["70b", "gguf"]) == 70_000_000_000
    assert manager._extract_param_count("org/model-v3-beta", []) == 0


def test_quant_list_is_derived_from_real_filenames(manager):
    assert manager._extract_quants_from_files(
        [
            "model-Q4_K_M.gguf",
            "model-IQ2_XXS.gguf",
            "model-Q8_0-00001-of-00002.gguf",
            "README.md",
        ]
    ) == ["IQ2_XXS", "Q4_K_M", "Q8_0"]


def test_local_library_ignores_invalid_files_and_deletes_exact_id(manager, tmp_path):
    first = tmp_path / "models" / "model.gguf"
    second = tmp_path / "models" / "model-extra.gguf"
    invalid = tmp_path / "models" / "broken.gguf"
    write_gguf(first)
    write_gguf(second)
    invalid.write_bytes(b"not-a-gguf")

    manager._save_cache_file("model.gguf", {"model_id": "org/model", "quant": "Q4_K_M"})
    manager._save_cache_file(
        "model-extra.gguf",
        {"model_id": "org/model-extra", "quant": "Q4_K_M"},
    )

    local_ids = {model.id for model in manager.get_local_models()}
    assert local_ids == {"org/model", "org/model-extra"}

    assert manager.delete_model("org/model") is True
    assert not first.exists()
    assert second.exists()
    assert manager.delete_model("org/model") is False


@pytest.mark.asyncio
async def test_download_finalizes_valid_file_and_caches_checksum(manager, monkeypatch):
    progress = []

    monkeypatch.setattr(
        manager,
        "_resolve_download",
        lambda model_id, quant: ("remote-Q4_K_M.gguf", "https://example.invalid/model", None),
    )

    async def fake_download(**kwargs):
        with open(kwargs["part_path"], "wb") as model_file:
            model_file.write(gguf_bytes(b"downloaded"))
        return kwargs["part_path"]

    monkeypatch.setattr(manager, "_download_to_part", fake_download)
    local_path = await manager.download_model(
        "org/Model-13B-GGUF",
        progress_callback=progress.append,
    )

    assert os.path.exists(local_path)
    assert "org_Model-13B-GGUF" in os.path.basename(local_path)
    assert progress[-1].status == "completed"

    cache_path = os.path.join(manager._cache_dir, f"{os.path.basename(local_path)}.json")
    with open(cache_path, encoding="utf-8") as cache_file:
        cache = json.load(cache_file)
    assert cache["model_id"] == "org/Model-13B-GGUF"
    assert cache["remote_filename"] == "remote-Q4_K_M.gguf"
    assert cache["parameter_count"] == 13_000_000_000
    assert cache["sha256"] == hashlib.sha256(gguf_bytes(b"downloaded")).hexdigest()


@pytest.mark.asyncio
async def test_download_rejects_invalid_gguf(manager, monkeypatch):
    monkeypatch.setattr(
        manager,
        "_resolve_download",
        lambda model_id, quant: ("remote-Q4_K_M.gguf", "https://example.invalid/model", None),
    )

    async def fake_download(**kwargs):
        with open(kwargs["part_path"], "wb") as model_file:
            model_file.write(b"HTML error response")
        return kwargs["part_path"]

    monkeypatch.setattr(manager, "_download_to_part", fake_download)
    with pytest.raises(RuntimeError, match="GGUF"):
        await manager.download_model("org/invalid")


def test_cache_checksum_matches_file(manager, tmp_path):
    model_file = tmp_path / "models" / "checksum.gguf"
    write_gguf(model_file, b"checksum")
    assert manager._sha256(str(model_file)) == hashlib.sha256(gguf_bytes(b"checksum")).hexdigest()


def test_download_resolution_matches_quant_token_not_substring(manager, monkeypatch):
    import huggingface_hub

    class FakeApi:
        def list_repo_files(self, **kwargs):
            return [
                "model-IQ4_K_M.gguf",
                "model-Q4_K_M.gguf",
                "notes.txt",
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_url",
        lambda model_id, filename: f"https://models.example/{model_id}/{filename}",
    )
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "token")

    filename, url, token = manager._resolve_download("org/model", "Q4_K_M")
    assert filename == "model-Q4_K_M.gguf"
    assert url.endswith("/org/model/model-Q4_K_M.gguf")
    assert token == "token"


def test_search_uses_current_huggingface_contract(manager, monkeypatch):
    import huggingface_hub

    captured = {}

    class CardData:
        def to_dict(self):
            return {"license": "apache-2.0", "description": "A local test model"}

    class FakeModel:
        id = "org/Test-13B-GGUF"
        tags = ["gguf"]
        siblings = [
            type("Sibling", (), {"rfilename": "test-Q4_K_M.gguf"})(),
            type("Sibling", (), {"rfilename": "test-Q8_0.gguf"})(),
        ]
        card_data = CardData()
        downloads = 123

    class FakeApi:
        def list_models(self, **kwargs):
            captured.update(kwargs)
            return [FakeModel()]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    results = manager._search_models_sync("test", limit=3)
    assert captured == {
        "filter": "gguf",
        "search": "test",
        "sort": "downloads",
        "limit": 3,
        "full": True,
        "cardData": True,
    }
    assert results[0].parameter_count == 13_000_000_000
    assert results[0].available_quants == ["Q4_K_M", "Q8_0"]
    assert results[0].license == "apache-2.0"
