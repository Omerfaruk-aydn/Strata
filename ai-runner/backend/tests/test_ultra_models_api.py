from pathlib import Path

import pytest

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


@pytest.mark.asyncio
async def test_ultra_models_lists_valid_and_invalid_containers(tmp_path: Path, monkeypatch):
    writer = StrataContainerWriter({"profile": "STRATA-Q0.5"})
    writer.add_tensor(TensorRecord("x", 1, 1, 1, "ternary-q05", b"\x02", b"\x00\x00\x80?"))
    writer.write(tmp_path / "valid.strata")
    (tmp_path / "broken.strata").write_bytes(b"broken")
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))
    result = await routes_ultra.ultra_models()
    assert len(result["models"]) == 2
    assert {model["valid"] for model in result["models"]} == {True, False}
