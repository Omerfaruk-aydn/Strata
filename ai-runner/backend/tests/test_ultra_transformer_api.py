import struct
from pathlib import Path

import pytest

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


@pytest.mark.asyncio
async def test_transformer_step_api_runs_one_block(tmp_path: Path, monkeypatch):
    writer = StrataContainerWriter()
    for part in ("q", "k", "v", "o", "gate", "up", "down"):
        writer.add_tensor(TensorRecord(f"b0.{part}", 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0)))
    writer.write(tmp_path / "model.strata")
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))
    request = routes_ultra.TransformerStepRequest(
        model_file="model.strata", block_prefixes=["b0"], width=2, hidden=[1.0, 0.5]
    )
    result = await routes_ultra.transformer_step(request)
    assert result["blocks"] == 1
    assert len(result["hidden"]) == 2
