import struct
from pathlib import Path

import pytest

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


@pytest.mark.asyncio
async def test_inspect_api_reports_container_preflight(tmp_path: Path, monkeypatch):
    writer = StrataContainerWriter({"profile": "STRATA-Q0.5", "conversion_quality": {"mse": 0.1}})
    writer.add_tensor(TensorRecord("blk.0.attn_q.weight", 1, 2, 2, "ternary-q05", b"\x06", struct.pack("<f", 1.0)))
    writer.write(tmp_path / "inspect.strata")
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))

    result = await routes_ultra.ultra_inspect("inspect.strata")
    assert result["tensor_count"] == 1
    assert result["codec_counts"] == {"ternary-q05": 1}
    assert result["metadata"]["conversion_quality"]["mse"] == 0.1
    assert result["layout"]["block_count"] == 1
