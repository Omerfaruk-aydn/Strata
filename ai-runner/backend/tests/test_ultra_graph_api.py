import struct
from pathlib import Path

import pytest

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


@pytest.mark.asyncio
async def test_graph_api_runs_nodes(tmp_path: Path, monkeypatch):
    writer = StrataContainerWriter()
    writer.add_tensor(TensorRecord("a", 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0)))
    writer.write(tmp_path / "graph.strata")
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))
    request = routes_ultra.GraphRunRequest(
        model_file="graph.strata",
        nodes=[routes_ultra.GraphNodeRequest(tensor_name="a", activation="relu")],
        vector=[2.0, -3.0],
        prefetch=False,
    )
    result = await routes_ultra.run_graph(request)
    assert result["values"] == [2.0, 0.0]
    assert result["nodes"] == 1
