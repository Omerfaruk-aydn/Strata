import struct
from pathlib import Path

import pytest

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


@pytest.mark.asyncio
async def test_runtime_benchmark_reports_iterations(tmp_path: Path, monkeypatch):
    writer = StrataContainerWriter()
    writer.add_tensor(TensorRecord("x", 1, 2, 2, "ternary-q05", bytes([0b10_01]), struct.pack("<f", 1.0)))
    writer.write(tmp_path / "bench.strata")
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))
    request = routes_ultra.RuntimeBenchmarkRequest(
        model_file="bench.strata", tensor_name="x", vector=[2.0, 3.0], iterations=3
    )
    result = await routes_ultra.runtime_benchmark(request)
    assert result["benchmark"]["iterations"] == 3
    assert result["benchmark"]["output_length"] == 1
    assert result["benchmark"]["average_time_ms"] >= 0
    assert result["benchmark"]["tensor_codec"] == "ternary-q05"
    assert result["benchmark"]["execution_path"] == "numpy-or-fallback"
