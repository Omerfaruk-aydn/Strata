import pytest

from backend.api.routes_ultra import (
    BenchmarkRequest,
    MemoryRequest,
    PagingPlanRequest,
    benchmark,
    memory,
    paging_plan,
)


@pytest.mark.asyncio
async def test_ultra_memory_and_benchmark_endpoints():
    memory_result = await memory(MemoryRequest(value_count=1024, group_size=64, sparse_nonzero_ratio=0.25))
    benchmark_result = await benchmark(BenchmarkRequest(value_count=1024, group_size=64))
    assert memory_result["report"]["sign1_bytes"] < memory_result["report"]["f16_bytes"]
    assert memory_result["report"]["sparse05_nonzero_ratio_assumption"] == 0.25
    assert benchmark_result["benchmark"]["decoded_values"] == 1024


@pytest.mark.asyncio
async def test_ultra_paging_plan_detects_resident_window():
    result = await paging_plan(PagingPlanRequest(
        layer_sizes_bytes=[10, 10, 10],
        memory_budget_bytes=20,
        resident_window=2,
    ))
    assert result["feasible"] is True
    assert result["paging_required"] is True
    assert result["initial_resident_layers"] == [0, 1]
