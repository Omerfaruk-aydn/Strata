import pytest

from backend.api import routes_ultra


@pytest.mark.asyncio
async def test_attention_step_api_returns_low_bit_cache_snapshot():
    request = routes_ultra.AttentionStepRequest(
        width=2,
        capacity_tokens=4,
        mode="sign1",
        query=[1.0, 0.0],
        key=[1.0, 0.0],
        value=[2.0, 0.0],
    )
    result = await routes_ultra.attention_step(request)
    assert len(result["output"]) == 2
    assert result["keys"]["tokens"] == 1
    assert result["keys"]["mode"] == "sign1"


@pytest.mark.asyncio
async def test_attention_step_api_accepts_sparse05_cache():
    request = routes_ultra.AttentionStepRequest(
        width=2,
        capacity_tokens=4,
        mode="sparse05",
        query=[1.0, 0.0],
        key=[1.0, 0.0],
        value=[2.0, 0.0],
    )
    result = await routes_ultra.attention_step(request)
    assert result["keys"]["mode"] == "sparse05"
    assert result["values"]["mode"] == "sparse05"
