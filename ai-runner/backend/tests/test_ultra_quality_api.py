import pytest

from backend.api import routes_ultra


@pytest.mark.asyncio
async def test_quality_api_returns_metrics():
    result = await routes_ultra.quality(routes_ultra.QualityRequest(
        reference=[1.0, 0.0, -1.0], reconstructed=[0.5, 0.0, -1.0]
    ))
    assert result["quality"]["rmse"] > 0
    assert result["quality"]["cosine_similarity"] > 0
