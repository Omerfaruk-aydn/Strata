import asyncio

import pytest

from backend.api.routes_ultra import GenerateRequest


def test_strata_generate_request_validates_timeout():
    assert GenerateRequest(
        model_file="model.strata",
        embedding_tensor="embedding",
        output_tensor="output",
        width=8,
        timeout_s=2.5,
    ).timeout_s == 2.5
    with pytest.raises(ValueError):
        GenerateRequest(
            model_file="model.strata",
            embedding_tensor="embedding",
            output_tensor="output",
            width=8,
            timeout_s=86_401,
        )
