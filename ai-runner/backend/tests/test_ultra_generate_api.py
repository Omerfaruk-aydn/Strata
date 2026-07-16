import pytest

from backend.api import routes_ultra


@pytest.mark.asyncio
async def test_generate_request_validates_experimental_contract():
    request = routes_ultra.GenerateRequest(
        model_file="model.strata",
        block_prefixes=["block.0"],
        embedding_tensor="token_embd.weight",
        output_tensor="output.weight",
        width=4,
        prompt="Hi",
        max_new_tokens=2,
    )
    assert request.kv_mode == "sign1"
    assert request.backend == "auto"
