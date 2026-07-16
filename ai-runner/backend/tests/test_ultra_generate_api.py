import pytest
import struct
from pathlib import Path

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


@pytest.mark.asyncio
async def test_generate_request_validates_experimental_contract():
    request = routes_ultra.GenerateRequest(
        model_file="model.strata",
        block_prefixes=[],
        embedding_tensor="token_embd.weight",
        output_tensor="output.weight",
        width=4,
        prompt="Hi",
        max_new_tokens=2,
    )
    assert request.kv_mode == "sign1"
    assert request.backend == "auto"


@pytest.mark.asyncio
async def test_generate_rejects_missing_embedding_before_execution(tmp_path: Path, monkeypatch):
    writer = StrataContainerWriter()
    writer.add_tensor(TensorRecord("output.weight", 1, 1, 1, "ternary-q05", b"\x02", struct.pack("<f", 1.0)))
    writer.write(tmp_path / "missing.strata")
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))
    request = routes_ultra.GenerateRequest(
        model_file="missing.strata", embedding_tensor="token_embd.weight", output_tensor="output.weight", width=1,
    )
    with pytest.raises(routes_ultra.HTTPException) as failure:
        await routes_ultra.generate_text(request)
    assert failure.value.status_code == 422
    assert "embedding tensor" in failure.value.detail
