import struct

import pytest

from backend.api import routes_ultra
from backend.core.strata_ultra import StrataContainerWriter, TensorRecord


def _ones(name: str, rows: int, cols: int) -> TensorRecord:
    count = rows * cols
    payload = bytes([0b10_10_10_10]) * ((count + 3) // 4)
    scales = struct.pack(f"<{count}f", *([1.0] * count))
    return TensorRecord(name, rows, cols, 1, "ternary-q05", payload, scales)


@pytest.mark.asyncio
async def test_generate_route_runs_real_container_and_returns_metadata(tmp_path, monkeypatch):
    target = tmp_path / "route-smoke.strata"
    writer = StrataContainerWriter()
    writer.add_tensor(_ones("embedding", 256, 1))
    writer.add_tensor(_ones("output", 256, 1))
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        writer.add_tensor(_ones(f"blk.0.{name}", 1, 1))
    writer.write(target)
    monkeypatch.setattr(routes_ultra.model_manager, "model_dir", str(tmp_path))

    response = await routes_ultra.generate_text(routes_ultra.GenerateRequest(
        model_file=target.name,
        block_prefixes=["blk.0"],
        embedding_tensor="embedding",
        output_tensor="output",
        width=1,
        prompt="A",
        max_new_tokens=1,
        backend="python",
    ))

    assert response["generated_tokens"] == 1
    assert response["finish_reason"] == "length"
    assert response["backend"] == "python"

    chat_response = await routes_ultra.strata_chat_completions(routes_ultra.StrataChatRequest(
        model_file=target.name,
        block_prefixes=["blk.0"],
        embedding_tensor="embedding",
        output_tensor="output",
        width=1,
        messages=[{"role": "user", "content": "Hello"}],
        max_new_tokens=1,
        backend="python",
    ))

    assert chat_response["object"] == "chat.completion"
    assert chat_response["choices"][0]["message"]["role"] == "assistant"
    assert chat_response["usage"]["completion_tokens"] == 1

    stream_response = await routes_ultra.strata_chat_completions(routes_ultra.StrataChatRequest(
        model_file=target.name,
        block_prefixes=["blk.0"],
        embedding_tensor="embedding",
        output_tensor="output",
        width=1,
        messages=[{"role": "user", "content": "Hello"}],
        max_new_tokens=1,
        backend="python",
        stream=True,
    ))
    chunks = []
    async for chunk in stream_response.body_iterator:
        chunks.append(chunk)

    assert '"role": "assistant"' in chunks[0]
    assert '"content":' in chunks[1]
    assert '"finish_reason": "length"' in chunks[2]
    assert chunks[-1] == "data: [DONE]\n\n"
