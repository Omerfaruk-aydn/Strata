import struct

from backend.core.strata_ultra import (
    GenerationConfig,
    StrataContainerWriter,
    StrataGenerator,
    StrataRuntime,
    TensorRecord,
    LowBitTransformer,
    LowBitTransformerBlock,
    ByteTokenizer,
)


def _ones(name: str, rows: int, cols: int, group_size: int = 1) -> TensorRecord:
    count = rows * cols
    payload = bytes([0b10_10_10_10]) * ((count + 3) // 4)
    scales = struct.pack(f"<{(count + group_size - 1) // group_size}f", *([1.0] * ((count + group_size - 1) // group_size)))
    return TensorRecord(name, rows, cols, group_size, "ternary-q05", payload, scales)


def test_real_strata_container_runs_one_generation_step(tmp_path):
    target = tmp_path / "smoke.strata"
    writer = StrataContainerWriter()
    writer.add_tensor(_ones("token_embd.weight", 256, 1))
    writer.add_tensor(_ones("output.weight", 256, 1))
    for role in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        writer.add_tensor(_ones(f"blk.0.{role}.weight", 1, 1))
    writer.write(target)

    with StrataRuntime(target, memory_budget_bytes=64 * 1024, backend="python") as runtime:
        layout = {role: f"blk.0.{role}.weight" for role in (
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"
        )}
        block = LowBitTransformerBlock.from_layout(
            runtime, {"q": layout["attn_q"], "k": layout["attn_k"], "v": layout["attn_v"],
                      "o": layout["attn_output"], "gate": layout["ffn_gate"],
                      "up": layout["ffn_up"], "down": layout["ffn_down"]},
            width=1, context_capacity=4,
        )
        generator = StrataGenerator(
            runtime, LowBitTransformer([block]), "token_embd.weight", "output.weight", ByteTokenizer(),
        )
        result = generator.generate_with_metadata("A", GenerationConfig(max_new_tokens=1))

    assert result["generated_tokens"] == 1
    assert result["finish_reason"] == "length"
    assert result["text"].startswith("A")
