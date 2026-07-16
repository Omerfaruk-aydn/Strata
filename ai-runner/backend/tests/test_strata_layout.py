from backend.core.strata_ultra import discover_layout


def test_layout_discovers_complete_common_transformer_block():
    names = [f"blk.0.{suffix}.weight" for suffix in (
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"
    )]
    result = discover_layout(names)
    assert result["block_count"] == 1
    assert result["complete_blocks"] == 1
    assert result["blocks"][0]["tensors"]["q"] == "blk.0.attn_q.weight"
