from backend.core.strata_ultra import LowBitAttention


def test_attention_uses_rolling_low_bit_kv_history():
    attention = LowBitAttention(width=2, capacity_tokens=2, mode="sign1")
    first = attention.step([1.0, 0.0], [1.0, 0.0], [2.0, 0.0])
    second = attention.step([1.0, 0.0], [1.0, 0.0], [4.0, 0.0])
    assert first[0] > 0.0
    assert second[0] > first[0]
    assert attention.keys.snapshot().tokens == 2


def test_attention_supports_ternary_cache():
    attention = LowBitAttention(width=2, capacity_tokens=3, mode="ternary05")
    result = attention.step([1.0, -1.0], [1.0, -1.0], [1.0, -1.0])
    assert len(result) == 2
