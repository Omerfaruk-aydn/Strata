from backend.core.strata_ultra import decode_kv, encode_kv, kv_memory_report


def test_sign1_cache_round_trip_preserves_signs():
    values = [-2.0, 0.0, 3.0, -0.1, 4.0, -1.5]
    cache = encode_kv(values, mode="sign1", group_size=4)
    decoded = decode_kv(cache)
    assert len(decoded) == len(values)
    assert [v >= 0 for v in decoded] == [v >= 0 for v in values]


def test_ternary05_cache_round_trip_preserves_zero_and_sign():
    values = [-2.0, 0.0, 3.0, 0.1, -4.0]
    decoded = decode_kv(encode_kv(values, mode="ternary05", group_size=4))
    assert decoded[1] == 0.0
    assert decoded[0] < 0.0
    assert decoded[2] > 0.0


def test_ultra_cache_is_smaller_than_f16():
    report = kv_memory_report(4096)
    assert report["sign1_bytes"] < report["f16_bytes"]
    assert report["ternary05_bytes"] < report["f16_bytes"]
