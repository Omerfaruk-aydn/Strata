from backend.core.strata_ultra import decode_sparse05, encode_sparse05


def test_sparse05_round_trip_preserves_sparse_signs():
    values = [0.0] * 128
    values[3] = 4.0
    values[100] = -2.0
    payload, scales = encode_sparse05(values, group_size=128)
    decoded = decode_sparse05(payload, scales, len(values), group_size=128)
    assert decoded[3] > 0
    assert decoded[100] < 0
    assert sum(value != 0 for value in decoded) == 2


def test_sparse05_is_compact_for_sparse_group():
    values = [0.0] * 128
    values[7] = 1.0
    payload, scales = encode_sparse05(values)
    assert len(payload) + len(scales) * 4 < 128 // 16
