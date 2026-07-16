from backend.core.strata_ultra import TensorHeader, decode_ternary, encode_ternary


def test_ternary_codec_round_trip_shape_and_signs():
    values = [-2.0, 0.0, 3.0, 0.1, -4.0, 1.5, 0.0]
    packed, scales = encode_ternary(values, group_size=4)
    decoded = decode_ternary(packed, scales, len(values), group_size=4)
    assert len(decoded) == len(values)
    assert [v < 0 for v in decoded] == [v < 0 for v in values]
    assert decoded[1] == 0.0
    assert decoded[2] > 0.0


def test_tensor_header_is_versioned_and_self_describing():
    header = TensorHeader("layers.0.attn.q", 12, 64, 128)
    restored, offset = TensorHeader.from_bytes(header.to_bytes())
    assert restored == header
    assert offset == len(header.to_bytes())
