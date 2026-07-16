import pytest

from backend.core.strata_ultra.iq_native import decode_iq_native, native_iq_available
from backend.core.strata_ultra.iq_registry import NATIVE_BRIDGE_TYPE_IDS, get_iq_codec


@pytest.mark.skipif(not native_iq_available(), reason="native GGML IQ bridge is not installed")
@pytest.mark.parametrize("type_id", sorted(NATIVE_BRIDGE_TYPE_IDS))
def test_native_bridge_decodes_zero_block_for_every_supported_iq_type(type_id):
    codec = get_iq_codec(type_id)
    values = decode_iq_native(type_id, bytes(codec.block_bytes), codec.block_values)
    assert len(values) == codec.block_values
    assert all(value == value for value in values)


def test_native_iq_rejects_unknown_type_before_loading_library():
    with pytest.raises(ValueError, match="unsupported native IQ type"):
        decode_iq_native(999, b"", 256)


def test_native_iq_rejects_misaligned_value_count():
    with pytest.raises(ValueError, match="positive multiple"):
        decode_iq_native(16, b"", 255)


def test_native_iq_rejects_wrong_payload_size():
    with pytest.raises(ValueError, match="expected 66"):
        decode_iq_native(16, b"", 256)


def test_iq4_xs_uses_native_block_geometry():
    with pytest.raises(ValueError, match="expected 136"):
        decode_iq_native(23, b"", 256)
