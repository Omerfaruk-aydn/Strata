import pytest

from backend.core.strata_ultra.iq_native import decode_iq_native


def test_native_iq_rejects_unknown_type_before_loading_library():
    with pytest.raises(ValueError, match="unsupported native IQ type"):
        decode_iq_native(999, b"", 256)


def test_native_iq_rejects_misaligned_value_count():
    with pytest.raises(ValueError, match="positive multiple"):
        decode_iq_native(16, b"", 255)


def test_native_iq_rejects_wrong_payload_size():
    with pytest.raises(ValueError, match="expected 66"):
        decode_iq_native(16, b"", 256)
