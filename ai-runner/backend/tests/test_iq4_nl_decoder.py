import struct

import pytest

from backend.core.strata_ultra.converter import _decode_iq4_nl


def test_iq4_nl_decodes_scale_and_nibbles():
    values = _decode_iq4_nl(struct.pack("<e", 0.5) + bytes(range(16)), 32)
    assert len(values) == 32
    assert values[0] == pytest.approx(-63.5)
    assert values[15] == pytest.approx(56.5)
    assert values[16] == pytest.approx(-63.5)


def test_iq4_nl_rejects_invalid_shape():
    with pytest.raises(ValueError, match="IQ4_NL"):
        _decode_iq4_nl(b"\0" * 18, 31)
