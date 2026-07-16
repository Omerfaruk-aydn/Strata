import pytest

from backend.core.strata_ultra import LowBitTransformerBlock


def test_layout_adapter_rejects_incomplete_mapping():
    with pytest.raises(ValueError, match="missing"):
        LowBitTransformerBlock.from_layout(None, {"q": "q"}, width=2, context_capacity=4)
