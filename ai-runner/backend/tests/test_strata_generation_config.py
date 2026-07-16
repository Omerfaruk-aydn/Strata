import pytest

from backend.core.strata_ultra.generation import GenerationConfig


def test_generation_config_validates_budget_and_stop_ids():
    config = GenerationConfig(max_new_tokens=12, stop_token_ids=(2, 3))
    assert config.stop_token_ids == (2, 3)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError):
        GenerationConfig(stop_token_ids=(-1,))
