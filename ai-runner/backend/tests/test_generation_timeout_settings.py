import pytest

from backend.api.routes_models import LoadRequest
from backend.api.routes_settings import SettingsValues


def test_generation_timeout_is_available_in_load_and_settings_contracts():
    assert LoadRequest(generation_timeout_s=12.5).generation_timeout_s == 12.5
    assert SettingsValues(generation_timeout_s=0).generation_timeout_s == 0
    with pytest.raises(ValueError):
        LoadRequest(generation_timeout_s=86_401)
