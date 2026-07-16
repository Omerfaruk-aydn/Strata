import pytest

from backend.api import routes_ultra


@pytest.mark.asyncio
async def test_strata_stop_is_idle_without_active_generation():
    routes_ultra._strata_generation_cancel = None
    assert await routes_ultra.stop_generate_text() == {"status": "idle"}


@pytest.mark.asyncio
async def test_strata_stop_signals_active_generation():
    import threading
    event = threading.Event()
    routes_ultra._strata_generation_cancel = event
    try:
        assert await routes_ultra.stop_generate_text() == {"status": "stopping"}
        assert event.is_set()
    finally:
        routes_ultra._strata_generation_cancel = None
