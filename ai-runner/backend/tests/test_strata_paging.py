from backend.core.strata_ultra import LayerPager


def test_layer_pager_enforces_byte_budget_and_lru():
    disposed = []

    def load(layer_id):
        return {"id": layer_id}, 10

    pager = LayerPager(2, 20, load, disposed.append)
    pager.get(0)
    pager.get(1)
    pager.get(0)
    pager.get(2)
    assert pager.resident_pages == 2
    assert pager.resident_bytes == 20
    assert disposed == [{"id": 1}]
    assert pager.events[-1].action == "load"
