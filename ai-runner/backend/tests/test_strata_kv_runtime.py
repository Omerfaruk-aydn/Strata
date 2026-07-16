from backend.core.strata_ultra import UltraKVCache


def test_kv_runtime_sliding_window_and_snapshot():
    cache = UltraKVCache(width=2, capacity_tokens=2, mode="sign1", group_size=4)
    cache.append([-1.0, 1.0, -2.0, 2.0])
    cache.append([3.0, -3.0, 4.0, -4.0])
    snapshot = cache.snapshot()
    assert snapshot.tokens == 2
    assert snapshot.evicted_tokens == 2
    assert [value >= 0 for value in cache.values()] == [True, False, True, False]


def test_ternary_runtime_preserves_zeroes():
    cache = UltraKVCache(width=2, capacity_tokens=4, mode="ternary05", group_size=2)
    cache.append([0.0, 2.0, -2.0, 0.0])
    values = cache.values()
    assert values[0] == 0.0
    assert values[1] > 0.0
    assert values[2] < 0.0
