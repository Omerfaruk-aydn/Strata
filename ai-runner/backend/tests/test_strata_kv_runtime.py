from backend.core.strata_ultra import UltraKVCache
from backend.core.strata_ultra import cuda_backend


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


def test_sparse05_runtime_round_trip():
    cache = UltraKVCache(width=2, capacity_tokens=2, mode="sparse05", group_size=2)
    cache.append([0.0, 2.0, -2.0, 0.0])
    values = cache.values()
    assert values[0] == 0.0
    assert values[1] > 0.0
    assert values[2] < 0.0


def test_sparse_threshold_is_configurable():
    cache = UltraKVCache(width=2, capacity_tokens=1, mode="sparse05", group_size=2, sparse_threshold=1.0)
    cache.append([0.5, 2.0])
    values = cache.values()
    assert values[0] == 0.0
    assert values[1] != 0.0


def test_cuda_backend_is_selectable_for_supported_kv_modes(monkeypatch):
    calls = []
    monkeypatch.setattr(cuda_backend, "cuda_available", lambda: True)
    monkeypatch.setattr(cuda_backend, "decode_kv_cuda", lambda cache: calls.append(cache.mode) or [1.0, -1.0])
    cache = UltraKVCache(width=2, capacity_tokens=1, mode="sign1", group_size=2, backend="cuda")
    cache.append([1.0, -1.0])
    assert cache.values() == [1.0, -1.0]
    assert calls == ["sign1"]
