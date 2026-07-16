from backend.core.strata_ultra import run_codec_benchmark


def test_codec_benchmark_reports_real_measurements():
    report = run_codec_benchmark(1024, 64)
    assert report["decoded_values"] == 1024
    assert report["encode_ms"] >= 0
    assert report["decode_ms"] >= 0
    assert report["packed_bytes"] < 1024 * 2
    assert report["sparse05"]["packed_bytes"] > 0
    assert report["sparse05"]["quality"]["mse"] >= 0
