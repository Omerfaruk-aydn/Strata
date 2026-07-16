from backend.core.strata_ultra import iq_native


def test_native_iq_adapter_reports_unavailable_without_library(monkeypatch):
    monkeypatch.setenv("STRATA_IQ_LIBRARY", "C:/does-not-exist/strata_iq.dll")
    iq_native._LIBRARY = None
    assert iq_native.native_iq_available() is False
