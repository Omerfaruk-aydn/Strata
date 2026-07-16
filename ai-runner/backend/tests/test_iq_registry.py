from backend.core.strata_ultra.iq_registry import BY_NAME, NATIVE_BRIDGE_TYPE_IDS, capability_report, source_codec_names
from pathlib import Path
import re


def test_iq_registry_exposes_verified_and_pending_codecs():
    report = {item["name"]: item for item in capability_report()}
    assert report["IQ4_NL"]["decodable"] is True
    assert report["IQ4_NL"]["decoder"] == "_decode_iq4_nl"
    assert report["IQ2_XXS"]["decodable"] is False


def test_native_capability_report_marks_bridge_codecs_decodable():
    report = {item["name"]: item for item in capability_report(native_bridge=True)}
    assert report["IQ2_XXS"]["decodable"] is True
    assert report["IQ2_XXS"]["decoder"] == "native-ggml"
    assert report["IQ4_XS"]["decodable"] is True
    assert report["IQ4_XS"]["decoder"] == "native-ggml"
    assert 23 in NATIVE_BRIDGE_TYPE_IDS


def test_source_codec_names_only_reports_active_decoders():
    assert source_codec_names() == ["IQ4_NL"]
    assert "IQ4_XS" in source_codec_names(native_bridge=True)
    assert BY_NAME["IQ3_S"].type_id == 21


def test_native_cpp_dispatch_matches_registry_contract():
    bridge = Path(__file__).resolve().parents[2] / "native" / "iq" / "strata_iq.cpp"
    source = bridge.read_text(encoding="utf-8")
    native_cases = {int(value) for value in re.findall(r"case\s+(\d+)\s*:", source)}
    assert native_cases == set(NATIVE_BRIDGE_TYPE_IDS)
    for branch in re.split(r"\bcase\s+\d+\s*:", source)[1:]:
        assert "dequantize_row_iq" in branch.split("default:", 1)[0]
