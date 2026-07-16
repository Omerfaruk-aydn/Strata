"""Extreme Model Mode planning and adaptive-loading tests."""

from __future__ import annotations

import pytest

from backend.core.extreme_model import (
    analyze_feasibility,
    estimated_specification,
    estimate_kv_cache_for_spec,
    hardware_fingerprint,
)
from backend.core.hardware_profile import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareProfile,
    RAMInfo,
    VirtualMemoryInfo,
)
from backend.core.inference_engine import EngineConfig, InferenceEngine, ModelInfo
from backend.core.runtime_capabilities import RuntimeCapabilities


def extreme_hardware(
    *,
    vram_total: int = 20 * 1024,
    vram_free: int = 19 * 1024,
    ram_total: int = 64 * 1024,
    ram_free: int = 58 * 1024,
    disk_type: str = "SSD",
) -> HardwareProfile:
    gpu = GPUInfo(
        name="NVIDIA RTX Test",
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        index=0,
    )
    return HardwareProfile(
        gpu=gpu,
        gpus=[gpu],
        ram=RAMInfo(total_mb=ram_total, free_mb=ram_free),
        virtual_memory=VirtualMemoryInfo(total_mb=64 * 1024, free_mb=60 * 1024),
        disk=DiskInfo(type=disk_type, free_gb=500, total_gb=1000, path="C:/models"),
        cpu=CPUInfo(name="Test CPU", cores=16, threads=32),
        os_info="Windows Test",
        selected_gpu_index=0,
    )


def cuda_runtime(installed: bool = True) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        llama_cpp_installed=installed,
        active_backend="cuda" if installed else "unknown",
        gpu_offload_supported=installed,
    )


def test_100b_q3_plan_uses_20gb_vram_and_system_ram():
    spec = estimated_specification("test/100B", "Q3_K_M", 100_000_000_000)
    report = analyze_feasibility(
        spec,
        extreme_hardware(),
        cuda_runtime(),
        preset_name="maximum_capacity",
        requested_context_length=2048,
    )

    assert report.status in {"ready", "ideal"}
    assert 0 < report.runtime.n_gpu_layers < spec.total_layers
    assert report.runtime.cpu_layers + report.runtime.n_gpu_layers == spec.total_layers
    assert report.memory.gpu_weights_mb > 0
    assert report.memory.cpu_weights_mb > 0
    assert report.memory.storage_mode == "ram_resident"
    assert report.runtime.n_batch == 64
    assert report.runtime.use_mmap is True


def test_multi_gpu_plan_uses_combined_safe_vram_capacity():
    spec = estimated_specification("test/100B", "Q3_K_M", 100_000_000_000)
    single_hardware = extreme_hardware(vram_total=12 * 1024, vram_free=11 * 1024)
    multi_hardware = extreme_hardware(vram_total=12 * 1024, vram_free=11 * 1024)
    multi_hardware.gpus.append(GPUInfo(
        name="NVIDIA RTX Test 2",
        vram_total_mb=12 * 1024,
        vram_free_mb=10 * 1024,
        index=1,
    ))

    single = analyze_feasibility(spec, single_hardware, cuda_runtime())
    multi = analyze_feasibility(spec, multi_hardware, cuda_runtime())

    assert multi.runtime.n_gpu_layers > single.runtime.n_gpu_layers
    assert multi.runtime.tensor_split is not None
    assert len(multi.runtime.tensor_split) == 2
    assert sum(multi.runtime.tensor_split) == pytest.approx(1.0, abs=1e-5)


def test_invalid_tensor_split_is_blocked():
    hardware = extreme_hardware()
    hardware.gpus.append(GPUInfo(
        name="NVIDIA RTX Test 2",
        vram_total_mb=8 * 1024,
        vram_free_mb=7 * 1024,
        index=1,
    ))
    report = analyze_feasibility(
        estimated_specification("test/7B", "Q4_K_M", 7_000_000_000),
        hardware,
        cuda_runtime(),
        tensor_split=[1.0],
    )

    assert report.status == "blocked"
    assert any("Tensor split" in blocker for blocker in report.blockers)


def test_mapped_memory_pressure_is_reported_truthfully():
    spec = estimated_specification("test/100B", "Q8_0", 100_000_000_000)
    hardware = extreme_hardware(ram_total=32 * 1024, ram_free=24 * 1024)
    report = analyze_feasibility(
        spec,
        hardware,
        cuda_runtime(),
        preset_name="maximum_capacity",
        requested_context_length=4096,
    )

    assert report.status == "constrained"
    assert report.memory.storage_mode == "memory_mapped"
    assert report.memory.physical_ram_shortfall_mb > 0
    assert report.runtime.use_mlock is False
    assert any("mmap" in warning.lower() for warning in report.warnings)


def test_hdd_with_memory_shortfall_is_blocked():
    spec = estimated_specification("test/100B", "Q8_0", 100_000_000_000)
    report = analyze_feasibility(
        spec,
        extreme_hardware(ram_total=16 * 1024, ram_free=10 * 1024, disk_type="HDD"),
        cuda_runtime(),
        preset_name="maximum_capacity",
        requested_context_length=2048,
    )

    assert report.status == "blocked"
    assert report.memory.storage_mode == "insufficient"
    assert any("HDD" in blocker for blocker in report.blockers)


def test_pre_download_plan_blocks_when_model_volume_is_too_small():
    spec = estimated_specification("test/100B", "Q3_K_M", 100_000_000_000)
    hardware = extreme_hardware()
    hardware.disk.free_gb = 8
    report = analyze_feasibility(
        spec,
        hardware,
        cuda_runtime(),
        preset_name="maximum_capacity",
        requested_context_length=2048,
    )

    assert report.status == "blocked"
    assert any("free space" in blocker.lower() for blocker in report.blockers)
    assert any("model directory" in action.lower() for action in report.actions)


def test_missing_runtime_blocks_load_plan():
    spec = estimated_specification("test/7B", "Q4_K_M", 7_000_000_000)
    report = analyze_feasibility(spec, extreme_hardware(), cuda_runtime(False))
    assert report.status == "blocked"
    assert any("llama-cpp-python" in blocker for blocker in report.blockers)


def test_explicit_cpu_mode_does_not_report_gpu_runtime_as_missing():
    report = analyze_feasibility(
        estimated_specification("test/7B", "Q4_K_M", 7_000_000_000),
        extreme_hardware(),
        cuda_runtime(),
        force_cpu=True,
    )

    assert report.runtime.backend == "cpu"
    assert report.runtime.n_gpu_layers == 0
    assert not any("GPU-enabled" in action for action in report.actions)


def test_unknown_quant_requires_an_explicit_file_size():
    with pytest.raises(ValueError, match="Unsupported quantization"):
        estimated_specification("test/model", "mystery", 7_000_000_000)

    spec = estimated_specification(
        "test/model",
        "mystery",
        7_000_000_000,
        file_size_mb=4096,
    )
    assert spec.file_size_mb == 4096


def test_context_is_capped_by_capacity_preset():
    spec = estimated_specification(
        "test/100B", "Q3_K_M", 100_000_000_000, context_length=32768,
    )
    report = analyze_feasibility(
        spec,
        extreme_hardware(),
        cuda_runtime(),
        preset_name="maximum_capacity",
        requested_context_length=32768,
    )
    assert report.runtime.context_length == 2048
    assert any("Context was reduced" in warning for warning in report.warnings)


def test_kv_cache_uses_gqa_metadata_and_quant_type():
    spec = estimated_specification("test/model", "Q4_K_M", 7_000_000_000)
    spec.embedding_length = 4096
    spec.head_count = 32
    spec.head_count_kv = 8
    q4 = estimate_kv_cache_for_spec(spec, 4096, "q4_0")
    f16 = estimate_kv_cache_for_spec(spec, 4096, "f16")
    assert q4 > 0
    assert f16 > q4 * 3


def test_hardware_fingerprint_is_stable_and_backend_specific():
    hardware = extreme_hardware()
    assert hardware_fingerprint(hardware, "cuda") == hardware_fingerprint(hardware, "cuda")
    assert hardware_fingerprint(hardware, "cuda") != hardware_fingerprint(hardware, "cpu")


def test_adaptive_loader_recovers_from_gpu_oom(monkeypatch):
    adaptive_engine = InferenceEngine()
    seen = []

    def fake_load(model_id, model_path, config):
        seen.append(config.model_copy(deep=True))
        if len(seen) < 3:
            raise RuntimeError("CUDA out of memory")
        adaptive_engine._config = config
        return ModelInfo(
            model_id=model_id,
            model_path=model_path,
            n_gpu_layers=config.n_gpu_layers,
            context_length=config.context_length,
            total_layers=96,
            is_loaded=True,
        )

    monkeypatch.setattr(adaptive_engine, "load_model", fake_load)
    info, report = adaptive_engine.load_model_adaptive(
        "test/100B",
        "C:/models/test.gguf",
        EngineConfig(n_gpu_layers=40, use_mlock=True, n_batch=256),
        max_attempts=5,
    )

    assert info.is_loaded is True
    assert report.succeeded is True
    assert report.recovered_from_oom is True
    assert len(report.attempts) == 3
    assert seen[1].use_mlock is False
    assert seen[2].n_gpu_layers < 40


def test_adaptive_loader_does_not_retry_corrupt_models(monkeypatch):
    adaptive_engine = InferenceEngine()
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("invalid GGUF magic")

    monkeypatch.setattr(adaptive_engine, "load_model", fake_load)
    with pytest.raises(RuntimeError, match="invalid GGUF"):
        adaptive_engine.load_model_adaptive(
            "test/broken", "broken.gguf", EngineConfig(n_gpu_layers=10), max_attempts=6,
        )
    assert len(calls) == 1
    assert adaptive_engine.last_load_report.succeeded is False


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory", "gpu_oom"),
        ("The paging file is too small", "host_oom"),
        ("failed to allocate KV cache", "context_oom"),
        ("invalid GGUF magic", "model_error"),
        ("unexpected loader failure", "unknown"),
    ],
)
def test_load_error_classification(message, expected):
    assert InferenceEngine.classify_load_error(message) == expected


def test_context_compaction_retains_summary_and_latest_message():
    adaptive_engine = InferenceEngine()
    adaptive_engine._config = EngineConfig(
        context_compaction_mode="extractive_summary",
        auto_context_prune=True,
    )
    adaptive_engine.count_prompt_tokens = lambda messages: sum(len(item["content"]) for item in messages) // 4
    messages = [{"role": "system", "content": "System"}]
    for index in range(20):
        messages.extend([
            {"role": "user", "content": f"Important question {index} " + "x" * 120},
            {"role": "assistant", "content": f"Useful answer {index} " + "y" * 120},
        ])
    messages.append({"role": "user", "content": "LATEST_REQUEST"})

    compacted = adaptive_engine._apply_context_shift(messages, max_tokens_estimate=300)
    assert any("sıkıştırılmış özeti" in item["content"] for item in compacted)
    assert sum(item["role"] == "system" for item in compacted) == 1
    assert compacted[-1]["content"] == "LATEST_REQUEST"
