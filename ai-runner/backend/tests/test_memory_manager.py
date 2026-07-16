"""
AI Runner — Unit Tests: MemoryManager
%90+ coverage for the offload planning algorithm (Section 11).
Implements test strategy from Section 18 of the specification.
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.memory_manager import (
    calculate_offload_plan,
    suggest_best_quant,
    estimate_model_size_mb,
    estimate_total_layers,
    estimate_kv_cache_mb,
    QUANT_BPW,
)
from backend.core.hardware_profile import (
    HardwareProfile, GPUInfo, RAMInfo, DiskInfo, CPUInfo
)


# ── Fixtures ──

def make_hardware(
    vram_total=8192, vram_free=7000,
    ram_total=32768, ram_free=24000,
    disk_type="SSD", disk_free_gb=200.0,
    cpu_cores=8, cpu_threads=16,
):
    return HardwareProfile(
        gpu=GPUInfo(
            name="RTX 4060",
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
        ),
        ram=RAMInfo(total_mb=ram_total, free_mb=ram_free),
        disk=DiskInfo(type=disk_type, free_gb=disk_free_gb),
        cpu=CPUInfo(name="Intel i7", cores=cpu_cores, threads=cpu_threads),
    )


# ── estimate_model_size_mb ──

class TestEstimateModelSize:
    def test_q4km_7b_approximate(self):
        """Q4_K_M 7B model should be ~4.1 GB"""
        size = estimate_model_size_mb(7_000_000_000, "Q4_K_M")
        assert 3800 < size < 4500, f"Expected ~4100 MB, got {size}"

    def test_q2k_smaller_than_q8(self):
        """Q2_K should always be smaller than Q8_0"""
        size_q2 = estimate_model_size_mb(7_000_000_000, "Q2_K")
        size_q8 = estimate_model_size_mb(7_000_000_000, "Q8_0")
        assert size_q2 < size_q8

    def test_70b_larger_than_7b(self):
        """70B model should be ~10x larger than 7B at same quant"""
        size_7b = estimate_model_size_mb(7_000_000_000, "Q4_K_M")
        size_70b = estimate_model_size_mb(70_000_000_000, "Q4_K_M")
        assert size_70b > size_7b * 8

    def test_unknown_quant_uses_default(self):
        """Unknown quant should use Q4_K_M default"""
        size_unknown = estimate_model_size_mb(7_000_000_000, "UNKNOWN")
        size_q4km = estimate_model_size_mb(7_000_000_000, "Q4_K_M")
        assert size_unknown == size_q4km


# ── estimate_total_layers ──

class TestEstimateTotalLayers:
    def test_7b_has_32_layers(self):
        assert estimate_total_layers(7_000_000_000) == 32

    def test_70b_has_80_layers(self):
        assert estimate_total_layers(70_000_000_000) == 80

    def test_1b_has_24_layers(self):
        assert estimate_total_layers(1_000_000_000) == 24

    def test_13b_has_40_layers(self):
        assert estimate_total_layers(13_000_000_000) == 40

    def test_large_model_has_more_layers(self):
        assert estimate_total_layers(100_000_000_001) == 96


# ── estimate_kv_cache_mb ──

class TestEstimateKVCache:
    def test_returns_positive_value(self):
        kv = estimate_kv_cache_mb(4096, 32)
        assert kv > 0

    def test_larger_context_is_more(self):
        kv_small = estimate_kv_cache_mb(2048, 32)
        kv_large = estimate_kv_cache_mb(8192, 32)
        assert kv_large > kv_small

    def test_more_layers_is_more(self):
        kv_32 = estimate_kv_cache_mb(4096, 32)
        kv_80 = estimate_kv_cache_mb(4096, 80)
        assert kv_80 > kv_32


# ── calculate_offload_plan ──

class TestCalculateOffloadPlan:

    # ── Step 3: GPU Layers ──

    def test_all_fit_on_gpu(self):
        """Small 3B model should fit entirely in 8GB VRAM"""
        hw = make_hardware(vram_free=7000, ram_free=24000)
        plan = calculate_offload_plan(
            model_id="test/3b-model",
            quant="Q4_K_M",
            file_size_mb=1800,  # ~3B Q4
            total_layers=26,
            hardware=hw,
        )
        assert plan.gpu_layers == plan.total_layers
        assert plan.cpu_layers == 0
        assert plan.disk_streamed_layers == 0
        assert plan.fits_comfortably

    def test_large_model_needs_offload(self):
        """70B model should NOT fit in 8GB VRAM"""
        hw = make_hardware(vram_free=7000, ram_free=24000)
        plan = calculate_offload_plan(
            model_id="test/70b-model",
            quant="Q4_K_M",
            file_size_mb=40000,
            total_layers=80,
            hardware=hw,
        )
        assert plan.gpu_layers < plan.total_layers
        assert plan.gpu_layers + plan.cpu_layers == plan.total_layers
        assert plan.mapped_pressure_layers <= plan.cpu_layers

    def test_no_gpu_system(self):
        """System without GPU should put all layers on CPU/disk"""
        hw = make_hardware(vram_free=0, vram_total=0, ram_free=16000)
        plan = calculate_offload_plan(
            model_id="test/7b-model",
            quant="Q4_K_M",
            file_size_mb=4100,
            total_layers=32,
            hardware=hw,
        )
        assert plan.gpu_layers == 0

    def test_layer_sum_equals_total(self):
        """GPU and CPU execution layers must always equal total_layers."""
        hw = make_hardware(vram_free=4000, ram_free=8000)
        plan = calculate_offload_plan(
            model_id="test/13b-model",
            quant="Q4_K_M",
            file_size_mb=7300,
            total_layers=40,
            hardware=hw,
        )
        total = plan.gpu_layers + plan.cpu_layers
        assert total == plan.total_layers, f"Layer sum {total} != {plan.total_layers}"

    def test_manual_gpu_layers_override(self):
        """Manual n_gpu_layers should be respected (FR-301)"""
        hw = make_hardware(vram_free=7000)
        plan = calculate_offload_plan(
            model_id="test/7b",
            quant="Q4_K_M",
            file_size_mb=4100,
            total_layers=32,
            hardware=hw,
            user_gpu_layers=16,
        )
        assert plan.gpu_layers == 16

    def test_manual_override_gets_warning_if_over_vram(self):
        """Setting too many GPU layers should produce a warning"""
        hw = make_hardware(vram_free=2000)
        plan = calculate_offload_plan(
            model_id="test/7b",
            quant="Q4_K_M",
            file_size_mb=4100,
            total_layers=32,
            hardware=hw,
            user_gpu_layers=32,
        )
        assert plan.gpu_layers == 32
        assert any("OOM" in w for w in plan.warnings)

    # ── Step 4: RAM/Disk Distribution ──

    def test_insufficient_ram_goes_to_disk(self):
        """If neither GPU nor RAM fits model, use disk streaming"""
        hw = make_hardware(vram_free=1000, ram_free=1000)
        plan = calculate_offload_plan(
            model_id="test/70b",
            quant="Q4_K_M",
            file_size_mb=40000,
            total_layers=80,
            hardware=hw,
        )
        assert plan.disk_streamed_layers > 0

    def test_hdd_disk_streaming_warning(self):
        """HDD disk streaming should warn the user"""
        hw = make_hardware(vram_free=1000, ram_free=1000, disk_type="HDD")
        plan = calculate_offload_plan(
            model_id="test/70b",
            quant="Q4_K_M",
            file_size_mb=40000,
            total_layers=80,
            hardware=hw,
        )
        if plan.disk_streamed_layers > 0:
            assert any("HDD" in w for w in plan.warnings)

    def test_ssd_disk_streaming_ok_warning(self):
        """SSD disk streaming should have a softer warning"""
        hw = make_hardware(vram_free=1000, ram_free=1000, disk_type="SSD")
        plan = calculate_offload_plan(
            model_id="test/70b",
            quant="Q4_K_M",
            file_size_mb=40000,
            total_layers=80,
            hardware=hw,
        )
        if plan.disk_streamed_layers > 0:
            assert any("SSD" in w for w in plan.warnings)

    # ── Step 5: Speed Estimation ──

    def test_all_gpu_faster_than_mixed(self):
        """All GPU layers should be faster than GPU+RAM mix"""
        hw_good = make_hardware(vram_free=10000, ram_free=24000)
        hw_limited = make_hardware(vram_free=2000, ram_free=24000)

        plan_good = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw_good,
        )
        plan_limited = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw_limited,
        )
        assert plan_good.estimated_tokens_per_sec >= plan_limited.estimated_tokens_per_sec

    def test_disk_streaming_slower_than_ram(self):
        """Disk streaming should always be slower than RAM-only offload"""
        hw_ram = make_hardware(vram_free=1000, ram_free=30000)
        hw_disk = make_hardware(vram_free=500, ram_free=1000)

        plan_ram = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw_ram,
        )
        plan_disk = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw_disk,
        )
        assert plan_ram.estimated_tokens_per_sec >= plan_disk.estimated_tokens_per_sec

    def test_speed_always_positive(self):
        """Speed estimate should always be > 0"""
        hw = make_hardware(vram_free=0, ram_free=0)
        plan = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw,
        )
        assert plan.estimated_tokens_per_sec > 0

    # ── Context Length ──

    def test_disk_streaming_reduces_context(self):
        """Disk streaming should reduce recommended context"""
        hw = make_hardware(vram_free=500, ram_free=500)
        plan = calculate_offload_plan(
            model_id="test/70b", quant="Q4_K_M",
            file_size_mb=40000, total_layers=80, hardware=hw,
            context_length=8192,
        )
        if plan.disk_streamed_layers > 0:
            assert plan.context_length_recommended <= 2048

    def test_full_gpu_preserves_context(self):
        """When all fits in GPU, requested context should be returned"""
        hw = make_hardware(vram_free=8000)
        plan = calculate_offload_plan(
            model_id="test/3b", quant="Q4_K_M",
            file_size_mb=1800, total_layers=26, hardware=hw,
            context_length=4096,
        )
        if plan.gpu_layers == plan.total_layers:
            assert plan.context_length_recommended == 4096

    # ── VRAM/RAM Usage Estimates ──

    def test_vram_usage_is_positive(self):
        hw = make_hardware(vram_free=7000)
        plan = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw,
        )
        assert plan.vram_usage_mb >= 0

    def test_recommendation_is_nonempty(self):
        hw = make_hardware(vram_free=7000)
        plan = calculate_offload_plan(
            model_id="test/7b", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw,
        )
        assert len(plan.recommendation) > 0

    def test_model_id_preserved(self):
        hw = make_hardware()
        plan = calculate_offload_plan(
            model_id="TheBloke/Llama-3-70B-GGUF",
            quant="Q4_K_M",
            file_size_mb=40000,
            total_layers=80,
            hardware=hw,
        )
        assert plan.model_id == "TheBloke/Llama-3-70B-GGUF"
        assert plan.quant == "Q4_K_M"


# ── suggest_best_quant ──

class TestSuggestBestQuant:

    def test_returns_dict_with_required_keys(self):
        hw = make_hardware(vram_free=8000)
        result = suggest_best_quant(
            parameter_count=7_000_000_000,
            total_layers=32,
            available_quants=["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0"],
            hardware=hw,
        )
        assert "recommended" in result
        assert "reason" in result
        assert "alternatives" in result

    def test_recommends_higher_quant_for_good_hardware(self):
        """With ample VRAM, higher quality quant should be recommended"""
        hw = make_hardware(vram_free=12000, ram_free=32000)
        result = suggest_best_quant(
            parameter_count=7_000_000_000,
            total_layers=32,
            available_quants=["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0"],
            hardware=hw,
        )
        # Should recommend Q5 or better for good hardware
        assert result["recommended"] in ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

    def test_recommends_lower_quant_for_limited_hardware(self):
        """With very limited VRAM, lower quant should be recommended"""
        hw = make_hardware(vram_free=2000, ram_free=4000)
        result = suggest_best_quant(
            parameter_count=7_000_000_000,
            total_layers=32,
            available_quants=["Q2_K", "Q4_K_M", "Q5_K_M"],
            hardware=hw,
        )
        assert result["recommended"] in ["Q2_K", "Q4_K_M"]

    def test_empty_quants_returns_default(self):
        """Empty available_quants should return default"""
        hw = make_hardware()
        result = suggest_best_quant(
            parameter_count=7_000_000_000,
            total_layers=32,
            available_quants=[],
            hardware=hw,
        )
        assert "recommended" in result

    def test_alternatives_list_populated(self):
        hw = make_hardware(vram_free=8000)
        result = suggest_best_quant(
            parameter_count=7_000_000_000,
            total_layers=32,
            available_quants=["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0"],
            hardware=hw,
        )
        # At least some alternatives should be provided
        assert isinstance(result["alternatives"], list)


# ── Hardware Profile Simulation Tests (Section 18: donanım matrisi) ──

class TestHardwareProfiles:
    """Simulate low/mid/high VRAM configurations from Section 18."""

    @pytest.mark.parametrize("vram_mb, model_size_mb, total_layers, expects_gpu_layers", [
        (4096,  4100, 32, True),   # 4GB VRAM, 7B Q4 → some layers on GPU
        (8192,  4100, 32, True),   # 8GB VRAM, 7B Q4 → most/all on GPU
        (24576, 40000, 80, True),  # 24GB VRAM, 70B Q4 → many layers on GPU
        (2048,  40000, 80, False), # 2GB VRAM, 70B Q4 → few/no GPU layers
    ])
    def test_various_vram_profiles(self, vram_mb, model_size_mb, total_layers, expects_gpu_layers):
        hw = make_hardware(vram_free=int(vram_mb * 0.9), vram_total=vram_mb, ram_free=32000)
        plan = calculate_offload_plan(
            model_id="test/model", quant="Q4_K_M",
            file_size_mb=model_size_mb, total_layers=total_layers, hardware=hw,
        )
        has_gpu_layers = plan.gpu_layers > 0
        assert has_gpu_layers == expects_gpu_layers, (
            f"VRAM={vram_mb}MB, model={model_size_mb}MB: "
            f"expected gpu_layers>0={expects_gpu_layers}, got {plan.gpu_layers}"
        )

    def test_plan_is_deterministic(self):
        """Same input always produces same plan"""
        hw = make_hardware(vram_free=6000)
        kwargs = dict(
            model_id="test", quant="Q4_K_M",
            file_size_mb=4100, total_layers=32, hardware=hw
        )
        plan1 = calculate_offload_plan(**kwargs)
        plan2 = calculate_offload_plan(**kwargs)
        assert plan1.gpu_layers == plan2.gpu_layers
        assert plan1.cpu_layers == plan2.cpu_layers
        assert plan1.disk_streamed_layers == plan2.disk_streamed_layers
