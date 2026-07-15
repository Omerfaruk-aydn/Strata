"""
AI Runner — Unit Tests: Hardware Profile
Tests for GPU/CPU/RAM/disk detection.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.hardware_profile import (
    GPUInfo, RAMInfo, DiskInfo, CPUInfo, HardwareProfile,
    detect_ram, detect_disk, detect_cpu, check_vram_change,
)


class TestGPUInfo:
    def test_vram_free_is_computed(self):
        gpu = GPUInfo(name="RTX 4060", vram_total_mb=8192, vram_used_mb=1000)
        assert gpu.vram_free_mb == 7192

    def test_vram_free_never_negative(self):
        gpu = GPUInfo(name="GPU", vram_total_mb=4096, vram_used_mb=5000)
        assert gpu.vram_free_mb >= 0

    def test_no_gpu_info(self):
        gpu = GPUInfo(name="No GPU")
        assert gpu.vram_total_mb == 0
        assert gpu.vram_free_mb == 0


class TestRAMInfo:
    def test_percent_used_calculation(self):
        ram = RAMInfo(total_mb=16384, used_mb=8192, free_mb=8192)
        assert abs(ram.percent_used - 50.0) < 1.0

    def test_zero_total_doesnt_raise(self):
        ram = RAMInfo(total_mb=0, free_mb=0)
        assert ram.percent_used == 0


class TestCPUInfo:
    def test_required_fields(self):
        cpu = CPUInfo(name="Intel i9", cores=16, threads=32)
        assert cpu.cores == 16
        assert cpu.threads == 32


class TestDiskInfo:
    def test_ssd_type(self):
        disk = DiskInfo(type="SSD", free_gb=100.0, total_gb=500.0)
        assert disk.type == "SSD"

    def test_hdd_type(self):
        disk = DiskInfo(type="HDD", free_gb=200.0, total_gb=1000.0)
        assert disk.type == "HDD"


class TestCheckVramChange:
    @patch("backend.core.hardware_profile.detect_gpus")
    def test_no_change_returns_none(self, mock_detect_gpus):
        profile1 = HardwareProfile(
            gpu=GPUInfo(name="RTX", vram_total_mb=8192, vram_free_mb=6000),
            ram=RAMInfo(total_mb=32768, free_mb=20000),
            disk=DiskInfo(type="SSD", free_gb=200),
            cpu=CPUInfo(name="i7", cores=8, threads=16),
            selected_gpu_index=0,
        )
        mock_detect_gpus.return_value = [profile1.gpu]
        result = check_vram_change(profile1)
        assert result is None

    @patch("backend.core.hardware_profile.detect_gpus")
    def test_detects_significant_drop(self, mock_detect_gpus):
        # Simulate cached profile with more free VRAM
        old_profile = HardwareProfile(
            gpu=GPUInfo(name="RTX", vram_total_mb=8192, vram_free_mb=7000),
            ram=RAMInfo(total_mb=32768, free_mb=20000),
            disk=DiskInfo(type="SSD", free_gb=200),
            cpu=CPUInfo(name="i7", cores=8, threads=16),
            selected_gpu_index=0,
        )
        new_profile = HardwareProfile(
            gpu=GPUInfo(name="RTX", vram_total_mb=8192, vram_free_mb=5500),
            ram=RAMInfo(total_mb=32768, free_mb=20000),
            disk=DiskInfo(type="SSD", free_gb=200),
            cpu=CPUInfo(name="i7", cores=8, threads=16),
            selected_gpu_index=0,
        )
        mock_detect_gpus.return_value = [new_profile.gpu]
        result = check_vram_change(old_profile)
        # Signficant drop should return warning message
        assert result is not None
        assert "VRAM" in result


class TestDetectFunctions:
    def test_detect_ram_returns_raminfo(self):
        ram = detect_ram()
        assert isinstance(ram, RAMInfo)
        assert ram.total_mb > 0

    def test_detect_cpu_returns_cpuinfo(self):
        cpu = detect_cpu()
        assert isinstance(cpu, CPUInfo)
        assert cpu.cores > 0
        assert cpu.threads >= cpu.cores

    def test_detect_disk_returns_diskinfo(self):
        disk = detect_disk()
        assert isinstance(disk, DiskInfo)
        assert disk.free_gb >= 0
        assert disk.type in ("SSD", "HDD", "Unknown")
