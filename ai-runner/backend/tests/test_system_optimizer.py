"""
AI Runner — Unit Tests: System Optimizer
Tests for pagefile analysis, service audit, RAM disk, and prompt pruning budget.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.system_optimizer import (
    analyze_pagefile,
    audit_services,
    get_top_processes,
    analyze_ramdisk,
    calculate_prompt_budget,
    get_optimizer_status,
    get_gpu_profiles,
    lock_cpu_affinity_and_priority,
    flush_vram_cache,
)


class TestSystemOptimizer:

    def test_calculate_prompt_budget(self):
        """Test calculation of token budgets and warning triggers."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am fine, thank you!"},
        ]
        # context = 1000 tokens
        budget = calculate_prompt_budget(
            context_length=1000,
            history_messages=messages,
            system_prompt="You are helpful.",
        )
        assert budget["context_length"] == 1000
        assert budget["total_used"] > 0
        assert budget["remaining"] == 1000 - budget["total_used"]
        assert budget["utilization_pct"] == round((budget["total_used"] / 1000) * 100, 1)

    def test_calculate_prompt_budget_warning(self):
        """Warning flags should trigger when budget usage is high."""
        # Make a very long history message to trigger warnings
        messages = [
            {"role": "user", "content": "A" * 3800},
        ]
        budget = calculate_prompt_budget(
            context_length=1000,
            history_messages=messages,
            system_prompt="",
        )
        assert budget["is_warning"] is True
        assert budget["is_critical"] is True

    def test_analyze_ramdisk_logic(self):
        """RAM disk calculation should recommend safe capacities based on total RAM."""
        ramdisk = analyze_ramdisk(model_size_mb=4000)
        assert ramdisk.physical_ram_mb > 0
        assert ramdisk.available_ram_mb > 0
        assert ramdisk.safe_ramdisk_mb == int(ramdisk.available_ram_mb * 0.40)
        assert len(ramdisk.setup_steps) >= 3
        cmd = ramdisk.powershell_command.lower()
        assert "imdisk" in cmd or "ramdisk" in cmd

    def test_top_processes_contains_valid_fields(self):
        """Top process list should contain pid, name, RAM, and stop commands."""
        procs = get_top_processes(limit=5)
        assert isinstance(procs, list)
        if procs:
            p = procs[0]
            assert p.pid > 0
            assert p.name
            assert p.memory_mb > 0
            assert "Stop-Process" in p.kill_command

    @patch("backend.core.system_optimizer.IS_WINDOWS", False)
    def test_pagefile_on_non_windows(self):
        """Non-Windows pagefile analysis should return clean status and instructions."""
        pagefile = analyze_pagefile(model_size_mb=8000)
        assert pagefile.status == "unavailable"
        assert "fstab" in pagefile.recommendation

    @patch("backend.core.system_optimizer.IS_WINDOWS", True)
    @patch("subprocess.run")
    def test_pagefile_on_windows_mocked(self, mock_run):
        """Windows pagefile parser should parse WMI JSON output successfully."""
        mock_output = MagicMock()
        mock_output.returncode = 0
        mock_output.stdout = '{"AllocatedBaseSize": 16384, "Name": "C:\\\\pagefile.sys"}'
        mock_run.return_value = mock_output

        pagefile = analyze_pagefile(model_size_mb=4000)
        assert pagefile.current_size_mb == 16384
        assert pagefile.current_path == "C:\\pagefile.sys"
        assert pagefile.status in ("ok", "low", "critical")

    def test_optimizer_status_score_calculation(self):
        """Total optimizer scorecard score should be an integer in [0, 100]."""
        status = get_optimizer_status()
        assert 0 <= status.optimization_score <= 100
        assert status.os_name
        assert len(status.recommendations) >= 1

    def test_gpu_profiles_detection(self):
        """GPU profiling should return GPU count, Recommended split, and commands."""
        profile = get_gpu_profiles()
        assert isinstance(profile.gpus, list)
        assert isinstance(profile.tensor_split_recommended, list)
        assert len(profile.tensor_split_recommended) == len(profile.gpus)
        if len(profile.gpus) > 0:
            assert sum(profile.tensor_split_recommended) > 0.0

    def test_lock_cpu_affinity_and_priority(self):
        """Process lock should raise scheduling class or lock affinity cores."""
        res = lock_cpu_affinity_and_priority()
        assert "priority" in res
        assert "affinity" in res

    def test_flush_vram_cache_execution(self):
        """VRAM flushing should trigger memory cleanup gracefully."""
        res = flush_vram_cache()
        assert "status" in res
        assert "bytes_reclaimed" in res

