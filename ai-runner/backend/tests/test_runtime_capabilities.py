"""Native runtime capability and quantization-service tests."""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from backend.core import runtime_capabilities as runtime_module
from backend.core.hardware_profile import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareProfile,
    RAMInfo,
)
from backend.core.quantization_service import QuantizationManager
from backend.core.runtime_capabilities import (
    RuntimeCapabilities,
    detect_runtime_capabilities,
    validate_backend_preference,
)


def hardware_with_nvidia() -> HardwareProfile:
    gpu = GPUInfo(name="NVIDIA RTX", vram_total_mb=20_000, vram_free_mb=18_000)
    return HardwareProfile(
        gpu=gpu,
        gpus=[gpu],
        ram=RAMInfo(total_mb=64_000, free_mb=50_000),
        disk=DiskInfo(type="SSD", free_gb=100),
        cpu=CPUInfo(name="CPU", cores=8, threads=16),
    )


def minimal_gguf(payload: bytes = b"") -> bytes:
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
        + payload
    )


def test_runtime_report_identifies_active_cuda(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "_inspect_llama_runtime",
        lambda: (True, "1.2.3", True, "CUDA CUBLAS", "cuda"),
    )
    monkeypatch.setattr(runtime_module, "_find_binary", lambda *args, **kwargs: None)
    report = detect_runtime_capabilities(hardware_with_nvidia())
    assert report.llama_cpp_installed is True
    assert report.active_backend == "cuda"
    assert report.gpu_offload_supported is True
    assert next(item for item in report.backends if item.name == "cuda").active is True


def test_cpu_runtime_warns_when_gpu_exists(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "_inspect_llama_runtime",
        lambda: (True, "1.2.3", False, "AVX2", "cpu"),
    )
    monkeypatch.setattr(runtime_module, "_find_binary", lambda *args, **kwargs: None)
    report = detect_runtime_capabilities(hardware_with_nvidia())
    assert report.active_backend == "cpu"
    assert any("CPU-only" in note for note in report.notes)
    cuda = next(item for item in report.backends if item.name == "cuda")
    assert cuda.available is False


def test_explicit_backend_must_match_native_build():
    report = RuntimeCapabilities(
        llama_cpp_installed=True,
        active_backend="cuda",
        gpu_offload_supported=True,
    )
    validate_backend_preference("auto", report)
    validate_backend_preference("cuda", report)
    validate_backend_preference("cpu", report)
    with pytest.raises(RuntimeError, match="not active"):
        validate_backend_preference("vulkan", report)


def test_quantization_progress_parser():
    assert QuantizationManager._parse_progress("quantizing 42.5 %") == 0.425
    assert QuantizationManager._parse_progress("loading model") == 0.08
    assert QuantizationManager._parse_progress("writing output") == 0.9
    assert QuantizationManager._parse_progress("unrelated line") is None


@pytest.mark.asyncio
async def test_quantization_rejects_unsafe_source_path(tmp_path):
    executable = tmp_path / "llama-quantize.exe"
    executable.write_bytes(b"binary")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF" + b"x" * 32)
    manager = QuantizationManager()

    with pytest.raises(ValueError, match="model directory"):
        await manager.start_job(
            executable=str(executable),
            model_id="test/model",
            source_path=str(outside),
            model_dir=str(model_dir),
            output_quant="Q3_K_M",
            threads=4,
            allow_requantize=False,
        )


@pytest.mark.asyncio
async def test_quantization_rejects_unknown_quant(tmp_path):
    executable = tmp_path / "llama-quantize.exe"
    executable.write_bytes(b"binary")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "source.gguf"
    source.write_bytes(minimal_gguf(b"source"))
    manager = QuantizationManager()

    with pytest.raises(ValueError, match="Unsupported"):
        await manager.start_job(
            executable=str(executable),
            model_id="test/model",
            source_path=str(source),
            model_dir=str(model_dir),
            output_quant="NOT_A_QUANT",
            threads=4,
            allow_requantize=False,
        )


class _FakeStdout:
    def __init__(self, lines):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeProcess:
    def __init__(self, lines, return_code):
        self.stdout = _FakeStdout(lines)
        self.returncode = None
        self._return_code = return_code

    async def wait(self):
        await asyncio.sleep(0)
        self.returncode = self._return_code
        return self.returncode


@pytest.mark.asyncio
async def test_quantization_job_completes_and_builds_safe_command(tmp_path, monkeypatch):
    executable = tmp_path / "llama-quantize.exe"
    executable.write_bytes(b"binary")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "source-Q8_0.gguf"
    source.write_bytes(minimal_gguf(b"source"))
    captured = {}

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[-3])
        output_path.write_bytes(minimal_gguf(b"quantized"))
        return _FakeProcess([b"loading model\n", b"quantizing 55 %\n", b"writing output\n"], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    manager = QuantizationManager()
    job = await manager.start_job(
        executable=str(executable),
        model_id="test/model",
        source_path=str(source),
        model_dir=str(model_dir),
        output_quant="Q3_K_M",
        threads=4,
        allow_requantize=True,
    )
    task = manager._tasks[job.id]
    await task

    assert job.status == "completed"
    assert job.progress == 1.0
    assert Path(job.output_path).read_bytes().startswith(b"GGUF")
    assert captured["command"][1] == "--allow-requantize"
    assert captured["command"][-2:] == ("Q3_K_M", "4")
    assert "shell" not in captured["kwargs"]


@pytest.mark.asyncio
async def test_quantization_failure_removes_partial_output(tmp_path, monkeypatch):
    executable = tmp_path / "llama-quantize.exe"
    executable.write_bytes(b"binary")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "source.gguf"
    source.write_bytes(minimal_gguf(b"source"))

    async def fake_create_subprocess_exec(*command, **kwargs):
        Path(command[-3]).write_bytes(b"partial")
        return _FakeProcess([b"fatal quantization error\n"], 2)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    manager = QuantizationManager()
    job = await manager.start_job(
        executable=str(executable),
        model_id="test/model",
        source_path=str(source),
        model_dir=str(model_dir),
        output_quant="Q3_K_M",
        threads=4,
        allow_requantize=False,
    )
    task = manager._tasks[job.id]
    await task

    assert job.status == "failed"
    assert "fatal quantization error" in (job.error or "")
    assert not Path(job.output_path).exists()
