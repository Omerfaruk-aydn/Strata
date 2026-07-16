"""Runtime/backend capability discovery for AI Runner.

llama.cpp compute backends are selected when the native library is built.  This
module reports what the installed runtime can actually do and deliberately does
not pretend that a CUDA wheel can be switched to Vulkan inside the same process.
"""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .hardware_profile import HardwareProfile


BackendName = Literal["cuda", "vulkan", "metal", "sycl", "cpu", "unknown"]


class BackendCapability(BaseModel):
    name: BackendName
    available: bool
    active: bool = False
    gpu_offload: bool = False
    reason: str = ""


class RuntimeCapabilities(BaseModel):
    llama_cpp_installed: bool = False
    llama_cpp_version: Optional[str] = None
    active_backend: BackendName = "unknown"
    gpu_offload_supported: bool = False
    system_info: str = ""
    backends: List[BackendCapability] = Field(default_factory=list)
    llama_quantize_path: Optional[str] = None
    llama_cli_path: Optional[str] = None
    restart_required_to_change_backend: bool = True
    notes: List[str] = Field(default_factory=list)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _find_binary(environment_key: str, names: List[str]) -> Optional[str]:
    configured = os.environ.get(environment_key, "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)

    for name in names:
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())

    search_roots = [
        Path(sys.executable).resolve().parent,
        Path(__file__).resolve().parents[2] / "bin",
        Path.home() / ".ai-runner" / "bin",
    ]
    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def _detect_active_backend(system_info: str, gpu_offload: bool) -> BackendName:
    info = system_info.lower()
    if "cuda" in info or "cublas" in info:
        return "cuda"
    if "vulkan" in info:
        return "vulkan"
    if "metal" in info:
        return "metal"
    if "sycl" in info or "oneapi" in info:
        return "sycl"
    if gpu_offload:
        return "unknown"
    return "cpu"


@lru_cache(maxsize=1)
def _inspect_llama_runtime() -> tuple[bool, Optional[str], bool, str, BackendName]:
    try:
        import llama_cpp

        version = getattr(llama_cpp, "__version__", None)
        native = getattr(llama_cpp, "llama_cpp", None)
        gpu_offload = False
        system_info = ""
        if native is not None:
            supports = getattr(native, "llama_supports_gpu_offload", None)
            if callable(supports):
                gpu_offload = bool(supports())
            print_info = getattr(native, "llama_print_system_info", None)
            if callable(print_info):
                system_info = _decode(print_info())
        backend = _detect_active_backend(system_info, gpu_offload)
        return True, version, gpu_offload, system_info, backend
    except Exception:
        return False, None, False, "", "unknown"


def detect_runtime_capabilities(
    hardware: Optional[HardwareProfile] = None,
    *,
    refresh: bool = False,
) -> RuntimeCapabilities:
    """Return installed backend and companion-tool capabilities."""
    if refresh:
        _inspect_llama_runtime.cache_clear()
    installed, version, gpu_offload, system_info, active_backend = _inspect_llama_runtime()

    has_nvidia = bool(
        hardware and any(
            "nvidia" in str(getattr(gpu, "name", "")).lower()
            or "rtx" in str(getattr(gpu, "name", "")).lower()
            for gpu in getattr(hardware, "gpus", [])
        )
    )
    backend_names: List[BackendName] = ["cuda", "vulkan", "metal", "sycl", "cpu"]
    capabilities: List[BackendCapability] = []
    for name in backend_names:
        active = name == active_backend
        if name == "cpu":
            available = installed
            reason = "CPU inference is available through the installed llama.cpp runtime." if installed else "llama-cpp-python is not installed."
        elif active:
            available = True
            reason = f"The installed native llama.cpp library was built with {name.upper()} support."
        elif name == "cuda" and has_nvidia:
            available = False
            reason = "An NVIDIA GPU is present, but the installed llama.cpp runtime is not CUDA-enabled."
        else:
            available = False
            reason = f"A separate {name.upper()}-enabled llama.cpp build is required."
        capabilities.append(BackendCapability(
            name=name,
            available=available,
            active=active,
            gpu_offload=active and gpu_offload,
            reason=reason,
        ))

    notes: List[str] = []
    if installed and active_backend == "cpu" and hardware and hardware.gpus:
        notes.append("GPU detected, but the active llama.cpp build is CPU-only.")
    if not installed:
        notes.append("Inference is unavailable until a compatible llama-cpp-python runtime is installed.")
    if active_backend == "unknown" and gpu_offload:
        notes.append("GPU offload is supported, but the native backend name could not be identified reliably.")

    return RuntimeCapabilities(
        llama_cpp_installed=installed,
        llama_cpp_version=version,
        active_backend=active_backend,
        gpu_offload_supported=gpu_offload,
        system_info=system_info[:4096],
        backends=capabilities,
        llama_quantize_path=_find_binary(
            "AI_RUNNER_LLAMA_QUANTIZE",
            ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"],
        ),
        llama_cli_path=_find_binary(
            "AI_RUNNER_LLAMA_CLI",
            ["llama-cli.exe", "llama-cli", "main.exe", "main"],
        ),
        notes=notes,
    )


def validate_backend_preference(
    preference: str,
    capabilities: RuntimeCapabilities,
) -> None:
    """Validate a requested backend against the native library in this process."""
    normalized = (preference or "auto").lower()
    if normalized == "auto":
        return
    if normalized == "cpu":
        if not capabilities.llama_cpp_installed:
            raise RuntimeError("CPU backend is unavailable because llama-cpp-python is not installed.")
        return
    if normalized != capabilities.active_backend:
        raise RuntimeError(
            f"Requested backend '{normalized}' is not active. The installed llama.cpp runtime uses "
            f"'{capabilities.active_backend}'. Install the matching native build and restart AI Runner."
        )
