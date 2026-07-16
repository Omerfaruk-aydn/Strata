"""
AI Runner — Hardware Profiler
Detects GPU, CPU, RAM, and disk information.
Implements FR-201, FR-202, FR-203, FR-204.
"""

import platform
import os
import warnings
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
import psutil


# ── Data Models ──

class GPUInfo(BaseModel):
    name: str = "No GPU Detected"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    vram_used_mb: int = 0
    temperature: Optional[float] = None
    driver_version: Optional[str] = None
    index: int = 0

    @model_validator(mode="after")
    def _compute_free(self):
        """Auto-compute vram_free_mb = total - used if not explicitly provided."""
        if self.vram_free_mb == 0 and self.vram_total_mb > 0 and self.vram_used_mb > 0:
            self.vram_free_mb = max(0, self.vram_total_mb - self.vram_used_mb)
        return self


class CPUInfo(BaseModel):
    name: str = "Unknown CPU"
    cores: int = 0
    threads: int = 0
    frequency_mhz: Optional[float] = None


class RAMInfo(BaseModel):
    total_mb: int = 0
    free_mb: int = 0
    used_mb: int = 0
    percent_used: float = 0.0

    @model_validator(mode="after")
    def _compute_percent(self):
        """Auto-compute percent_used = used / total * 100 if total > 0."""
        if self.total_mb > 0 and self.used_mb > 0 and self.percent_used == 0.0:
            self.percent_used = round((self.used_mb / self.total_mb) * 100, 1)
        return self


class VirtualMemoryInfo(BaseModel):
    """Operating-system paging/swap capacity available to mapped models."""

    total_mb: int = 0
    free_mb: int = 0
    used_mb: int = 0
    percent_used: float = 0.0


class DiskInfo(BaseModel):
    type: str = "Unknown"  # SSD or HDD
    free_gb: float = 0.0
    total_gb: float = 0.0
    path: str = ""


class HardwareProfile(BaseModel):
    gpu: GPUInfo
    gpus: List[GPUInfo] = Field(default_factory=list)
    ram: RAMInfo
    disk: DiskInfo
    cpu: CPUInfo
    virtual_memory: VirtualMemoryInfo = Field(default_factory=VirtualMemoryInfo)
    os_info: str = ""
    selected_gpu_index: int = 0


# ── Detection Functions ──

def detect_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs using pynvml. Returns empty list if no GPU found."""
    gpus = []
    pynvml = None
    initialized = False
    try:
        # Some environments contain the deprecated redirector package beside
        # NVIDIA's supported nvidia-ml-py distribution. The public module name
        # remains ``pynvml``, so suppress only that redirector warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The pynvml package is deprecated.*",
                category=FutureWarning,
            )
            import pynvml
        pynvml.nvmlInit()
        initialized = True
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total = mem_info.total // (1024 * 1024)
            vram_free = mem_info.free // (1024 * 1024)
            vram_used = mem_info.used // (1024 * 1024)

            # Temperature
            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass

            # Driver version
            driver = None
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8")
            except Exception:
                pass

            gpus.append(GPUInfo(
                name=name,
                vram_total_mb=vram_total,
                vram_free_mb=vram_free,
                vram_used_mb=vram_used,
                temperature=temp,
                driver_version=driver,
                index=i,
            ))

    except ImportError:
        pass
    except Exception:
        pass
    finally:
        if initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    return gpus


def detect_cpu() -> CPUInfo:
    """Detect CPU information."""
    name = platform.processor() or "Unknown CPU"

    # Try to get a friendlier CPU name on Windows
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            pass

    freq = psutil.cpu_freq()
    return CPUInfo(
        name=name,
        cores=psutil.cpu_count(logical=False) or 1,
        threads=psutil.cpu_count(logical=True) or 1,
        frequency_mhz=freq.current if freq else None,
    )


def detect_ram() -> RAMInfo:
    """Detect system RAM information."""
    mem = psutil.virtual_memory()
    return RAMInfo(
        total_mb=mem.total // (1024 * 1024),
        free_mb=mem.available // (1024 * 1024),
        used_mb=mem.used // (1024 * 1024),
        percent_used=mem.percent,
    )


def detect_virtual_memory() -> VirtualMemoryInfo:
    """Detect pagefile/swap capacity without mutating operating-system settings."""
    swap = psutil.swap_memory()
    return VirtualMemoryInfo(
        total_mb=swap.total // (1024 * 1024),
        free_mb=swap.free // (1024 * 1024),
        used_mb=swap.used // (1024 * 1024),
        percent_used=round(float(swap.percent), 1),
    )


def detect_disk(path: Optional[str] = None) -> DiskInfo:
    """Detect disk information for the given path or default model directory."""
    if path is None:
        path = os.path.expanduser("~")

    try:
        usage = psutil.disk_usage(path)
        total_gb = usage.total / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
    except Exception:
        total_gb = 0.0
        free_gb = 0.0

    # Determine disk type (SSD vs HDD)
    disk_type = _detect_disk_type(path)

    return DiskInfo(
        type=disk_type,
        free_gb=round(free_gb, 1),
        total_gb=round(total_gb, 1),
        path=path,
    )


def _detect_disk_type(path: str) -> str:
    """Attempt to detect if the disk is SSD or HDD."""
    system = platform.system()

    if system == "Windows":
        try:
            import subprocess
            drive = os.path.splitdrive(os.path.abspath(path))[0]
            result = subprocess.run(
                ["powershell", "-Command",
                 f"Get-PhysicalDisk | Where-Object {{ $_.DeviceID -eq 0 }} | Select-Object MediaType | Format-List"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout.lower()
            if "ssd" in output or "solid" in output:
                return "SSD"
            elif "hdd" in output or "unspecified" in output:
                return "HDD"
        except Exception:
            pass
    elif system == "Linux":
        try:
            import subprocess
            result = subprocess.run(
                ["lsblk", "-d", "-o", "name,rota"],
                capture_output=True, text=True, timeout=5
            )
            # rota=0 means SSD, rota=1 means HDD
            if "0" in result.stdout:
                return "SSD"
            elif "1" in result.stdout:
                return "HDD"
        except Exception:
            pass
    elif system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPStorageDataType"],
                capture_output=True, text=True, timeout=5
            )
            if "Solid State" in result.stdout:
                return "SSD"
        except Exception:
            pass

    return "SSD"  # Default assumption for modern systems


def get_hardware_profile(
    model_dir: Optional[str] = None,
    selected_gpu: int = 0
) -> HardwareProfile:
    """
    Generate a complete hardware profile.
    FR-201: Auto-detect on startup
    FR-202: Multi-GPU selection
    FR-203: Hardware Card display
    """
    gpus = detect_gpus()
    cpu = detect_cpu()
    ram = detect_ram()
    virtual_memory = detect_virtual_memory()
    disk = detect_disk(model_dir)

    # Select the active GPU
    active_gpu = GPUInfo()  # fallback: no GPU
    if gpus and 0 <= selected_gpu < len(gpus):
        active_gpu = gpus[selected_gpu]
    elif gpus:
        active_gpu = gpus[0]

    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    return HardwareProfile(
        gpu=active_gpu,
        gpus=gpus,
        ram=ram,
        virtual_memory=virtual_memory,
        disk=disk,
        cpu=cpu,
        os_info=os_info,
        selected_gpu_index=selected_gpu if gpus else -1,
    )


def check_vram_change(
    previous: HardwareProfile,
    threshold_pct: float = 15.0
) -> Optional[str]:
    """
    FR-204: Detect significant VRAM changes (e.g., another app using GPU).
    Returns a warning message if VRAM changed significantly, else None.
    """
    current_gpus = detect_gpus()
    if not current_gpus or previous.selected_gpu_index < 0:
        return None

    idx = previous.selected_gpu_index
    if idx >= len(current_gpus):
        return None

    current_free = current_gpus[idx].vram_free_mb
    previous_free = previous.gpu.vram_free_mb

    if previous_free == 0:
        return None

    change_pct = abs(current_free - previous_free) / previous_free * 100

    if change_pct >= threshold_pct:
        diff = previous_free - current_free
        if diff > 0:
            return (
                f"VRAM kullanımı değişti: {abs(diff)} MB daha az boş alan. "
                f"Başka bir uygulama GPU belleğini kullanıyor olabilir."
            )
        else:
            return (
                f"VRAM kullanımı değişti: {abs(diff)} MB daha fazla boş alan. "
                f"GPU belleği serbest bırakıldı."
            )

    return None
