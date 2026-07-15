"""
AI Runner — System Optimizer Core
Provides hardware-level performance analysis and optimization recommendations
without requiring elevated privileges.

Features:
  ① Pagefile (Virtual Memory) analysis and recommendations
  ② Background service & process memory audit
  ③ RAM Disk feasibility analysis and setup guide
  ④ Prompt pruning context budget calculation
"""

import os
import platform
import subprocess
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import psutil
import logging

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

class PagefileInfo(BaseModel):
    """Current pagefile configuration and recommendations."""
    current_size_mb: int = 0
    current_path: str = ""
    system_managed: bool = True
    physical_ram_mb: int = 0
    recommended_min_mb: int = 0
    recommended_max_mb: int = 0
    recommendation: str = ""
    powershell_command: str = ""
    status: str = "unknown"   # "ok" | "low" | "critical" | "unavailable"


class ServiceInfo(BaseModel):
    """A running Windows service that can be safely optimized."""
    name: str
    display_name: str
    status: str                 # "running" | "stopped"
    memory_mb: float = 0.0
    description: str
    description_en: str
    risk: str                   # "safe" | "moderate" | "caution"
    stop_command: str           # PowerShell command to stop
    disable_command: str        # PowerShell command to disable


class ProcessInfo(BaseModel):
    """A running process consuming significant memory."""
    pid: int
    name: str
    memory_mb: float
    cpu_percent: float
    kill_command: str


class RamDiskInfo(BaseModel):
    """RAM Disk feasibility analysis."""
    physical_ram_mb: int
    available_ram_mb: int
    safe_ramdisk_mb: int         # RAM we can safely dedicate to ramdisk
    recommended_ramdisk_mb: int
    imdisk_installed: bool
    ramdisk_drives: List[str]    # Already existing RAM disk drives
    setup_steps: List[str]       # Step-by-step instructions
    powershell_command: str
    status: str                  # "ready" | "insufficient" | "already_exists"


class PromptPruningConfig(BaseModel):
    """Context budget configuration for prompt pruning."""
    max_context_tokens: int = 4096
    max_history_messages: int = 20
    token_warning_threshold: float = 0.80   # Warn at 80% full
    auto_prune: bool = True
    estimated_tokens_per_message: int = 150


class SystemOptimizerStatus(BaseModel):
    """Overall system optimization status summary."""
    os_name: str
    total_ram_mb: int
    available_ram_mb: int
    pagefile_status: str
    top_memory_process: str
    optimization_score: int      # 0-100
    recommendations: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Known Windows Services — Safe to Disable for LLM Workloads
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_HEAVY_SERVICES: List[Dict[str, Any]] = [
    {
        "name": "WSearch",
        "display_name": "Windows Search",
        "description": "Windows arama dizinleme servisi. RAM tüketimi: 200-500 MB.",
        "description_en": "Windows search indexing. RAM usage: 200-500 MB.",
        "risk": "safe",
    },
    {
        "name": "SysMain",
        "display_name": "SysMain (Superfetch)",
        "description": "Uygulamaları önceden RAM'e yükler. LLM çalıştırırken bu RAM modele gerekir.",
        "description_en": "Preloads apps into RAM. That RAM is needed for LLM.",
        "risk": "safe",
    },
    {
        "name": "DiagTrack",
        "display_name": "Connected User Experiences (Telemetry)",
        "description": "Microsoft'a kullanım verisi gönderir. Arka planda disk ve CPU kullanır.",
        "description_en": "Sends usage data to Microsoft. Uses background CPU and disk.",
        "risk": "safe",
    },
    {
        "name": "wuauserv",
        "display_name": "Windows Update",
        "description": "Otomatik güncellemeleri kontrol eder. Çalışırken disk I/O'yu engeller.",
        "description_en": "Checks for updates. Blocks disk I/O when running.",
        "risk": "moderate",
    },
    {
        "name": "WinDefend",
        "display_name": "Windows Defender Antivirus",
        "description": "Gerçek zamanlı virüs taraması. Model dosyası okunurken her bloğu tarar (%30 disk yavaşlaması).",
        "description_en": "Real-time antivirus. Scans every block when model file is read (30% disk slowdown).",
        "risk": "caution",
    },
    {
        "name": "BITS",
        "display_name": "Background Intelligent Transfer Service",
        "description": "Arka planda dosya indirme servisi. Windows Update ile bağlantılı.",
        "description_en": "Background file downloads, linked to Windows Update.",
        "risk": "safe",
    },
    {
        "name": "OneDrive",
        "display_name": "Microsoft OneDrive",
        "description": "Bulut senkronizasyon servisi. Sürekli disk I/O ve CPU kullanır.",
        "description_en": "Cloud sync. Constant disk I/O and CPU usage.",
        "risk": "safe",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ① Pagefile Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pagefile(model_size_mb: int = 0) -> PagefileInfo:
    """
    Read current Windows pagefile configuration and generate recommendations.
    Uses WMI via PowerShell on Windows, returns best-effort info on Linux/macOS.
    """
    ram = psutil.virtual_memory()
    ram_mb = ram.total // (1024 * 1024)

    # Recommended: at least RAM + model_size, minimum 16 GB
    recommended_min = max(16 * 1024, ram_mb + model_size_mb)
    recommended_max = recommended_min + 8 * 1024   # add 8 GB headroom

    if not IS_WINDOWS:
        return PagefileInfo(
            physical_ram_mb=ram_mb,
            recommended_min_mb=recommended_min,
            recommended_max_mb=recommended_max,
            recommendation="Linux/macOS: swap boyutunu artırmak için /etc/fstab düzenleyin.",
            powershell_command="sudo fallocate -l 32G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile",
            status="unavailable",
        )

    # Query pagefile via PowerShell WMI
    current_mb = 0
    current_path = "C:\\pagefile.sys"
    system_managed = True

    try:
        result = subprocess.run(
            [
                "powershell", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize | ConvertTo-Json"
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            import json
            data = json.loads(result.stdout.strip())
            if isinstance(data, dict):
                current_mb = int(data.get("AllocatedBaseSize", 0))
                current_path = data.get("Name", current_path)
            elif isinstance(data, list) and data:
                current_mb = int(data[0].get("AllocatedBaseSize", 0))
                current_path = data[0].get("Name", current_path)
            system_managed = current_mb == 0
    except Exception as e:
        logger.warning(f"Pagefile WMI query failed: {e}")

    # Determine status
    if current_mb == 0:
        status = "unknown"
    elif current_mb < recommended_min:
        status = "low"
    elif current_mb >= recommended_min:
        status = "ok"
    else:
        status = "critical"

    # Build PowerShell command (sets pagefile to recommended_max on C:)
    ps_cmd = (
        f"$cs = Get-CimInstance Win32_ComputerSystem; "
        f"$cs.AutomaticManagedPagefile = $false; "
        f"$cs | Set-CimInstance; "
        f"$pf = Get-CimInstance Win32_PageFileSetting -Filter 'Name=\"C:\\\\pagefile.sys\"'; "
        f"if ($pf) {{ $pf.InitialSize = {recommended_min}; $pf.MaximumSize = {recommended_max}; $pf | Set-CimInstance }} "
        f"else {{ New-CimInstance -ClassName Win32_PageFileSetting -Property @{{Name='C:\\\\pagefile.sys';InitialSize={recommended_min};MaximumSize={recommended_max}}} }}"
    )

    recommendation = (
        f"Mevcut Pagefile: {current_mb:,} MB. "
        f"Önerilen: {recommended_min:,} MB – {recommended_max:,} MB. "
        f"Model için en az {model_size_mb:,} MB ek alan gerekiyor. "
        "Değişiklik için yönetici PowerShell'de aşağıdaki komutu çalıştırın."
        if current_mb < recommended_min else
        f"Pagefile yapılandırması optimal ({current_mb:,} MB). Herhangi bir işlem gerekmez."
    )

    return PagefileInfo(
        current_size_mb=current_mb,
        current_path=current_path,
        system_managed=system_managed,
        physical_ram_mb=ram_mb,
        recommended_min_mb=recommended_min,
        recommended_max_mb=recommended_max,
        recommendation=recommendation,
        powershell_command=ps_cmd,
        status=status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ② Service & Process Auditor
# ─────────────────────────────────────────────────────────────────────────────

def audit_services() -> List[ServiceInfo]:
    """
    Check status and memory usage of known heavy Windows services.
    Returns list of services with stop/disable commands (not executed here).
    """
    results: List[ServiceInfo] = []

    # Build a process name → memory map using psutil for RAM estimation
    proc_memory: Dict[str, float] = {}
    try:
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                name = proc.info["name"].lower() if proc.info["name"] else ""
                mem = proc.info["memory_info"].rss / (1024 * 1024) if proc.info["memory_info"] else 0
                proc_memory[name] = proc_memory.get(name, 0) + mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        logger.warning(f"Process enumeration error: {e}")

    for svc in KNOWN_HEAVY_SERVICES:
        svc_name = svc["name"]
        status = "unknown"
        mem_mb = 0.0

        # Try to get service status
        if IS_WINDOWS:
            try:
                result = subprocess.run(
                    ["powershell", "-NonInteractive", "-Command",
                     f"(Get-Service -Name '{svc_name}' -ErrorAction SilentlyContinue).Status"],
                    capture_output=True, text=True, timeout=5
                )
                raw = result.stdout.strip()
                if raw:
                    status = "running" if "Running" in raw else "stopped"
            except Exception:
                status = "unknown"
        else:
            status = "unavailable"

        # Estimate memory from known process names
        name_lower = svc_name.lower()
        for pname, pmem in proc_memory.items():
            if name_lower[:5] in pname:
                mem_mb = pmem
                break

        stop_cmd   = f"Stop-Service -Name '{svc_name}' -Force"
        disable_cmd = f"Set-Service -Name '{svc_name}' -StartupType Disabled"

        results.append(ServiceInfo(
            name=svc_name,
            display_name=svc["display_name"],
            status=status,
            memory_mb=round(mem_mb, 1),
            description=svc["description"],
            description_en=svc["description_en"],
            risk=svc["risk"],
            stop_command=stop_cmd,
            disable_command=disable_cmd,
        ))

    return results


def get_top_processes(limit: int = 10) -> List[ProcessInfo]:
    """Return the top N RAM-consuming processes."""
    procs: List[ProcessInfo] = []
    try:
        for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
            try:
                mem = proc.info["memory_info"].rss / (1024 * 1024)
                if mem > 10:   # Skip trivial processes
                    procs.append(ProcessInfo(
                        pid=proc.info["pid"],
                        name=proc.info["name"] or "unknown",
                        memory_mb=round(mem, 1),
                        cpu_percent=round(proc.info["cpu_percent"] or 0, 1),
                        kill_command=f"Stop-Process -Id {proc.info['pid']} -Force",
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        logger.warning(f"Top process error: {e}")

    procs.sort(key=lambda p: p.memory_mb, reverse=True)
    return procs[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# ③ RAM Disk Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ramdisk(model_size_mb: int = 0) -> RamDiskInfo:
    """
    Determine if a RAM disk is feasible for the given model size.
    Generates setup commands for ImDisk (Windows) or tmpfs (Linux).
    """
    ram = psutil.virtual_memory()
    total_mb   = ram.total    // (1024 * 1024)
    avail_mb   = ram.available // (1024 * 1024)

    # Conservative: keep 60% of available for OS + model compute, rest for disk
    safe_mb = int(avail_mb * 0.40)
    recommended_mb = min(safe_mb, model_size_mb) if model_size_mb > 0 else safe_mb

    # Detect ImDisk on Windows
    imdisk_installed = False
    ramdisk_drives: List[str] = []

    if IS_WINDOWS:
        try:
            result = subprocess.run(
                ["where", "imdisk"], capture_output=True, text=True, timeout=5
            )
            imdisk_installed = result.returncode == 0
        except Exception:
            pass

        # Detect existing ramdisk drives (ImDisk virtual disks)
        try:
            result = subprocess.run(
                ["powershell", "-NonInteractive", "-Command",
                 "Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Used -eq 0} | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                drives = [d.strip() + ":" for d in result.stdout.strip().split() if d.strip()]
                ramdisk_drives = drives
        except Exception:
            pass

    # Build setup steps
    if IS_WINDOWS:
        if imdisk_installed:
            setup_steps = [
                "ImDisk zaten yüklü ✓",
                f"Adım 1: {recommended_mb:,} MB RAM Disk oluştur",
                "Adım 2: Model dosyasını (.gguf) RAM Disk'e kopyala",
                "Adım 3: AI Runner'da model yolunu RAM Disk'e güncelle",
                "⚠️ Bilgisayar kapanırsa RAM Disk içeriği silinir — .gguf dosyasını HDD/SSD'de de sakla",
            ]
            ps_cmd = (
                f"imdisk -a -s {recommended_mb}M -m R: -p \"/fs:ntfs /q /y\" && "
                f"echo 'RAM Disk R: hazır ({recommended_mb} MB)'"
            )
        else:
            setup_steps = [
                "Adım 1: ImDisk Toolkit'i indir — https://sourceforge.net/projects/imdisk-toolkit/",
                "Adım 2: Yönetici olarak kurulumu çalıştır",
                "Adım 3: Aşağıdaki PowerShell komutunu çalıştır",
                f"Adım 4: Model dosyasını (.gguf) R: sürücüsüne kopyala",
                "Adım 5: AI Runner'da model yolunu 'R:\\model.gguf' olarak ayarla",
            ]
            ps_cmd = (
                f"# ImDisk kurulduktan sonra:\n"
                f"imdisk -a -s {recommended_mb}M -m R: -p \"/fs:ntfs /q /y\""
            )
    else:
        setup_steps = [
            f"Adım 1: tmpfs RAM Disk oluştur ({recommended_mb} MB)",
            "Adım 2: Model dosyasını (.gguf) /mnt/ramdisk klasörüne kopyala",
            "Adım 3: AI Runner'da model yolunu /mnt/ramdisk/model.gguf olarak ayarla",
        ]
        ps_cmd = (
            f"sudo mkdir -p /mnt/ramdisk && "
            f"sudo mount -t tmpfs -o size={recommended_mb}m tmpfs /mnt/ramdisk && "
            f"echo 'RAM Disk /mnt/ramdisk hazır ({recommended_mb} MB)'"
        )

    if avail_mb < 2048:
        status = "insufficient"
    elif ramdisk_drives:
        status = "already_exists"
    else:
        status = "ready"

    return RamDiskInfo(
        physical_ram_mb=total_mb,
        available_ram_mb=avail_mb,
        safe_ramdisk_mb=safe_mb,
        recommended_ramdisk_mb=recommended_mb,
        imdisk_installed=imdisk_installed,
        ramdisk_drives=ramdisk_drives,
        setup_steps=setup_steps,
        powershell_command=ps_cmd,
        status=status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Prompt Pruning Budget
# ─────────────────────────────────────────────────────────────────────────────

def calculate_prompt_budget(
    context_length: int,
    history_messages: List[Dict[str, str]],
    system_prompt: str = "",
    avg_tokens_per_char: float = 0.25,  # ~4 chars per token
) -> Dict[str, Any]:
    """
    Calculate how many tokens the current conversation uses and
    how close it is to the context limit.
    """
    system_tokens = int(len(system_prompt) * avg_tokens_per_char)
    history_tokens = sum(
        int(len(m.get("content", "")) * avg_tokens_per_char)
        for m in history_messages
    )
    total_used = system_tokens + history_tokens
    remaining = max(0, context_length - total_used)
    utilization = min(1.0, total_used / max(1, context_length))

    return {
        "context_length": context_length,
        "system_tokens": system_tokens,
        "history_tokens": history_tokens,
        "total_used": total_used,
        "remaining": remaining,
        "utilization_pct": round(utilization * 100, 1),
        "is_warning": utilization >= 0.80,
        "is_critical": utilization >= 0.95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overall Status
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizer_status() -> SystemOptimizerStatus:
    """Return a high-level system optimization scorecard."""
    ram = psutil.virtual_memory()
    total_mb = ram.total // (1024 * 1024)
    avail_mb  = ram.available // (1024 * 1024)

    score = 100
    recommendations: List[str] = []

    # Pagefile check
    pagefile = analyze_pagefile()
    if pagefile.status == "low":
        score -= 20
        recommendations.append(f"Pagefile çok küçük ({pagefile.current_size_mb:,} MB). En az {pagefile.recommended_min_mb:,} MB önerilir.")
    elif pagefile.status == "critical":
        score -= 35
        recommendations.append("Pagefile kritik seviyede küçük! Sistem optimizasyonu yapın.")

    # Memory pressure check
    used_pct = ram.percent
    if used_pct > 85:
        score -= 20
        recommendations.append(f"RAM kullanımı yüksek (%{used_pct:.0f}). Arka plan servisleri kapatmayı deneyin.")
    elif used_pct > 70:
        score -= 10
        recommendations.append(f"RAM kullanımı orta (%{used_pct:.0f}). Gereksiz uygulamaları kapatın.")

    # Top process
    top_procs = get_top_processes(limit=1)
    top_name = top_procs[0].name if top_procs else "N/A"

    if not recommendations:
        recommendations.append("Sistem durumu iyi. Ek optimizasyon gerekmez.")

    return SystemOptimizerStatus(
        os_name=f"{platform.system()} {platform.release()}",
        total_ram_mb=total_mb,
        available_ram_mb=avail_mb,
        pagefile_status=pagefile.status,
        top_memory_process=top_name,
        optimization_score=max(0, score),
        recommendations=recommendations,
    )
