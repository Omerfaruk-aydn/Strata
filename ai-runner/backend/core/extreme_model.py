"""Hardware-aware feasibility analysis for very large GGUF models.

The planner distinguishes three real storage domains:

* GPU-resident weights selected through llama.cpp ``n_gpu_layers``.
* CPU/RAM-resident weights handled by llama.cpp.
* File-backed pages provided by ``mmap`` when the physical working set is
  smaller than the model.  This is reported as storage pressure, not as a
  fictional third llama.cpp layer-offload target.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .hardware_profile import HardwareProfile
from .memory_manager import QUANT_BPW, estimate_model_size_mb, estimate_total_layers
from .model_loader import GGUFMetadata, validate_gguf_file
from .runtime_capabilities import RuntimeCapabilities


PresetName = Literal["safe", "balanced", "performance", "maximum_capacity"]
FeasibilityStatus = Literal["ideal", "ready", "constrained", "blocked"]


class PlannerPreset(BaseModel):
    name: PresetName
    vram_fraction: float
    ram_fraction: float
    reserve_vram_mb: int
    reserve_ram_mb: int
    n_batch: int
    context_cap: int
    kv_cache_type: str
    description: str


PRESETS: Dict[PresetName, PlannerPreset] = {
    "safe": PlannerPreset(
        name="safe", vram_fraction=0.74, ram_fraction=0.70,
        reserve_vram_mb=1536, reserve_ram_mb=4096, n_batch=128,
        context_cap=2048, kv_cache_type="q4_0",
        description="Largest safety margin and conservative buffers.",
    ),
    "balanced": PlannerPreset(
        name="balanced", vram_fraction=0.82, ram_fraction=0.78,
        reserve_vram_mb=1024, reserve_ram_mb=3072, n_batch=256,
        context_cap=4096, kv_cache_type="q4_0",
        description="Balanced reliability, speed, and context capacity.",
    ),
    "performance": PlannerPreset(
        name="performance", vram_fraction=0.89, ram_fraction=0.84,
        reserve_vram_mb=768, reserve_ram_mb=2048, n_batch=512,
        context_cap=8192, kv_cache_type="q8_0",
        description="Higher throughput when the model already fits comfortably.",
    ),
    "maximum_capacity": PlannerPreset(
        name="maximum_capacity", vram_fraction=0.86, ram_fraction=0.90,
        reserve_vram_mb=896, reserve_ram_mb=2048, n_batch=64,
        context_cap=2048, kv_cache_type="q4_0",
        description="Prioritizes loading very large models over prompt throughput.",
    ),
}


KV_BYTES = {
    "q4_0": 0.5625,
    "q5_0": 0.6875,
    "q5_1": 0.6875,
    "q8_0": 1.0625,
    "f16": 2.0,
}


class ModelSpecification(BaseModel):
    model_id: str
    local_path: Optional[str] = None
    quant: str = "Q4_K_M"
    file_size_mb: float = Field(gt=0)
    parameter_count: int = 0
    architecture: str = ""
    total_layers: int = Field(gt=0)
    native_context_length: int = 4096
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    metadata_source: Literal["gguf", "estimate"] = "estimate"


class MemoryBudget(BaseModel):
    model_weights_mb: float
    gpu_weights_mb: float
    cpu_weights_mb: float
    kv_cache_mb: float
    compute_buffer_mb: float
    vram_reserve_mb: float
    ram_reserve_mb: float
    estimated_vram_usage_mb: float
    estimated_ram_working_set_mb: float
    physical_ram_shortfall_mb: float
    pagefile_free_mb: float
    disk_free_mb: float
    storage_mode: Literal["gpu_resident", "ram_resident", "memory_mapped", "insufficient"]


class RuntimeRecommendation(BaseModel):
    backend: str
    n_gpu_layers: int
    cpu_layers: int
    context_length: int
    n_batch: int
    n_threads: int
    kv_cache_type: str
    use_mmap: bool
    use_mlock: bool
    flash_attn: bool
    speculative_decoding: bool
    selected_gpu_index: int
    tensor_split: Optional[List[float]] = None
    adaptive_retry: bool = True
    max_load_attempts: int = 6


class QuantCandidate(BaseModel):
    quant: str
    estimated_size_mb: float
    physical_memory_fit: bool
    requires_mmap_pressure: bool
    quality_rank: float
    recommendation: Literal["recommended", "possible", "not_recommended"]


class FeasibilityReport(BaseModel):
    analysis_version: int = 1
    status: FeasibilityStatus
    status_label: str
    model: ModelSpecification
    preset: PlannerPreset
    memory: MemoryBudget
    runtime: RuntimeRecommendation
    gpu_layer_ratio: float
    estimated_tokens_per_second_min: float
    estimated_tokens_per_second_max: float
    quant_candidates: List[QuantCandidate] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    hardware_fingerprint: str
    disclaimer: str = (
        "Memory and speed values are conservative planning estimates. "
        "Run the built-in benchmark after a successful load for measured performance."
    )


def hardware_fingerprint(hardware: HardwareProfile, backend: str) -> str:
    cpu = getattr(hardware, "cpu", None)
    ram = getattr(hardware, "ram", None)
    gpus = getattr(hardware, "gpus", []) or []
    payload = "|".join([
        str(getattr(cpu, "name", "unknown")),
        str(getattr(cpu, "cores", 0)),
        str(getattr(ram, "total_mb", 0)),
        ",".join(
            f"{getattr(gpu, 'name', 'unknown')}:{getattr(gpu, 'vram_total_mb', 0)}"
            for gpu in gpus
        ),
        str(getattr(hardware, "os_info", "unknown")),
        backend,
    ])
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:20]


def model_path_fingerprint(path: str) -> str:
    """Fast identity for profiles without hashing multi-gigabyte model weights."""
    try:
        stat = os.stat(path)
        payload = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        payload = os.path.abspath(path)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:20]


def specification_from_gguf(
    model_id: str,
    local_path: str,
    quant: str,
    parameter_count: int = 0,
) -> tuple[ModelSpecification, GGUFMetadata]:
    metadata = validate_gguf_file(local_path)
    if not metadata.is_valid:
        raise ValueError(metadata.error or "Invalid GGUF file")
    resolved_params = metadata.parameter_count or parameter_count
    total_layers = metadata.block_count or estimate_total_layers(resolved_params or 7_000_000_000)
    return ModelSpecification(
        model_id=model_id,
        local_path=local_path,
        quant=quant.upper(),
        file_size_mb=metadata.file_size_bytes / (1024 * 1024),
        parameter_count=resolved_params,
        architecture=metadata.architecture,
        total_layers=total_layers,
        native_context_length=metadata.context_length,
        embedding_length=metadata.embedding_length,
        head_count=metadata.head_count,
        head_count_kv=metadata.head_count_kv,
        metadata_source="gguf",
    ), metadata


def estimated_specification(
    model_id: str,
    quant: str,
    parameter_count: int,
    *,
    file_size_mb: Optional[float] = None,
    total_layers: Optional[int] = None,
    context_length: int = 4096,
) -> ModelSpecification:
    if parameter_count <= 0 and not file_size_mb:
        raise ValueError("parameter_count or file_size_mb is required")
    normalized_quant = quant.upper()
    if file_size_mb is None and normalized_quant not in QUANT_BPW:
        raise ValueError(
            f"Unsupported quantization for size estimation: {normalized_quant}. "
            "Provide an explicit file_size_mb or choose a known GGUF quantization."
        )
    estimated_size = file_size_mb or estimate_model_size_mb(parameter_count, normalized_quant)
    return ModelSpecification(
        model_id=model_id,
        quant=normalized_quant,
        file_size_mb=estimated_size,
        parameter_count=parameter_count,
        total_layers=total_layers or estimate_total_layers(parameter_count or 7_000_000_000),
        native_context_length=context_length,
        metadata_source="estimate",
    )


def estimate_kv_cache_for_spec(
    spec: ModelSpecification,
    context_length: int,
    kv_cache_type: str,
) -> float:
    heads = spec.head_count or 32
    kv_heads = spec.head_count_kv or max(1, heads // 4)
    embedding = spec.embedding_length or heads * 128
    head_dim = max(1, embedding // heads)
    bytes_per_element = KV_BYTES.get(kv_cache_type, KV_BYTES["q4_0"])
    kv_bytes = 2 * spec.total_layers * context_length * kv_heads * head_dim * bytes_per_element
    return max(32.0, kv_bytes / (1024 * 1024))


def _quant_candidates(
    parameter_count: int,
    hardware: HardwareProfile,
    recommended_quant: str,
) -> List[QuantCandidate]:
    if parameter_count <= 0:
        return []
    physical_budget = hardware.ram.free_mb * 0.82 + hardware.gpu.vram_free_mb * 0.82
    candidates: List[QuantCandidate] = []
    for quant in ("IQ1_S", "IQ2_XXS", "IQ2_XS", "Q2_K", "IQ3_XS", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"):
        if quant not in QUANT_BPW:
            continue
        size = estimate_model_size_mb(parameter_count, quant)
        physical_fit = size + 2048 <= physical_budget
        mapped = not physical_fit and size <= physical_budget + hardware.ram.total_mb * 0.35
        if quant == recommended_quant:
            recommendation = "recommended"
        elif physical_fit or mapped:
            recommendation = "possible"
        else:
            recommendation = "not_recommended"
        candidates.append(QuantCandidate(
            quant=quant,
            estimated_size_mb=round(size, 1),
            physical_memory_fit=physical_fit,
            requires_mmap_pressure=mapped,
            quality_rank=QUANT_BPW[quant],
            recommendation=recommendation,
        ))
    return candidates


def analyze_feasibility(
    spec: ModelSpecification,
    hardware: HardwareProfile,
    capabilities: RuntimeCapabilities,
    *,
    preset_name: PresetName = "maximum_capacity",
    requested_context_length: int = 2048,
    selected_gpu_index: int = 0,
    tensor_split: Optional[List[float]] = None,
    force_cpu: bool = False,
) -> FeasibilityReport:
    preset = PRESETS[preset_name]
    blockers: List[str] = []
    warnings: List[str] = []
    actions: List[str] = []

    if not capabilities.llama_cpp_installed:
        blockers.append("A compatible llama-cpp-python runtime is not installed.")
    if selected_gpu_index < 0 or (hardware.gpus and selected_gpu_index >= len(hardware.gpus)):
        blockers.append("The selected GPU does not exist.")

    context_length = max(512, min(requested_context_length, preset.context_cap, spec.native_context_length or requested_context_length))
    kv_cache_mb = estimate_kv_cache_for_spec(spec, context_length, preset.kv_cache_type)
    compute_buffer_mb = max(384.0, min(2048.0, 320.0 + preset.n_batch * 1.5 + spec.embedding_length / 16.0))

    gpu_devices = list(hardware.gpus or [hardware.gpu])
    gpu_available = not force_cpu and capabilities.gpu_offload_supported and any(
        gpu.vram_free_mb > 0 for gpu in gpu_devices
    )
    runtime_vram = kv_cache_mb + compute_buffer_mb if gpu_available else 0.0
    per_gpu_reserves = [
        max(float(preset.reserve_vram_mb), gpu.vram_total_mb * (1 - preset.vram_fraction))
        for gpu in gpu_devices
    ]
    layer_capacities = [
        max(0.0, gpu.vram_free_mb - reserve)
        for gpu, reserve in zip(gpu_devices, per_gpu_reserves)
    ]
    main_gpu_position = selected_gpu_index if 0 <= selected_gpu_index < len(gpu_devices) else 0
    if gpu_available and layer_capacities:
        layer_capacities[main_gpu_position] = max(
            0.0,
            layer_capacities[main_gpu_position] - runtime_vram,
        )

    normalized_split: Optional[List[float]] = None
    participating_indices = [main_gpu_position]
    if tensor_split is not None:
        if len(tensor_split) != len(gpu_devices) or any(part <= 0 for part in tensor_split):
            blockers.append("Tensor split must contain one positive proportion for every detected GPU.")
        else:
            split_total = sum(tensor_split)
            normalized_split = [round(part / split_total, 6) for part in tensor_split]
            participating_indices = list(range(len(gpu_devices)))
    elif len(gpu_devices) > 1 and all(capacity > 0 for capacity in layer_capacities):
        capacity_total = sum(layer_capacities)
        normalized_split = [round(capacity / capacity_total, 6) for capacity in layer_capacities]
        participating_indices = list(range(len(gpu_devices)))
    elif len(gpu_devices) > 1 and gpu_available:
        warnings.append(
            "Automatic multi-GPU splitting was skipped because at least one GPU has no safe VRAM headroom."
        )

    if not gpu_available:
        layer_vram = 0.0
        vram_reserve = 0.0
        normalized_split = None
    elif normalized_split:
        # Respect explicit split ratios: total offload is capped by whichever
        # participating GPU reaches its safe capacity first.
        layer_vram = min(
            layer_capacities[index] / normalized_split[index]
            for index in participating_indices
        )
        vram_reserve = sum(per_gpu_reserves[index] for index in participating_indices)
    else:
        layer_vram = layer_capacities[main_gpu_position]
        vram_reserve = per_gpu_reserves[main_gpu_position]

    layer_size = spec.file_size_mb / spec.total_layers
    gpu_layers = min(spec.total_layers, int(layer_vram // layer_size)) if gpu_available and layer_size > 0 else 0
    gpu_weights = min(spec.file_size_mb, gpu_layers * layer_size)
    cpu_weights = max(0.0, spec.file_size_mb - gpu_weights)

    ram_reserve = max(float(preset.reserve_ram_mb), hardware.ram.total_mb * (1 - preset.ram_fraction))
    host_kv = 0.0 if gpu_layers > 0 else kv_cache_mb
    host_compute = max(256.0, compute_buffer_mb * 0.35)
    estimated_ram = cpu_weights + host_kv + host_compute
    usable_ram = max(0.0, hardware.ram.free_mb - ram_reserve)
    shortfall = max(0.0, estimated_ram - usable_ram)
    disk_free_mb = hardware.disk.free_gb * 1024

    if spec.metadata_source == "estimate":
        # A pre-download plan must also prove that the GGUF can be stored.
        # Local GGUF analysis does not require this check because its file is
        # already present on the selected model volume.
        required_download_mb = spec.file_size_mb * 1.05
        if disk_free_mb < required_download_mb:
            missing_disk_mb = required_download_mb - disk_free_mb
            blockers.append(
                "The model volume does not have enough free space for the estimated GGUF "
                f"(approximately {missing_disk_mb / 1024:.1f} GB more is required)."
            )
            actions.append("Free disk space or move the model directory to a larger SSD/NVMe volume.")

    if gpu_layers == spec.total_layers:
        storage_mode = "gpu_resident"
    elif shortfall <= 0:
        storage_mode = "ram_resident"
    elif hardware.disk.type != "HDD" and disk_free_mb >= max(1024.0, shortfall * 0.25):
        storage_mode = "memory_mapped"
        warnings.append(
            f"The physical RAM working set is short by approximately {shortfall / 1024:.1f} GB. "
            "File-backed mmap paging will be used and may reduce throughput sharply."
        )
    else:
        storage_mode = "insufficient"
        blockers.append("The model working set does not fit available memory and safe mapped execution is unavailable.")

    if hardware.disk.type == "HDD" and shortfall > 0:
        blockers.append("A spinning HDD is not suitable for a memory-constrained model working set; use an SSD/NVMe drive.")
    if spec.metadata_source == "estimate":
        warnings.append("Model architecture metadata is estimated; analyze the downloaded GGUF for a final plan.")
    if (
        gpu_layers == 0
        and hardware.gpus
        and not capabilities.gpu_offload_supported
        and not force_cpu
    ):
        warnings.append("A GPU is installed, but the active llama.cpp build cannot offload layers to it.")
        actions.append("Install a GPU-enabled llama.cpp runtime that matches the installed driver and restart AI Runner.")
    if context_length < requested_context_length:
        warnings.append(f"Context was reduced to {context_length} tokens by the selected capacity preset.")
    if shortfall > 0:
        actions.append("Close memory-heavy applications and keep the model on the fastest available SSD/NVMe drive.")
    if spec.parameter_count >= 80_000_000_000:
        actions.append("Run the measured benchmark after loading; large-model speed cannot be predicted reliably from VRAM alone.")

    if blockers:
        status: FeasibilityStatus = "blocked"
        status_label = "Blocked"
    elif gpu_layers == spec.total_layers:
        status = "ideal"
        status_label = "Fits on GPU"
    elif storage_mode == "ram_resident":
        status = "ready"
        status_label = "Ready with GPU/CPU offload"
    else:
        status = "constrained"
        status_label = "Runs with mapped-memory pressure"

    gpu_ratio = gpu_layers / max(spec.total_layers, 1)
    if status == "ideal":
        speed_min, speed_max = 12.0, 45.0
    elif storage_mode == "ram_resident":
        speed_min = max(0.2, 0.35 + gpu_ratio * 2.5)
        speed_max = max(0.6, 1.2 + gpu_ratio * 7.0)
    elif status == "constrained":
        speed_min, speed_max = 0.05, max(0.2, 0.5 + gpu_ratio)
    else:
        speed_min, speed_max = 0.0, 0.0

    active_backend = "cpu" if force_cpu else capabilities.active_backend
    if active_backend == "unknown" and capabilities.gpu_offload_supported:
        active_backend = "unknown"
    n_threads = max(1, hardware.cpu.cores or hardware.cpu.threads or 1)
    use_mlock = shortfall <= 0 and estimated_ram < hardware.ram.total_mb * 0.55
    if not use_mlock:
        actions.append("Memory locking is disabled for this plan so the OS can manage the large mapped working set.")

    memory = MemoryBudget(
        model_weights_mb=round(spec.file_size_mb, 1),
        gpu_weights_mb=round(gpu_weights, 1),
        cpu_weights_mb=round(cpu_weights, 1),
        kv_cache_mb=round(kv_cache_mb, 1),
        compute_buffer_mb=round(compute_buffer_mb, 1),
        vram_reserve_mb=round(vram_reserve, 1),
        ram_reserve_mb=round(ram_reserve, 1),
        estimated_vram_usage_mb=round(gpu_weights + runtime_vram, 1),
        estimated_ram_working_set_mb=round(estimated_ram, 1),
        physical_ram_shortfall_mb=round(shortfall, 1),
        pagefile_free_mb=float(hardware.virtual_memory.free_mb),
        disk_free_mb=round(disk_free_mb, 1),
        storage_mode=storage_mode,
    )
    runtime = RuntimeRecommendation(
        backend="cpu" if gpu_layers == 0 else str(active_backend),
        n_gpu_layers=gpu_layers,
        cpu_layers=spec.total_layers - gpu_layers,
        context_length=context_length,
        n_batch=preset.n_batch,
        n_threads=n_threads,
        kv_cache_type=preset.kv_cache_type,
        use_mmap=True,
        use_mlock=use_mlock,
        flash_attn=gpu_layers > 0,
        speculative_decoding=False,
        selected_gpu_index=max(0, selected_gpu_index),
        tensor_split=normalized_split,
    )
    return FeasibilityReport(
        status=status,
        status_label=status_label,
        model=spec,
        preset=preset,
        memory=memory,
        runtime=runtime,
        gpu_layer_ratio=round(gpu_ratio, 4),
        estimated_tokens_per_second_min=round(speed_min, 2),
        estimated_tokens_per_second_max=round(speed_max, 2),
        quant_candidates=_quant_candidates(spec.parameter_count, hardware, spec.quant),
        blockers=list(dict.fromkeys(blockers)),
        warnings=list(dict.fromkeys(warnings)),
        actions=list(dict.fromkeys(actions)),
        hardware_fingerprint=hardware_fingerprint(hardware, str(active_backend)),
    )
