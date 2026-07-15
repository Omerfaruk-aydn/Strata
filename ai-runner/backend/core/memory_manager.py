"""
AI Runner — Memory Manager / Offload Planner
Implements the 6-step optimization algorithm from Section 11.
Calculates n_gpu_layers, RAM layers, disk layers for any model+hardware combo.
"""

import math
from typing import Optional, List
from pydantic import BaseModel
from .hardware_profile import HardwareProfile


# ── Data Models ──

class OffloadPlan(BaseModel):
    model_id: str
    quant: str
    total_layers: int
    gpu_layers: int
    cpu_layers: int
    disk_streamed_layers: int
    estimated_tokens_per_sec: float
    context_length_recommended: int
    fits_comfortably: bool
    warnings: List[str] = []
    recommendation: str = ""

    # Detailed breakdown
    vram_usage_mb: float = 0.0
    ram_usage_mb: float = 0.0
    kv_cache_mb: float = 0.0


# ── Constants ──

# Safety margin — reserve 15% for OS/other apps (Section 11, Step 1)
VRAM_SAFETY_FACTOR = 0.85
RAM_SAFETY_FACTOR = 0.80

# KV-cache size estimation constants
# Approximate bytes per token per layer for KV-cache
# For Q4_K_M with GQA: ~0.5 MB per 1K context per layer (rough estimate)
KV_CACHE_BYTES_PER_TOKEN_PER_LAYER = 256  # bytes, conservative estimate

# Base speed estimates (tokens/sec) for different configurations
# These are rough baselines that get adjusted by hardware
BASE_SPEED_ALL_GPU = 30.0      # All layers on GPU
BASE_SPEED_GPU_RAM = 8.0       # Mixed GPU+RAM
BASE_SPEED_WITH_DISK = 2.0     # Any disk streaming

# Disk streaming penalty factor
DISK_STREAMING_PENALTY = 0.4

# Minimum recommended context length
MIN_CONTEXT_LENGTH = 512
DEFAULT_CONTEXT_LENGTH = 4096


# ── Quantization Size Multipliers ──
# Approximate bytes per parameter for each quant level.
# IQ (Importance Matrix) types use fractional bits — same BPW as name implies
# but with significantly better quality due to importance-aware quantization.
QUANT_BPW = {
    # ── I-Matrix Ultra-Compression (IQ series) ──────────────────────────────
    "IQ1_S":   1.56,  # ~1.5-bit effective — 70B model fits in ~10 GB
    "IQ2_XXS": 2.06,  # ~2-bit — 50B fits ~14 GB, far better quality than Q2_K
    "IQ2_XS":  2.31,  # ~2.3-bit — slightly larger than IQ2_XXS, better quality
    "IQ3_XS":  3.3,   # ~3.3-bit — better than Q3_K_M at smaller size
    "IQ3_S":   3.5,   # ~3.5-bit
    "IQ4_XS":  4.25,  # ~4.25-bit — close to Q4_K_M but smaller
    "IQ4_NL":  4.5,   # ~4.5-bit non-linear
    # ── Standard K-Quant Series ─────────────────────────────────────────────
    "Q2_K":   2.6,
    "Q3_K_S": 3.4,
    "Q3_K_M": 3.9,
    "Q3_K_L": 4.3,
    "Q4_0":   4.5,
    "Q4_K_S": 4.6,
    "Q4_K_M": 4.8,
    "Q5_0":   5.5,
    "Q5_K_S": 5.5,
    "Q5_K_M": 5.7,
    "Q6_K":   6.6,
    "Q8_0":   8.5,
    "F16":   16.0,
    "F32":   32.0,
}



def estimate_model_size_mb(
    parameter_count: int,
    quant: str
) -> float:
    """Estimate model file size in MB for a given parameter count and quantization."""
    bpw = QUANT_BPW.get(quant, 4.8)  # Default to Q4_K_M if unknown
    # Convert: params * bits_per_weight / 8 bits_per_byte / 1024^2 bytes_per_mb
    size_bytes = parameter_count * bpw / 8
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def estimate_total_layers(parameter_count: int) -> int:
    """
    Estimate total transformer layers based on parameter count.
    This is a rough heuristic — real layer count comes from model metadata.
    """
    if parameter_count <= 1_000_000_000:      # ≤1B
        return 24
    elif parameter_count <= 3_000_000_000:    # ≤3B
        return 26
    elif parameter_count <= 7_000_000_000:    # ≤7B
        return 32
    elif parameter_count <= 13_000_000_000:   # ≤13B
        return 40
    elif parameter_count <= 34_000_000_000:   # ≤34B
        return 48
    elif parameter_count <= 70_000_000_000:   # ≤70B
        return 80
    else:  # 100B+
        return 96


def estimate_kv_cache_mb(
    context_length: int,
    total_layers: int,
    n_heads: int = 32,
    head_dim: int = 128,
    n_kv_heads: Optional[int] = None,
) -> float:
    """
    Estimate KV-cache memory in MB.
    KV-cache = 2 * n_layers * context_length * n_kv_heads * head_dim * sizeof(float16)
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads  # No GQA assumption

    # 2 for K and V, 2 bytes for float16
    kv_bytes = 2 * total_layers * context_length * n_kv_heads * head_dim * 2
    return kv_bytes / (1024 * 1024)


def calculate_offload_plan(
    model_id: str,
    quant: str,
    file_size_mb: float,
    total_layers: int,
    hardware: HardwareProfile,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    parameter_count: Optional[int] = None,
    user_gpu_layers: Optional[int] = None,
) -> OffloadPlan:
    """
    Main offload planning algorithm — implements Section 11 (6 steps).

    Args:
        model_id: Model identifier
        quant: Quantization level (e.g., "Q4_K_M")
        file_size_mb: Model file size in MB
        total_layers: Total transformer layers in the model
        hardware: Current hardware profile
        context_length: Desired context window size
        parameter_count: Number of parameters (for KV-cache estimation)
        user_gpu_layers: Manual override for n_gpu_layers (FR-301)

    Returns:
        OffloadPlan with layer distribution and performance estimates
    """
    warnings = []

    # ── Step 1: Calculate usable VRAM ──
    vram_free = hardware.gpu.vram_free_mb
    usable_vram = vram_free * VRAM_SAFETY_FACTOR

    # ── Step 2: Layer size estimation ──
    layer_size_mb = file_size_mb / total_layers if total_layers > 0 else file_size_mb

    # ── KV-cache estimation ──
    kv_cache_mb = estimate_kv_cache_mb(
        context_length=context_length,
        total_layers=total_layers,
    )

    # Reserve VRAM for KV-cache
    vram_for_layers = max(0, usable_vram - kv_cache_mb)

    # ── Step 3: GPU layers ──
    if user_gpu_layers is not None:
        # FR-301: Manual override
        gpu_layers = max(0, min(user_gpu_layers, total_layers))
        if gpu_layers * layer_size_mb > usable_vram:
            warnings.append(
                f"Manuel ayar ({gpu_layers} katman) kullanılabilir VRAM'i aşabilir. "
                f"OOM riski var."
            )
    else:
        # Automatic calculation
        if layer_size_mb > 0:
            gpu_layers = min(
                int(math.floor(vram_for_layers / layer_size_mb)),
                total_layers
            )
        else:
            gpu_layers = 0

    # ── Step 4: RAM and disk distribution ──
    remaining = total_layers - gpu_layers

    usable_ram = hardware.ram.free_mb * RAM_SAFETY_FACTOR
    # Reserve RAM for KV-cache portions not on GPU
    ram_for_layers = max(0, usable_ram - (kv_cache_mb * 0.3))  # Partial KV on RAM

    if remaining > 0 and layer_size_mb > 0:
        ram_layers = min(
            remaining,
            int(math.floor(ram_for_layers / layer_size_mb))
        )
    else:
        ram_layers = 0

    disk_layers = remaining - ram_layers

    # ── Disk streaming warnings ──
    if disk_layers > 0:
        if hardware.disk.type == "HDD":
            warnings.append(
                "Disk streaming aktif — HDD tespit edildi. "
                "SSD kullanılması önerilir, aksi halde performans ciddi düşer."
            )
        else:
            warnings.append(
                f"Disk streaming aktif — SSD tespit edildi ✓ "
                f"({disk_layers} katman diskten okunacak)"
            )

    # ── Step 5: Speed estimation ──
    estimated_speed = _estimate_speed(
        gpu_layers=gpu_layers,
        ram_layers=ram_layers,
        disk_layers=disk_layers,
        total_layers=total_layers,
        vram_total=hardware.gpu.vram_total_mb,
        is_hdd=hardware.disk.type == "HDD",
    )

    # ── Context length recommendation ──
    if gpu_layers == total_layers:
        # All on GPU — can support full context
        context_recommended = context_length
    elif disk_layers > 0:
        # Disk streaming — reduce context to save memory
        context_recommended = min(context_length, 2048)
        if context_length > 2048:
            warnings.append(
                f"Disk streaming nedeniyle context penceresi {context_recommended} "
                f"olarak önerilir (istenen: {context_length})."
            )
    else:
        context_recommended = min(context_length, DEFAULT_CONTEXT_LENGTH)

    # ── Comfort assessment ──
    fits_comfortably = (
        disk_layers == 0 and
        gpu_layers >= total_layers * 0.3 and
        len([w for w in warnings if "OOM" in w]) == 0
    )

    # ── VRAM/RAM usage estimates ──
    vram_usage = gpu_layers * layer_size_mb + kv_cache_mb * 0.7
    ram_usage = ram_layers * layer_size_mb + kv_cache_mb * 0.3

    # ── Recommendation text ──
    recommendation = _generate_recommendation(
        quant=quant,
        gpu_layers=gpu_layers,
        ram_layers=ram_layers,
        disk_layers=disk_layers,
        total_layers=total_layers,
        estimated_speed=estimated_speed,
        hardware=hardware,
    )

    return OffloadPlan(
        model_id=model_id,
        quant=quant,
        total_layers=total_layers,
        gpu_layers=gpu_layers,
        cpu_layers=ram_layers,
        disk_streamed_layers=disk_layers,
        estimated_tokens_per_sec=round(estimated_speed, 1),
        context_length_recommended=context_recommended,
        fits_comfortably=fits_comfortably,
        warnings=warnings,
        recommendation=recommendation,
        vram_usage_mb=round(vram_usage, 1),
        ram_usage_mb=round(ram_usage, 1),
        kv_cache_mb=round(kv_cache_mb, 1),
    )


def _estimate_speed(
    gpu_layers: int,
    ram_layers: int,
    disk_layers: int,
    total_layers: int,
    vram_total: int,
    is_hdd: bool,
) -> float:
    """
    Step 5 from Section 11: Rough speed estimation.
    GPU ratio increases token/s logarithmically.
    Disk streaming adds severe penalty.
    """
    if total_layers == 0:
        return 0.0

    gpu_ratio = gpu_layers / total_layers

    if disk_layers > 0:
        # Disk streaming active — heavy penalty
        base = BASE_SPEED_WITH_DISK
        disk_penalty = DISK_STREAMING_PENALTY if not is_hdd else DISK_STREAMING_PENALTY * 0.3
        speed = base * (1 + gpu_ratio) * disk_penalty
    elif ram_layers > 0:
        # Mixed GPU + RAM
        speed = BASE_SPEED_GPU_RAM * (1 + math.log2(1 + gpu_ratio * 3))
    else:
        # All GPU — best case
        speed = BASE_SPEED_ALL_GPU

    # Scale by VRAM tier (rough proxy for GPU power)
    if vram_total >= 24000:
        speed *= 1.5
    elif vram_total >= 16000:
        speed *= 1.2
    elif vram_total >= 8000:
        speed *= 1.0
    elif vram_total >= 4000:
        speed *= 0.7
    else:
        speed *= 0.5

    return max(0.1, speed)


def _generate_recommendation(
    quant: str,
    gpu_layers: int,
    ram_layers: int,
    disk_layers: int,
    total_layers: int,
    estimated_speed: float,
    hardware: HardwareProfile,
) -> str:
    """Generate a human-readable recommendation string."""
    parts = []

    if gpu_layers == total_layers:
        parts.append(
            f"{quant} seçildi — tüm {total_layers} katman GPU'da çalışacak. "
            f"Tahmini hız: ~{estimated_speed:.1f} token/sn."
        )
    elif disk_layers > 0:
        parts.append(
            f"{quant} ile kısmi disk streaming gerekir "
            f"({disk_layers} katman diskten okunacak). "
            f"Tahmini hız: ~{estimated_speed:.1f} token/sn."
        )
        # Suggest a lower quant
        lower_quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        current_idx = next(
            (i for i, q in enumerate(lower_quants) if q == quant), -1
        )
        if current_idx > 0:
            suggested = lower_quants[current_idx - 1]
            parts.append(
                f"{suggested} seçilirse disk streaming gerekmeyebilir."
            )
    else:
        parts.append(
            f"{quant} seçildi — {gpu_layers} katman GPU'da, "
            f"{ram_layers} katman RAM'de çalışacak. "
            f"Tahmini hız: ~{estimated_speed:.1f} token/sn."
        )

    return " ".join(parts)


def suggest_best_quant(
    parameter_count: int,
    total_layers: int,
    available_quants: List[str],
    hardware: HardwareProfile,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> dict:
    """
    Given a model and hardware, suggest the best quantization level.
    Returns the quant that provides the best quality without disk streaming.
    Implements Section 12 logic.
    """
    results = []

    for quant in available_quants:
        if quant not in QUANT_BPW:
            continue

        file_size_mb = estimate_model_size_mb(parameter_count, quant)
        plan = calculate_offload_plan(
            model_id="evaluation",
            quant=quant,
            file_size_mb=file_size_mb,
            total_layers=total_layers,
            hardware=hardware,
            context_length=context_length,
            parameter_count=parameter_count,
        )

        results.append({
            "quant": quant,
            "plan": plan,
            "score": _score_plan(plan, quant),
        })

    # Sort by score (higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)

    if not results:
        return {
            "recommended": "Q4_K_M",
            "reason": "Varsayılan quantization seviyesi önerildi.",
            "alternatives": [],
        }

    best = results[0]
    return {
        "recommended": best["quant"],
        "reason": best["plan"].recommendation,
        "alternatives": [
            {
                "quant": r["quant"],
                "fits_comfortably": r["plan"].fits_comfortably,
                "estimated_speed": r["plan"].estimated_tokens_per_sec,
                "disk_streaming": r["plan"].disk_streamed_layers > 0,
            }
            for r in results[1:5]  # Top 4 alternatives
        ],
    }


def _score_plan(plan: OffloadPlan, quant: str) -> float:
    """
    Score an offload plan for ranking.
    Higher quality quant is preferred IF it doesn't require disk streaming.
    """
    bpw = QUANT_BPW.get(quant, 4.8)
    quality_score = bpw * 10  # Higher bits = higher quality

    # Penalty for disk streaming
    if plan.disk_streamed_layers > 0:
        quality_score -= 100  # Heavy penalty

    # Penalty for not fitting comfortably
    if not plan.fits_comfortably:
        quality_score -= 30

    # Bonus for speed
    quality_score += plan.estimated_tokens_per_sec * 2

    return quality_score
