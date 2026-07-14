"""
AI Runner — Quantization Decision Matrix
Implements Section 12 of the specification.
"""

from typing import List, Optional
from pydantic import BaseModel


class QuantInfo(BaseModel):
    level: str
    bits_per_weight: float
    quality_loss: str
    quality_loss_en: str
    recommended_use: str
    recommended_use_en: str


# ── Quantization Matrix (Section 12) ──

QUANT_MATRIX: List[QuantInfo] = [
    QuantInfo(
        level="Q2_K",
        bits_per_weight=2.6,
        quality_loss="Belirgin",
        quality_loss_en="Significant",
        recommended_use="Sadece VRAM çok kısıtlıysa (son çare)",
        recommended_use_en="Only when VRAM is severely limited (last resort)",
    ),
    QuantInfo(
        level="Q3_K_M",
        bits_per_weight=3.9,
        quality_loss="Orta",
        quality_loss_en="Moderate",
        recommended_use="Düşük VRAM'de kabul edilebilir kalite",
        recommended_use_en="Acceptable quality on low VRAM",
    ),
    QuantInfo(
        level="Q4_K_M",
        bits_per_weight=4.8,
        quality_loss="Az",
        quality_loss_en="Minor",
        recommended_use="Varsayılan öneri — kalite/boyut dengesi en iyi",
        recommended_use_en="Default recommendation — best quality/size balance",
    ),
    QuantInfo(
        level="Q5_K_M",
        bits_per_weight=5.7,
        quality_loss="Çok az",
        quality_loss_en="Very minor",
        recommended_use="Orta-üst VRAM, kalite öncelikliyse",
        recommended_use_en="Mid-to-high VRAM, when quality is priority",
    ),
    QuantInfo(
        level="Q6_K",
        bits_per_weight=6.6,
        quality_loss="İhmal edilebilir",
        quality_loss_en="Negligible",
        recommended_use="Bol VRAM varsa",
        recommended_use_en="When VRAM is abundant",
    ),
    QuantInfo(
        level="Q8_0",
        bits_per_weight=8.5,
        quality_loss="Neredeyse yok",
        quality_loss_en="Nearly none",
        recommended_use="Sadece bol kaynaklı sistemler",
        recommended_use_en="Only for well-resourced systems",
    ),
]


def get_quant_info(level: str) -> Optional[QuantInfo]:
    """Get quantization info for a specific level."""
    for q in QUANT_MATRIX:
        if q.level == level:
            return q
    return None


def get_all_quants() -> List[QuantInfo]:
    """Return the full quantization matrix."""
    return QUANT_MATRIX


def get_quant_recommendation_text(
    selected_quant: str,
    reason: str,
    language: str = "tr"
) -> str:
    """
    Generate a human-readable recommendation explanation.
    The system shows why a particular quant was chosen (Section 12 requirement).
    """
    info = get_quant_info(selected_quant)
    if not info:
        return reason

    if language == "tr":
        return (
            f"{selected_quant} seçildi "
            f"(ağırlık başına ~{info.bits_per_weight} bit, "
            f"kalite kaybı: {info.quality_loss}). "
            f"{reason}"
        )
    else:
        return (
            f"{selected_quant} selected "
            f"(~{info.bits_per_weight} bits per weight, "
            f"quality loss: {info.quality_loss_en}). "
            f"{reason}"
        )


def estimate_file_size_gb(
    parameter_count: int,
    quant: str
) -> float:
    """Estimate model file size in GB for display."""
    info = get_quant_info(quant)
    bpw = info.bits_per_weight if info else 4.8
    size_bytes = parameter_count * bpw / 8
    return round(size_bytes / (1024 ** 3), 1)
