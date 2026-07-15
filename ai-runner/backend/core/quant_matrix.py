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
    # ── Ultra Sıkıştırma: I-Matrix Serisi (Importance Matrix) ──────────────
    # I-Matrix, hangi ağırlıkların kritik olduğunu analiz ederek düşük bitte
    # çok daha iyi kalite korur. Normal Q2'den belirgin şekilde üstündür.
    QuantInfo(
        level="IQ1_S",
        bits_per_weight=1.56,
        quality_loss="Çok Belirgin",
        quality_loss_en="Very Significant",
        recommended_use="Sadece çok kısıtlı RAM'de (≤6 GB). 70B+ modelleri 10 GB'a indirmek için.",
        recommended_use_en="Only for very limited RAM (≤6 GB). Fits 70B+ into ~10 GB.",
    ),
    QuantInfo(
        level="IQ2_XXS",
        bits_per_weight=2.06,
        quality_loss="Belirgin (I-Matrix ile azaltılmış)",
        quality_loss_en="Significant (mitigated by I-Matrix)",
        recommended_use="Çok kısıtlı VRAM — 50B modeli ~14 GB'a indirir. Normal Q2'den çok daha iyi kalite.",
        recommended_use_en="Very limited VRAM — fits 50B into ~14 GB. Far better than regular Q2.",
    ),
    QuantInfo(
        level="IQ2_XS",
        bits_per_weight=2.31,
        quality_loss="Orta-Belirgin",
        quality_loss_en="Moderate-Significant",
        recommended_use="Kısıtlı VRAM — IQ2_XXS'ten daha iyi kalite, biraz daha büyük.",
        recommended_use_en="Limited VRAM — better quality than IQ2_XXS, slightly larger.",
    ),
    QuantInfo(
        level="IQ3_XS",
        bits_per_weight=3.3,
        quality_loss="Orta",
        quality_loss_en="Moderate",
        recommended_use="Dengeli ultra-sıkıştırma — Q3_K_M'den daha iyi kalite, daha küçük boyut.",
        recommended_use_en="Balanced ultra-compression — better quality than Q3_K_M, smaller size.",
    ),
    # ── Standart K-Quant Serisi ─────────────────────────────────────────────
    QuantInfo(
        level="Q2_K",
        bits_per_weight=2.6,
        quality_loss="Belirgin",
        quality_loss_en="Significant",
        recommended_use="Sadece VRAM çok kısıtlıysa — IQ2_XXS önerilir bunun yerine",
        recommended_use_en="Only when VRAM is severely limited — prefer IQ2_XXS instead",
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
