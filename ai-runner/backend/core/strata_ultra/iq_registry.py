"""GGML importance-quantization type registry.

The registry is deliberately data-first: GGUF type IDs and block geometry are
kept in one place, while a codec is marked decodable only after its reference
implementation has been verified with known blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IQCodec:
    name: str
    type_id: int
    bits_per_weight: float
    block_values: int
    block_bytes: Optional[int]
    decoder: Optional[str] = None

    @property
    def decodable(self) -> bool:
        return self.decoder is not None


IQ_CODECS = (
    IQCodec("IQ2_XXS", 16, 2.0625, 256, 66),
    IQCodec("IQ2_XS", 17, 2.3125, 256, 74),
    IQCodec("IQ3_XXS", 18, 3.0625, 256, 98),
    IQCodec("IQ1_S", 19, 1.5625, 256, 50),
    IQCodec("IQ4_NL", 20, 4.5000, 32, 18, "_decode_iq4_nl"),
    IQCodec("IQ3_S", 21, 3.4375, 256, 110),
    IQCodec("IQ2_S", 22, 2.5000, 256, 82),
    IQCodec("IQ4_XS", 23, 4.2500, 256, 136),
    IQCodec("IQ1_M", 29, 1.7500, 256, 56),
)

# Type IDs implemented by native/iq/strata_iq.cpp. Keep this list next to the
# registry so capability reporting and future conversion checks share one ABI
# contract.
NATIVE_BRIDGE_TYPE_IDS = frozenset({16, 17, 18, 19, 21, 22, 23, 29})

BY_TYPE_ID = {codec.type_id: codec for codec in IQ_CODECS}
BY_NAME = {codec.name: codec for codec in IQ_CODECS}


def get_iq_codec(type_id: int) -> Optional[IQCodec]:
    return BY_TYPE_ID.get(type_id)


def capability_report(native_bridge: bool = False) -> list[dict]:
    """Return codec capabilities for the active runtime configuration.

    The pure-Python registry remains deterministic by default.  When the
    optional GGML bridge is loaded, the bridge-backed IQ codecs are reported
    as decodable as well, so API consumers do not have to infer capability
    from a separate boolean flag.
    """
    return [
        {
            "name": codec.name,
            "type_id": codec.type_id,
            "bits_per_weight": codec.bits_per_weight,
            "block_values": codec.block_values,
            "block_bytes": codec.block_bytes,
            "decodable": codec.decodable or (native_bridge and codec.type_id in NATIVE_BRIDGE_TYPE_IDS),
            "decoder": codec.decoder or ("native-ggml" if native_bridge and codec.type_id in NATIVE_BRIDGE_TYPE_IDS else None),
        }
        for codec in IQ_CODECS
    ]
