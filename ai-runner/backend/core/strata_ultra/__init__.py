"""Strata Ultra: experimental ultra-low-bit tensor primitives.

This package is intentionally independent from llama.cpp.  The first milestone
provides a deterministic ternary tensor codec that can be used by future CPU,
CUDA, and Vulkan execution backends.
"""

from .format import STRATA_FORMAT_VERSION, TensorHeader
from .ternary import decode_ternary, encode_ternary
from .kv_cache import PackedKV, decode_kv, encode_kv, estimate_kv_bytes, kv_memory_report
from .paging import LayerPager, PageEvent
from .benchmark import run_codec_benchmark
from .container import StrataContainerReader, StrataContainerWriter, TensorRecord
from .converter import convert_gguf_to_strata
from .executor import StrataRuntime, matmul, matvec
from .kv_runtime import KVSnapshot, UltraKVCache
from .graph import LinearNode, StrataGraph
from .attention import LowBitAttention
from .layers import LowBitMLP, rms_norm, silu
from .transformer import LowBitTransformer, LowBitTransformerBlock

__all__ = [
    "STRATA_FORMAT_VERSION",
    "TensorHeader",
    "encode_ternary",
    "decode_ternary",
    "PackedKV",
    "encode_kv",
    "decode_kv",
    "estimate_kv_bytes",
    "kv_memory_report",
    "LayerPager",
    "PageEvent",
    "run_codec_benchmark",
    "StrataContainerReader",
    "StrataContainerWriter",
    "TensorRecord",
    "convert_gguf_to_strata",
    "StrataRuntime",
    "matvec",
    "matmul",
    "KVSnapshot",
    "UltraKVCache",
    "LinearNode",
    "StrataGraph",
    "LowBitAttention",
    "LowBitMLP",
    "rms_norm",
    "silu",
    "LowBitTransformerBlock",
    "LowBitTransformer",
]
