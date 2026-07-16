"""Automatic tensor-role discovery for common Llama/Qwen-style layouts."""

from __future__ import annotations

import re


ROLE_SUFFIXES = {
    "attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o",
    "ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down",
}


def discover_layout(tensor_names: list[str]) -> dict:
    blocks: dict[str, dict[str, str]] = {}
    for name in tensor_names:
        match = re.match(r"^(.*(?:blk|block)\.\d+)\.(.+?)(?:\.weight)?$", name)
        if not match:
            continue
        prefix, suffix = match.groups()
        role = ROLE_SUFFIXES.get(suffix)
        if role:
            blocks.setdefault(prefix, {})[role] = name
    ordered = [
        {
            "prefix": prefix,
            "tensors": blocks[prefix],
            "complete": len(blocks[prefix]) == len(ROLE_SUFFIXES),
            "missing_roles": sorted(set(ROLE_SUFFIXES.values()) - set(blocks[prefix])),
        }
        for prefix in sorted(blocks, key=lambda value: int(re.search(r"\.(\d+)$", value).group(1)))
    ]
    return {
        "blocks": ordered,
        "block_count": len(ordered),
        "complete_blocks": sum(item["complete"] for item in ordered),
        "required_roles": sorted(ROLE_SUFFIXES.values()),
    }
