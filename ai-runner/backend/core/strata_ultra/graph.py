"""Composable reference graph for Strata low-bit linear layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .executor import StrataRuntime


@dataclass(frozen=True)
class LinearNode:
    tensor_name: str
    activation: Literal["none", "relu"] = "none"


class StrataGraph:
    """Run a sequence of packed linear tensors through one pager-backed runtime."""

    def __init__(self, runtime: StrataRuntime, nodes: list[LinearNode]):
        if not nodes:
            raise ValueError("graph must contain at least one node")
        self.runtime = runtime
        self.nodes = nodes

    def run(self, vector: list[float]) -> list[float]:
        output = list(vector)
        for node in self.nodes:
            output = self.runtime.tensor_matvec(node.tensor_name, output)
            if node.activation == "relu":
                output = [max(0.0, value) for value in output]
        return output
