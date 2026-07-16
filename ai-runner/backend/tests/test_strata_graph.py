import struct
from pathlib import Path

from backend.core.strata_ultra import (
    LinearNode,
    StrataContainerWriter,
    StrataGraph,
    StrataRuntime,
    TensorRecord,
)


def test_graph_runs_sequential_low_bit_linear_layers(tmp_path: Path):
    target = tmp_path / "graph.strata"
    writer = StrataContainerWriter()
    # First layer is identity; second is [[1, 1], [-1, 1]].
    writer.add_tensor(TensorRecord("layer.0", 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0)))
    writer.add_tensor(TensorRecord("layer.1", 2, 2, 4, "ternary-q05", bytes([0b10_01_10_10]), struct.pack("<f", 1.0)))
    writer.write(target)
    with StrataRuntime(target, 1024, resident_window=1) as runtime:
        graph = StrataGraph(runtime, [LinearNode("layer.0", "relu"), LinearNode("layer.1")])
        assert graph.run([2.0, -3.0]) == [2.0, -2.0]
        assert any(event.action == "load" and event.layer_id == "layer.1" for event in runtime.pager.events)
