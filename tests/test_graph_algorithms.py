import pytest
from codegen.graph_algorithms import (
    topological_sort,
    has_cycle,
    condensation_execution_order,
    tarjan_scc,
    build_layer_graph,
)
from codegen.types import BaseLayer


def _layer(name: str, inputs=(), outputs=()):
    """Helper to create test layers."""
    return BaseLayer(
        layer_id=0,
        name=name,
        op_type="Add",
        input_size=1,
        output_size=1,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


# ============================================================================
# Topological Sort Tests (Acyclic Graphs Only)
# ============================================================================


def test_topological_sort_for_acyclic_graph():
    """Simple linear dependency chain."""
    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}
    order = topological_sort(layers, tensor_producers, ("x",))
    assert order == ["A", "B", "C"]
    assert has_cycle(layers, tensor_producers, ("x",)) is False


def test_topological_sort_with_multiple_inputs():
    """Multiple inputs to a single layer (fan-in)."""
    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("y",), outputs=("b",)),
        "C": _layer("C", inputs=("a", "b"), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}
    order = topological_sort(layers, tensor_producers, ("x", "y"))
    # A and B can be in any order, but both must come before C
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("C")


def test_topological_sort_with_fan_out():
    """Single output used by multiple layers (fan-out)."""
    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("a",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}
    order = topological_sort(layers, tensor_producers, ("x",))
    assert order[0] == "A"
    assert set(order[1:]) == {"B", "C"}


def test_topological_sort_raises_on_cycle():
    """Topological sort should raise ValueError when cycle is detected."""
    layers = {
        "A": _layer("A", inputs=("x", "c"), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}

    with pytest.raises(ValueError, match="Cycle detected"):
        topological_sort(layers, tensor_producers, ("x",))


def test_topological_sort_single_node():
    """Single isolated node."""
    layers = {"A": _layer("A", inputs=("x",), outputs=("a",))}
    tensor_producers = {"a": "A"}
    order = topological_sort(layers, tensor_producers, ("x",))
    assert order == ["A"]


# ============================================================================
# Has Cycle Tests
# ============================================================================


def test_has_cycle_detects_simple_cycle():
    """Simple 3-node cycle."""
    layers = {
        "A": _layer("A", inputs=("x", "c"), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}
    assert has_cycle(layers, tensor_producers, ("x",)) is True


def test_has_cycle_detects_self_loop():
    """Self-loop (node depends on itself)."""
    layers = {
        "A": _layer("A", inputs=("x", "a"), outputs=("a",)),
    }
    tensor_producers = {"a": "A"}
    assert has_cycle(layers, tensor_producers, ("x",)) is True


def test_has_cycle_detects_multiple_sccs():
    """Multiple separate SCCs."""
    layers = {
        "A": _layer("A", inputs=("x", "b"), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("y", "d"), outputs=("c",)),
        "D": _layer("D", inputs=("c",), outputs=("d",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C", "d": "D"}
    assert has_cycle(layers, tensor_producers, ("x", "y")) is True


def test_has_cycle_acyclic_graph():
    """Acyclic DAG should return False."""
    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}
    assert has_cycle(layers, tensor_producers, ("x",)) is False


# ============================================================================
# Tarjan SCC Tests
# ============================================================================


def test_tarjan_scc_single_cycle():
    """Simple cycle should produce one SCC."""
    from codegen.graph_algorithms import tarjan_scc

    adjacency = {
        "A": {"B"},
        "B": {"C"},
        "C": {"A"},
    }
    sccs = tarjan_scc(adjacency)
    assert len(sccs) == 1
    assert set(sccs[0]) == {"A", "B", "C"}


def test_tarjan_scc_multiple_sccs():
    """Multiple SCCs should be separated."""
    from codegen.graph_algorithms import tarjan_scc

    adjacency = {
        "A": {"B"},
        "B": {"A"},  # A-B form an SCC
        "C": {"D"},
        "D": {"C"},  # C-D form an SCC
        "B": {"C"},  # Edge from first SCC to second
    }

    # Rewrite without duplicate key
    adjacency = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"D"},
        "D": {"C"},
    }
    sccs = tarjan_scc(adjacency)
    assert len(sccs) == 2
    # Both SCCs should be present
    nodes_found = set()
    for scc in sccs:
        nodes_found.update(scc)
    assert nodes_found == {"A", "B", "C", "D"}


def test_tarjan_scc_acyclic_graph():
    """Each node is its own SCC in acyclic graph."""
    from codegen.graph_algorithms import tarjan_scc

    adjacency = {
        "A": {"B"},
        "B": {"C"},
        "C": set(),
    }
    sccs = tarjan_scc(adjacency)
    assert len(sccs) == 3
    # Each SCC should be a singleton
    for scc in sccs:
        assert len(scc) == 1


# ============================================================================
# Condensation Execution Order Tests
# ============================================================================


def test_condensation_execution_order_for_cycle_returns_all_nodes():
    """Cyclic graph returns all nodes in valid dependency order."""
    layers = {
        "A": _layer("A", inputs=("x", "c"), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
        "D": _layer("D", inputs=("a",), outputs=("d",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C", "d": "D"}

    assert has_cycle(layers, tensor_producers, ("x",)) is True

    order = condensation_execution_order(layers, tensor_producers, ("x",))
    assert set(order) == set(layers.keys())

    # D depends on A's output, so it cannot execute before the SCC containing A/B/C.
    assert order.index("D") > order.index("A")


def test_condensation_execution_order_acyclic_mimics_topological():
    """Acyclic graph should produce same order as topological_sort."""
    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}

    topo_order = topological_sort(layers, tensor_producers, ("x",))
    cond_order = condensation_execution_order(layers, tensor_producers, ("x",))

    assert topo_order == cond_order


def test_condensation_execution_order_multi_scc_dag():
    """Multiple SCCs with cross-SCC dependencies."""
    layers = {
        # First SCC: A-B cycle
        "A": _layer("A", inputs=("x", "b"), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        # Second SCC: C-D cycle
        "C": _layer("C", inputs=("a",), outputs=("c",)),
        "D": _layer("D", inputs=("c",), outputs=("d",)),
        # Node that depends on second SCC
        "E": _layer("E", inputs=("d",), outputs=("e",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}

    order = condensation_execution_order(layers, tensor_producers, ("x",))
    assert set(order) == set(layers.keys())

    # First SCC must come before second
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("C")
    # Second SCC must come before E
    assert order.index("C") < order.index("E")
    assert order.index("D") < order.index("E")


def test_condensation_execution_order_empty_graph():
    """Empty graph should return empty list."""
    layers = {}
    tensor_producers = {}
    order = condensation_execution_order(layers, tensor_producers, ())
    assert order == []


def test_condensation_execution_order_single_node():
    """Single node graph."""
    layers = {"A": _layer("A", inputs=("x",), outputs=("a",))}
    tensor_producers = {"a": "A"}
    order = condensation_execution_order(layers, tensor_producers, ("x",))
    assert order == ["A"]


# ============================================================================
# Build Layer Graph Tests
# ============================================================================


def test_build_layer_graph_simple_chain():
    """Simple linear chain produces correct adjacency."""
    from codegen.graph_algorithms import build_layer_graph

    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_consumers = {"a": ["B"], "b": ["C"], "c": []}

    adj, rev_adj = build_layer_graph(layers, {}, tensor_consumers)

    assert adj["A"] == {"B"}
    assert adj["B"] == {"C"}
    assert adj["C"] == set()
    assert rev_adj["A"] == set()
    assert rev_adj["B"] == {"A"}
    assert rev_adj["C"] == {"B"}


def test_build_layer_graph_fan_in_fan_out():
    """Complex fan-in/fan-out structure."""
    from codegen.graph_algorithms import build_layer_graph

    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("y",), outputs=("b",)),
        "C": _layer("C", inputs=("a", "b"), outputs=("c",)),
        "D": _layer("D", inputs=("c",), outputs=("d",)),
        "E": _layer("E", inputs=("c",), outputs=("e",)),
    }
    tensor_consumers = {"a": ["C"], "b": ["C"], "c": ["D", "E"], "d": [], "e": []}

    adj, rev_adj = build_layer_graph(layers, {}, tensor_consumers)

    assert adj["A"] == {"C"}
    assert adj["B"] == {"C"}
    assert adj["C"] == {"D", "E"}
    assert adj["D"] == set()
    assert adj["E"] == set()


# ============================================================================
# Integration Tests
# ============================================================================


def test_cycle_detection_integration():
    """Integration test: detect cycle in realistic RNN-like structure."""
    # Simulates: hidden_t = f(input_t, hidden_{t-1})
    layers = {
        "input_embed": _layer("input_embed", inputs=("x_t",), outputs=("x_embed",)),
        "rnn_cell": _layer(
            "rnn_cell",
            inputs=("x_embed", "h_prev"),
            outputs=("h_t",),
        ),
        "output": _layer("output", outputs=("y_t",)),
    }
    tensor_producers = {"x_embed": "input_embed", "h_t": "rnn_cell", "y_t": "output"}
    tensor_producers["h_prev"] = "rnn_cell"  # Self-dependency for hidden state

    # Should detect cycle
    assert has_cycle(layers, tensor_producers, ("x_t",)) is True

    # Should still produce execution order
    order = condensation_execution_order(layers, tensor_producers, ("x_t",))
    assert set(order) == set(layers.keys())
