from codegen.graph_algorithms import (
    topological_sort,
    has_cycle,
    condensation_execution_order,
)
from codegen.types import BaseLayer


def _layer(name: str, inputs=(), outputs=()):
    return BaseLayer(
        layer_id=0,
        name=name,
        op_type="Add",
        input_size=1,
        output_size=1,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def test_topological_sort_for_acyclic_graph():
    layers = {
        "A": _layer("A", inputs=("x",), outputs=("a",)),
        "B": _layer("B", inputs=("a",), outputs=("b",)),
        "C": _layer("C", inputs=("b",), outputs=("c",)),
    }
    tensor_producers = {"a": "A", "b": "B", "c": "C"}
    order = topological_sort(layers, tensor_producers, ("x",))
    assert order == ["A", "B", "C"]
    assert has_cycle(layers, tensor_producers, ("x",)) is False


def test_condensation_execution_order_for_cycle_returns_all_nodes():
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
