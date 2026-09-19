import pytest

from provenance_graph import Edge, GraphError, Node, validate_graph


def test_graph_rejects_cycles():
    nodes = [Node("a", "x", "a"), Node("b", "x", "b")]
    with pytest.raises(GraphError, match="cycle"):
        validate_graph(nodes, [Edge("a", "b", "uses"), Edge("b", "a", "feeds_back")])
