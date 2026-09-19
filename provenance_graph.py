"""Acyclic provenance graphs for TIMDR evidence reports."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from evidence_runner import validate_report


class GraphError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str
    sha256: str = ""


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str


def _id(kind: str, value: str) -> str:
    return f"{kind}:{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def build_report_graph(report: Mapping[str, Any]) -> tuple[list[Node], list[Edge]]:
    """Map one validated evidence report to a directed, auditable graph."""
    errors = validate_report(report)
    if errors:
        raise GraphError("Invalid report: " + " | ".join(errors))

    prereg, manifest, source = report["preregistration"], report["manifest"], report["source_result"]
    dataset_id = str(report["dataset_id"])
    raw_hashes = manifest.get("raw_file_hashes_declared", {})
    report_json = json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    dataset = Node(_id("dataset", dataset_id), "dataset", dataset_id)
    raw = Node(_id("raw-hashes", json.dumps(raw_hashes, sort_keys=True)), "raw_hashes", "declared raw-file hashes")
    prereg_node = Node(_id("prereg", prereg["sha256"]), "preregistration", "frozen preregistration", prereg["sha256"])
    manifest_node = Node(_id("manifest", manifest["sha256"]), "manifest", "dataset manifest", manifest["sha256"])
    result_node = Node(_id("result", source["sha256"]), "source_result", "source result artifact", source["sha256"])
    controls = Node(_id("controls", source["sha256"]), "controls", "positive and negative controls")
    test = Node(_id("test", source["sha256"]), "test", str(report["method"]))
    report_node = Node(_id("report", report_json), "evidence_report", str(report["declared_verdict"]))
    nodes = [dataset, raw, prereg_node, manifest_node, result_node, controls, test, report_node]
    edges = [
        Edge(raw.id, dataset.id, "identifies"),
        Edge(dataset.id, manifest_node.id, "described_by"),
        Edge(prereg_node.id, report_node.id, "governs"),
        Edge(manifest_node.id, report_node.id, "freezes_data"),
        Edge(result_node.id, controls.id, "contains"),
        Edge(result_node.id, test.id, "contains"),
        Edge(controls.id, report_node.id, "gates"),
        Edge(test.id, report_node.id, "evidence_for"),
    ]
    validate_graph(nodes, edges)
    return nodes, edges


def validate_graph(nodes: list[Node], edges: list[Edge]) -> None:
    """Require unique nodes, valid edges, an acyclic flow, and sourced reports."""
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        raise GraphError("Duplicate node identifiers.")
    known = set(ids)
    if any(edge.source not in known or edge.target not in known for edge in edges):
        raise GraphError("An edge points to a missing node.")
    children = {node_id: [] for node_id in known}
    indegree = {node_id: 0 for node_id in known}
    for edge in edges:
        children[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = [node_id for node_id, count in indegree.items() if count == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        raise GraphError("Provenance graph contains a forbidden cycle.")
    for node in nodes:
        if node.kind == "evidence_report" and not any(edge.target == node.id for edge in edges):
            raise GraphError("An evidence report has no provenance sources.")


def export_graph(nodes: list[Node], edges: list[Edge]) -> dict[str, Any]:
    validate_graph(nodes, edges)
    return {"schema": "timdr-provenance-graph/1", "nodes": [node.__dict__ for node in nodes], "edges": [edge.__dict__ for edge in edges]}


def write_graph(graph: Mapping[str, Any], json_path: Path, dot_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(graph, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    node_lines = [f'  "{n["id"]}" [label="{n["kind"]}: {n["label"]}"];' for n in graph["nodes"]]
    edge_lines = [f'  "{e["source"]}" -> "{e["target"]}" [label="{e["relation"]}"];' for e in graph["edges"]]
    dot_path.write_text("digraph TIMDR_Provenance {\n  rankdir=LR;\n" + "\n".join(node_lines + edge_lines) + "\n}\n", encoding="utf-8")
