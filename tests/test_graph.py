from __future__ import annotations

from evidenceagent_mm.graph import build_evidence_graph, intervals_overlap
from evidenceagent_mm.schema import FixtureDocument


def test_graph_connects_transcript_to_visible_slide(fixture_document: FixtureDocument) -> None:
    graph = build_evidence_graph(fixture_document.atoms)
    relations = {
        (edge.source_id, edge.target_id, edge.relation)
        for edges in graph.adjacency.values()
        for edge in edges
    }
    assert ("utt-01", "ocr-01", "shown_during") in relations
    assert graph.expand(["utt-01"], max_hops=1)["ocr-01"] == 1


def test_interval_overlap_is_symmetric(fixture_document: FixtureDocument) -> None:
    left, right = fixture_document.atoms[:2]
    assert intervals_overlap(left, right)
    assert intervals_overlap(right, left)
