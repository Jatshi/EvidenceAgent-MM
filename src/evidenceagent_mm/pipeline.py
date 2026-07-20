"""Deterministic fixture ingestion and production adapter boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from evidenceagent_mm.graph import build_evidence_graph
from evidenceagent_mm.schema import EvidenceEdge, FixtureDocument
from evidenceagent_mm.store import EvidenceStore


def load_fixture(path: str | Path) -> FixtureDocument:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FixtureDocument.model_validate(payload)


def ingest_fixture(store: EvidenceStore, fixture: FixtureDocument) -> dict[str, int | str]:
    store.upsert_session(
        fixture.session_id, fixture.title, fixture.duration_ms, fixture.source_license
    )
    atom_count = store.upsert_atoms(fixture.atoms)
    graph = build_evidence_graph(fixture.atoms)
    generated_edges: list[EvidenceEdge] = []
    for edges in graph.adjacency.values():
        generated_edges.extend(edges)
    unique: dict[tuple[str, str, str], EvidenceEdge] = {
        (edge.source_id, edge.target_id, edge.relation): edge for edge in generated_edges
    }
    for edge in fixture.edges:
        unique[(edge.source_id, edge.target_id, edge.relation)] = edge
    edge_count = store.upsert_edges(unique.values())
    return {"session_id": fixture.session_id, "atoms": atom_count, "edges": edge_count}


def ingest_fixture_file(store: EvidenceStore, path: str | Path) -> dict[str, int | str]:
    return ingest_fixture(store, load_fixture(path))
