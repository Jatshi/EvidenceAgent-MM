"""Evidence graph construction and bounded expansion."""

from __future__ import annotations

from collections import defaultdict, deque

from evidenceagent_mm.schema import EvidenceAtom, EvidenceEdge, Modality


class EvidenceGraph:
    def __init__(self, atoms: list[EvidenceAtom] | None = None) -> None:
        self.atoms: dict[str, EvidenceAtom] = {}
        self.adjacency: dict[str, list[EvidenceEdge]] = defaultdict(list)
        for atom in atoms or []:
            self.add_atom(atom)

    def add_atom(self, atom: EvidenceAtom) -> None:
        existing = self.atoms.get(atom.evidence_id)
        if existing is not None and existing != atom:
            raise ValueError(f"conflicting atom id: {atom.evidence_id}")
        self.atoms[atom.evidence_id] = atom

    def add_edge(self, edge: EvidenceEdge, *, bidirectional: bool = True) -> None:
        if edge.source_id not in self.atoms or edge.target_id not in self.atoms:
            raise KeyError("edge endpoint is not present in graph")
        if edge not in self.adjacency[edge.source_id]:
            self.adjacency[edge.source_id].append(edge)
        if bidirectional:
            reverse = edge.model_copy(
                update={"source_id": edge.target_id, "target_id": edge.source_id}
            )
            if reverse not in self.adjacency[edge.target_id]:
                self.adjacency[edge.target_id].append(reverse)

    def expand(self, seed_ids: list[str], max_hops: int = 1) -> dict[str, int]:
        """Return reachable evidence IDs with their shortest graph distance."""
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for evidence_id in seed_ids:
            if evidence_id in self.atoms and evidence_id not in distances:
                distances[evidence_id] = 0
                queue.append((evidence_id, 0))
        while queue:
            current, distance = queue.popleft()
            if distance >= max_hops:
                continue
            for edge in self.adjacency.get(current, []):
                if edge.target_id not in distances:
                    distances[edge.target_id] = distance + 1
                    queue.append((edge.target_id, distance + 1))
        return distances


def intervals_overlap(left: EvidenceAtom, right: EvidenceAtom, tolerance_ms: int = 0) -> bool:
    return (
        left.start_ms <= right.end_ms + tolerance_ms
        and right.start_ms <= left.end_ms + tolerance_ms
    )


def build_evidence_graph(
    atoms: list[EvidenceAtom], *, slide_tolerance_ms: int = 1_500
) -> EvidenceGraph:
    graph = EvidenceGraph(atoms)
    by_session: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        by_session[atom.session_id].append(atom)

    for session_atoms in by_session.values():
        ordered = sorted(
            session_atoms, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)
        )
        for left, right in zip(ordered, ordered[1:], strict=False):
            graph.add_edge(
                EvidenceEdge(
                    source_id=left.evidence_id, target_id=right.evidence_id, relation="next"
                )
            )
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right.start_ms > left.end_ms + slide_tolerance_ms:
                    break
                if left.speaker_id and left.speaker_id == right.speaker_id:
                    graph.add_edge(
                        EvidenceEdge(
                            source_id=left.evidence_id,
                            target_id=right.evidence_id,
                            relation="same_speaker",
                            confidence=min(left.confidence, right.confidence),
                        )
                    )
                if intervals_overlap(left, right):
                    relation = "overlaps"
                    if {left.modality, right.modality} & {Modality.SLIDE, Modality.OCR} and {
                        left.modality,
                        right.modality,
                    } & {Modality.TRANSCRIPT, Modality.AUDIO}:
                        relation = "shown_during"
                    graph.add_edge(
                        EvidenceEdge(
                            source_id=left.evidence_id,
                            target_id=right.evidence_id,
                            relation=relation,
                            confidence=min(left.confidence, right.confidence),
                        )
                    )
    return graph
