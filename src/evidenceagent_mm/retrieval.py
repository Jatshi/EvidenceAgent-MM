"""Hybrid lexical/dense retrieval with reciprocal-rank fusion and graph expansion."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from evidenceagent_mm.schema import RetrievalHit
from evidenceagent_mm.store import EvidenceStore

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    cjk = [char for char in lowered if "\u3400" <= char <= "\u9fff"]
    cjk_bigrams = ["".join(cjk[index : index + 2]) for index in range(len(cjk) - 1)]
    return words + cjk + cjk_bigrams


class DenseEncoder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


class HashingEncoder:
    """Offline deterministic dense baseline; production can swap in BGE-M3."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimensions
                sign = 1.0 if value & 1 else -1.0
                matrix[row, column] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEncoder:
    """Lazy BGE adapter so the base package remains CPU/offline installable."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install evidenceagent-mm[gpu] for BGE embeddings") from exc
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class HybridRetriever:
    def __init__(
        self,
        store: EvidenceStore,
        encoder: DenseEncoder | None = None,
        *,
        rrf_k: int = 60,
        graph_weight: float = 0.08,
    ) -> None:
        self.store = store
        self.encoder = encoder or HashingEncoder()
        self.rrf_k = rrf_k
        self.graph_weight = graph_weight

    def search(
        self, session_id: str, query: str, *, top_k: int = 8, graph_hops: int = 1
    ) -> list[RetrievalHit]:
        atoms = self.store.list_atoms(session_id)
        if not atoms:
            return []
        fts_ids = self.store.search_lexical(session_id, query, limit=max(top_k * 4, 20))
        query_tokens = set(tokenize(query))
        overlap_scores = {
            atom.evidence_id: len(query_tokens & set(tokenize(atom.text)))
            / max(len(query_tokens), 1)
            for atom in atoms
        }
        local_ids = [
            evidence_id
            for evidence_id, score in sorted(
                overlap_scores.items(), key=lambda item: (-item[1], item[0])
            )
            if score > 0
        ]
        # FTS5 is strong for space-delimited text; token overlap supplies a deterministic
        # CJK baseline when the SQLite build has no language-specific tokenizer.
        lexical_ids = local_ids + [
            evidence_id for evidence_id in fts_ids if evidence_id not in local_ids
        ]
        lexical_rank = {evidence_id: rank for rank, evidence_id in enumerate(lexical_ids, 1)}

        corpus_vectors = self.encoder.encode([atom.text for atom in atoms])
        query_vector = self.encoder.encode([query])[0]
        dense_scores = corpus_vectors @ query_vector
        dense_order = np.argsort(-dense_scores)
        dense_rank = {atoms[index].evidence_id: rank for rank, index in enumerate(dense_order, 1)}

        fused: dict[str, float] = defaultdict(float)
        for evidence_id, rank in lexical_rank.items():
            fused[evidence_id] += 1.0 / (self.rrf_k + rank)
        for evidence_id, rank in dense_rank.items():
            fused[evidence_id] += 1.0 / (self.rrf_k + rank)

        seed_ids = [item[0] for item in sorted(fused.items(), key=lambda item: -item[1])[:top_k]]
        graph = self.store.load_graph(session_id)
        distances = graph.expand(seed_ids, max_hops=graph_hops)
        for evidence_id, distance in distances.items():
            if distance:
                fused[evidence_id] += self.graph_weight / distance

        ranked_ids = sorted(fused, key=lambda evidence_id: (-fused[evidence_id], evidence_id))[
            :top_k
        ]
        hits: list[RetrievalHit] = []
        for evidence_id in ranked_ids:
            atom = graph.atoms[evidence_id]
            lexical_position = lexical_rank.get(evidence_id)
            dense_position = dense_rank.get(evidence_id)
            dense_index = next(
                index for index, item in enumerate(atoms) if item.evidence_id == evidence_id
            )
            semantic = max(0.0, float(dense_scores[dense_index]))
            if lexical_position == 1:
                calibrated_score = 1.0
            elif lexical_position is not None:
                calibrated_score = 0.65 / lexical_position + 0.35 * semantic
            elif distances.get(evidence_id, 0) > 0:
                calibrated_score = 0.35 + 0.35 * semantic
            else:
                calibrated_score = 0.45 * semantic
            hits.append(
                RetrievalHit(
                    atom=atom,
                    score=max(0.0, min(1.0, calibrated_score)),
                    lexical_rank=lexical_position,
                    dense_rank=dense_position,
                    graph_distance=distances.get(evidence_id, 0),
                )
            )
        return hits


def cosine_similarity(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if not math.isclose(denominator, 0.0) else 0.0
