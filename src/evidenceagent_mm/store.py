"""SQLite evidence store with FTS5 and explicit graph edges."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from evidenceagent_mm.graph import EvidenceGraph
from evidenceagent_mm.schema import EvidenceAtom, EvidenceEdge, Modality


class EvidenceStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._fts_enabled = True
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                duration_ms INTEGER NOT NULL CHECK(duration_ms > 0),
                source_license TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                speaker_id TEXT,
                page_no INTEGER,
                bbox_json TEXT,
                confidence REAL NOT NULL,
                attributes_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_session_time
                ON evidence(session_id, start_ms, end_ms);
            CREATE INDEX IF NOT EXISTS idx_evidence_speaker
                ON evidence(session_id, speaker_id);
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY(source_id, target_id, relation),
                FOREIGN KEY(source_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE
            );
            """
        )
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(evidence_id UNINDEXED, text)"
            )
        except sqlite3.OperationalError:
            self._fts_enabled = False
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert_session(
        self, session_id: str, title: str, duration_ms: int, source_license: str
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO sessions(session_id, title, duration_ms, source_license)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              title=excluded.title,
              duration_ms=excluded.duration_ms,
              source_license=excluded.source_license
            """,
            (session_id, title, duration_ms, source_license),
        )
        self.connection.commit()

    def upsert_atoms(self, atoms: Iterable[EvidenceAtom]) -> int:
        atom_list = list(atoms)
        with self.connection:
            for atom in atom_list:
                values = (
                    atom.evidence_id,
                    atom.session_id,
                    atom.modality.value,
                    atom.start_ms,
                    atom.end_ms,
                    atom.text,
                    atom.source_uri,
                    atom.speaker_id,
                    atom.page_no,
                    json.dumps(atom.bbox),
                    atom.confidence,
                    json.dumps(atom.attributes, ensure_ascii=False, sort_keys=True),
                )
                self.connection.execute(
                    """
                    INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(evidence_id) DO UPDATE SET
                      session_id=excluded.session_id, modality=excluded.modality,
                      start_ms=excluded.start_ms, end_ms=excluded.end_ms,
                      text=excluded.text, source_uri=excluded.source_uri,
                      speaker_id=excluded.speaker_id, page_no=excluded.page_no,
                      bbox_json=excluded.bbox_json, confidence=excluded.confidence,
                      attributes_json=excluded.attributes_json
                    """,
                    values,
                )
                if self._fts_enabled:
                    self.connection.execute(
                        "DELETE FROM evidence_fts WHERE evidence_id = ?", (atom.evidence_id,)
                    )
                    self.connection.execute(
                        "INSERT INTO evidence_fts(evidence_id, text) VALUES (?, ?)",
                        (atom.evidence_id, atom.text),
                    )
        return len(atom_list)

    def upsert_edges(self, edges: Iterable[EvidenceEdge]) -> int:
        edge_list = list(edges)
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO edges(source_id, target_id, relation, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation)
                DO UPDATE SET confidence=excluded.confidence
                """,
                [
                    (edge.source_id, edge.target_id, edge.relation, edge.confidence)
                    for edge in edge_list
                ],
            )
        return len(edge_list)

    def get_atom(self, evidence_id: str) -> EvidenceAtom | None:
        row = self.connection.execute(
            "SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return self._row_to_atom(row) if row else None

    def list_atoms(self, session_id: str) -> list[EvidenceAtom]:
        rows = self.connection.execute(
            "SELECT * FROM evidence WHERE session_id = ? ORDER BY start_ms, evidence_id",
            (session_id,),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def search_lexical(self, session_id: str, query: str, limit: int = 20) -> list[str]:
        terms = [token.strip(".,?!:;\"'()[]{}") for token in query.split()]
        terms = [term for term in terms if term]
        if not terms:
            return []
        if self._fts_enabled:
            safe_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
            try:
                rows = self.connection.execute(
                    """
                    SELECT e.evidence_id
                    FROM evidence_fts f JOIN evidence e ON e.evidence_id=f.evidence_id
                    WHERE evidence_fts MATCH ? AND e.session_id = ?
                    ORDER BY bm25(evidence_fts) LIMIT ?
                    """,
                    (safe_query, session_id, limit),
                ).fetchall()
                return [str(row[0]) for row in rows]
            except sqlite3.OperationalError:
                pass
        clauses = " OR ".join("LOWER(text) LIKE ?" for _ in terms)
        rows = self.connection.execute(
            f"SELECT evidence_id FROM evidence WHERE session_id = ? AND ({clauses}) LIMIT ?",  # noqa: S608
            (session_id, *(f"%{term.lower()}%" for term in terms), limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def load_graph(self, session_id: str) -> EvidenceGraph:
        graph = EvidenceGraph(self.list_atoms(session_id))
        rows = self.connection.execute(
            """
            SELECT g.* FROM edges g
            JOIN evidence e ON e.evidence_id = g.source_id
            WHERE e.session_id = ?
            """,
            (session_id,),
        ).fetchall()
        for row in rows:
            graph.add_edge(EvidenceEdge(**dict(row)), bidirectional=False)
        return graph

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> EvidenceAtom:
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
        return EvidenceAtom(
            evidence_id=row["evidence_id"],
            session_id=row["session_id"],
            modality=Modality(row["modality"]),
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            text=row["text"],
            source_uri=row["source_uri"],
            speaker_id=row["speaker_id"],
            page_no=row["page_no"],
            bbox=bbox,
            confidence=row["confidence"],
            attributes=json.loads(row["attributes_json"]),
        )
