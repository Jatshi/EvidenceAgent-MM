"""FastAPI application factory. Bind to localhost unless authentication is added."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from evidenceagent_mm.agent import EvidenceAgent
from evidenceagent_mm.pipeline import ingest_fixture
from evidenceagent_mm.retrieval import HybridRetriever
from evidenceagent_mm.schema import AgentResponse, EvidenceAtom, FixtureDocument, QueryRequest
from evidenceagent_mm.store import EvidenceStore


def create_app(db_path: str | Path = "data/processed/evidence.db") -> FastAPI:
    store = EvidenceStore(db_path)
    retriever = HybridRetriever(store)
    agent = EvidenceAgent(retriever)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(
        title="EvidenceAgent-MM",
        version="0.1.0",
        description="Evidence-grounded multimodal QA with explicit abstention.",
        lifespan=lifespan,
    )
    app.state.store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.post(
        "/v1/sessions/import-fixture",
        status_code=status.HTTP_201_CREATED,
        response_model=dict[str, int | str],
    )
    def import_fixture(fixture: FixtureDocument) -> dict[str, int | str]:
        return ingest_fixture(store, fixture)

    @app.post("/v1/query", response_model=AgentResponse)
    def query(request: QueryRequest) -> AgentResponse:
        return agent.answer(request.session_id, request.question, top_k=request.top_k)

    @app.get("/v1/evidence/{evidence_id}", response_model=EvidenceAtom)
    def evidence(evidence_id: str) -> EvidenceAtom:
        atom = store.get_atom(evidence_id)
        if atom is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        return atom

    static_path = Path(__file__).with_name("static") / "index.html"

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_path)

    return app


app = create_app()
