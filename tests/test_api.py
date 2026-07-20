from __future__ import annotations

from fastapi.testclient import TestClient

from evidenceagent_mm.api import create_app
from evidenceagent_mm.schema import FixtureDocument


def test_api_health_import_and_query(tmp_path, fixture_document: FixtureDocument) -> None:
    with TestClient(create_app(tmp_path / "api.db")) as client:
        assert client.get("/health").json()["status"] == "ok"
        imported = client.post(
            "/v1/sessions/import-fixture", json=fixture_document.model_dump(mode="json")
        )
        assert imported.status_code == 201
        response = client.post(
            "/v1/query",
            json={
                "session_id": "demo-session",
                "question": "Who proposed design B and which slide page was on screen?",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "answered"
        assert len(body["citations"]) == 2


def test_api_returns_404_for_unknown_evidence(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db")) as client:
        assert client.get("/v1/evidence/missing").status_code == 404


def test_demo_page_exposes_evidence_console(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db")) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "EVIDENCE REPORT" in response.text
        assert "prefers-reduced-motion" in response.text
