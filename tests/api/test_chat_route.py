"""HTTP contract for the chat endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from domain.exceptions import LLMUnavailableError, RetrievalError
from domain.models import (
    Confidence,
    ConfidenceBand,
    Language,
    RAGResponse,
    RegulationDomain,
)
from infrastructure.sessions.memory_store import MemorySessionStore
from presentation.api.app import create_app
from presentation.api.deps import Container


def make_response(
    *, band: ConfidenceBand = ConfidenceBand.HIGH, sources=None, empty: bool = False
) -> RAGResponse:
    return RAGResponse(
        query="GPA tối thiểu là bao nhiêu?",
        answer="GPA tối thiểu là 2.0/4.0.",
        language=Language.VI,
        confidence=Confidence(band=band, score=0.91),
        sources=sources if sources is not None else [],
        is_empty_context=empty,
    )


@pytest.fixture
def container(sample_chunks):
    settings = Settings(
        gemini_api_key="test",
        zilliz_cloud_endpoint="https://example.zillizcloud.com",
        zilliz_cloud_api_key="test",
        admin_token="secret",
        retrieval_k=5,
    )
    rag_service = AsyncMock()
    rag_service.answer.return_value = make_response(sources=sample_chunks)
    return Container(
        settings=settings,
        rag_service=rag_service,
        ingestion_service=AsyncMock(),
        admin_service=AsyncMock(),
        sessions=MemorySessionStore(max_turns=5, ttl_seconds=3600),
        vector_store=AsyncMock(),
        registry=MagicMock(),
    )


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as test_client:
        yield test_client


class TestChat:
    def test_answers_a_question(self, client):
        response = client.post("/api/chat", json={"message": "GPA tối thiểu là bao nhiêu?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "GPA tối thiểu là 2.0/4.0."
        assert body["language"] == "vi"
        assert body["confidence"]["band"] == "HIGH"

    def test_issues_a_session_id(self, client):
        body = client.post("/api/chat", json={"message": "xin chào"}).json()
        assert body["session_id"]

    def test_every_source_carries_a_page_and_section(self, client):
        sources = client.post("/api/chat", json={"message": "GPA?"}).json()["sources"]
        assert sources
        for source in sources:
            assert source["source_url"]
            assert source["doc_title"]
            assert source["page_start"] >= 1
            assert "section" in source
            assert "doc_updated_at" in source

    def test_forwards_the_domain_filter(self, client, container):
        client.post("/api/chat", json={"message": "học bổng?", "domain": "SCHOLARSHIP"})
        request = container.rag_service.answer.await_args.args[0]
        assert request.domain is RegulationDomain.SCHOLARSHIP

    def test_forwards_configured_top_k(self, client, container):
        client.post("/api/chat", json={"message": "GPA?"})
        assert container.rag_service.answer.await_args.args[0].top_k == 5

    def test_second_turn_carries_the_first(self, client, container):
        """US-04: the session, not the client, holds the context."""
        first = client.post("/api/chat", json={"message": "GPA tối thiểu?"}).json()
        client.post(
            "/api/chat",
            json={"message": "Còn điểm F thì sao?", "session_id": first["session_id"]},
        )
        history = container.rag_service.answer.await_args.args[0].history
        assert [turn.content for turn in history] == [
            "GPA tối thiểu?",
            "GPA tối thiểu là 2.0/4.0.",
        ]

    def test_new_session_starts_without_history(self, client, container):
        client.post("/api/chat", json={"message": "câu hỏi một"})
        client.post("/api/chat", json={"message": "câu hỏi hai"})
        assert container.rag_service.answer.await_args.args[0].history == []

    def test_low_confidence_adds_the_advisor_disclaimer(self, client, container):
        container.rag_service.answer.return_value = make_response(band=ConfidenceBand.LOW)
        body = client.post("/api/chat", json={"message": "câu hỏi lạ"}).json()
        assert body["disclaimer"]
        assert "advisor" in body["disclaimer"].lower()

    def test_high_confidence_has_no_disclaimer(self, client):
        assert client.post("/api/chat", json={"message": "GPA?"}).json()["disclaimer"] is None

    def test_empty_context_is_reported(self, client, container):
        container.rag_service.answer.return_value = make_response(
            band=ConfidenceBand.LOW, empty=True
        )
        body = client.post("/api/chat", json={"message": "chuyện gì đó"}).json()
        assert body["is_empty_context"] is True
        assert body["sources"] == []

    @pytest.mark.parametrize(
        "error", [RetrievalError("zilliz down"), LLMUnavailableError("429")]
    )
    def test_backend_outage_becomes_503_without_leaking_internals(
        self, client, container, error
    ):
        container.rag_service.answer.side_effect = error
        response = client.post("/api/chat", json={"message": "GPA?"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "zilliz" not in detail.lower()
        assert "429" not in detail

    @pytest.mark.parametrize(
        "payload",
        [{}, {"message": ""}, {"message": "x" * 2001}, {"message": "ok", "domain": "NOPE"}],
    )
    def test_rejects_bad_input(self, client, payload):
        assert client.post("/api/chat", json=payload).status_code == 422


class TestSessions:
    def test_clear_forgets_the_history(self, client, container):
        first = client.post("/api/chat", json={"message": "GPA tối thiểu?"}).json()
        session_id = first["session_id"]

        assert client.delete(f"/api/sessions/{session_id}").status_code == 204

        client.post("/api/chat", json={"message": "tiếp theo", "session_id": session_id})
        assert container.rag_service.answer.await_args.args[0].history == []

    def test_clearing_an_unknown_session_is_fine(self, client):
        assert client.delete("/api/sessions/never-existed").status_code == 204


class TestDomains:
    def test_lists_the_five_regulatory_domains(self, client):
        domains = client.get("/api/domains").json()
        assert len(domains) == 5
        assert {d["value"] for d in domains} == {
            "COURSE_CURRICULUM",
            "GRADUATION",
            "SCHOLARSHIP",
            "DISCIPLINARY",
            "ADMISSION_ENROLLMENT",
        }

    def test_every_domain_is_labelled_in_both_languages(self, client):
        for domain in client.get("/api/domains").json():
            assert domain["label_vi"]
            assert domain["label_en"]


class TestHealth:
    def test_health_is_open(self, client):
        assert client.get("/api/health").json() == {"status": "ok"}
