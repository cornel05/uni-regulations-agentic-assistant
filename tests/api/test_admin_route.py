"""HTTP contract for the admin endpoints, including the auth boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from domain.exceptions import ScraperError
from domain.models import (
    DocumentRecord,
    DocumentStatus,
    IndexStatus,
    IngestionResult,
    RegulationDomain,
)
from infrastructure.sessions.memory_store import MemorySessionStore
from presentation.api.app import create_app
from presentation.api.deps import Container

TOKEN = "secret-admin-token"
AUTH = {"X-Admin-Token": TOKEN}
PDF = b"%PDF-1.7\ntest\n%%EOF"

ADMIN_ROUTES = [
    ("get", "/api/admin/status"),
    ("get", "/api/admin/documents"),
    ("post", "/api/admin/crawl"),
]


def build_container(admin_token: str = TOKEN) -> Container:
    settings = Settings(
        gemini_api_key="test",
        zilliz_cloud_endpoint="https://example.zillizcloud.com",
        zilliz_cloud_api_key="test",
        admin_token=admin_token,
    )
    admin_service = AsyncMock()
    admin_service.status.return_value = IndexStatus(
        total_documents=3,
        indexed_documents=2,
        failed_documents=1,
        total_chunks=120,
        documents_per_domain={"SCHOLARSHIP": 2, "GRADUATION": 1},
        last_crawled_at="2026-09-10T08:00:00+07:00",
    )
    admin_service.crawl.return_value = IngestionResult(
        scraped_count=22, changed_count=2, skipped_count=20, indexed_chunks=48
    )
    admin_service.upload.return_value = 7

    registry = MagicMock()
    registry.list_documents.return_value = [
        DocumentRecord(
            source_url="https://drive.google.com/file/d/a/view",
            doc_id="aaa",
            title="Quy định học bổng",
            domain=RegulationDomain.SCHOLARSHIP,
            chunk_count=7,
            status=DocumentStatus.INDEXED,
        )
    ]
    registry.get.return_value = DocumentRecord(
        source_url="upload://quy-dinh.pdf",
        doc_id="bbb",
        title="Quy định học bổng 2026",
        domain=RegulationDomain.SCHOLARSHIP,
        chunk_count=7,
        status=DocumentStatus.INDEXED,
    )

    return Container(
        settings=settings,
        rag_service=AsyncMock(),
        ingestion_service=AsyncMock(),
        admin_service=admin_service,
        sessions=MemorySessionStore(),
        vector_store=AsyncMock(),
        registry=registry,
    )


@pytest.fixture
def container():
    return build_container()


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as test_client:
        yield test_client


class TestAuthBoundary:
    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_rejects_a_missing_token(self, client, method, path):
        assert getattr(client, method)(path).status_code == 401

    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_rejects_a_wrong_token(self, client, method, path):
        response = getattr(client, method)(path, headers={"X-Admin-Token": "guess"})
        assert response.status_code == 401

    def test_accepts_the_configured_token(self, client):
        assert client.get("/api/admin/status", headers=AUTH).status_code == 200

    def test_unset_token_disables_the_admin_api(self):
        """Fails closed: no configured secret means unavailable, not open."""
        with TestClient(create_app(build_container(admin_token=""))) as client:
            for headers in ({}, AUTH):
                assert client.get("/api/admin/status", headers=headers).status_code == 503

    def test_upload_is_also_guarded(self, client):
        response = client.post(
            "/api/admin/documents",
            files={"file": ("a.pdf", PDF, "application/pdf")},
            data={"title": "A"},
        )
        assert response.status_code == 401


class TestStatus:
    def test_reports_the_dashboard_payload(self, client):
        body = client.get("/api/admin/status", headers=AUTH).json()
        assert body["total_documents"] == 3
        assert body["total_chunks"] == 120
        assert body["documents_per_domain"] == {"SCHOLARSHIP": 2, "GRADUATION": 1}
        assert body["last_crawled_at"]

    def test_lists_documents(self, client):
        body = client.get("/api/admin/documents", headers=AUTH).json()
        assert body[0]["title"] == "Quy định học bổng"
        assert body[0]["status"] == "indexed"


class TestCrawl:
    def test_triggers_a_run(self, client, container):
        body = client.post("/api/admin/crawl", headers=AUTH).json()
        assert body["changed_count"] == 2
        assert body["indexed_chunks"] == 48
        assert container.admin_service.crawl.await_args.kwargs["only_new"] is True

    def test_forwards_the_query_options(self, client, container):
        client.post("/api/admin/crawl?only_new=false&limit=3", headers=AUTH)
        kwargs = container.admin_service.crawl.await_args.kwargs
        assert kwargs["only_new"] is False
        assert kwargs["limit"] == 3

    def test_crawl_failure_is_a_bad_gateway(self, client, container):
        container.admin_service.crawl.side_effect = ScraperError("page did not load")
        assert client.post("/api/admin/crawl", headers=AUTH).status_code == 502


class TestUpload:
    def test_indexes_a_pdf(self, client, container):
        response = client.post(
            "/api/admin/documents",
            headers=AUTH,
            files={"file": ("quy-dinh.pdf", PDF, "application/pdf")},
            data={"title": "Quy định học bổng 2026", "domain": "SCHOLARSHIP"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["indexed_chunks"] == 7
        assert body["domain"] == "SCHOLARSHIP"
        kwargs = container.admin_service.upload.await_args.kwargs
        assert kwargs["title"] == "Quy định học bổng 2026"
        assert kwargs["domain"] is RegulationDomain.SCHOLARSHIP

    def test_domain_is_optional(self, client, container):
        client.post(
            "/api/admin/documents",
            headers=AUTH,
            files={"file": ("a.pdf", PDF, "application/pdf")},
            data={"title": "Không rõ lĩnh vực"},
        )
        assert container.admin_service.upload.await_args.kwargs["domain"] is None

    def test_rejects_a_non_pdf_whatever_it_claims(self, client, container):
        """The bytes decide, not the filename or the declared content type."""
        response = client.post(
            "/api/admin/documents",
            headers=AUTH,
            files={"file": ("evil.pdf", b"<html>not a pdf</html>", "application/pdf")},
            data={"title": "Fake"},
        )
        assert response.status_code == 415
        container.admin_service.upload.assert_not_awaited()

    def test_rejects_an_empty_file(self, client):
        response = client.post(
            "/api/admin/documents",
            headers=AUTH,
            files={"file": ("empty.pdf", b"", "application/pdf")},
            data={"title": "Empty"},
        )
        assert response.status_code == 400

    def test_requires_a_title(self, client):
        response = client.post(
            "/api/admin/documents",
            headers=AUTH,
            files={"file": ("a.pdf", PDF, "application/pdf")},
        )
        assert response.status_code == 422
