"""SQLiteDocumentRegistry: round-trips, idempotency, dashboard aggregation."""

from __future__ import annotations

import sqlite3

import pytest

from domain.models import (
    DocumentRecord,
    DocumentStatus,
    ExtractionMethod,
    RegulationDomain,
    RunStatus,
    RunTrigger,
)
from infrastructure.registry.sqlite_registry import SQLiteDocumentRegistry


@pytest.fixture
def registry(tmp_path):
    return SQLiteDocumentRegistry(str(tmp_path / "nested" / "regulations.db"))


def make_record(**overrides) -> DocumentRecord:
    defaults = dict(
        source_url="https://drive.google.com/file/d/a/view",
        doc_id="aaa111",
        title="Quy chế đào tạo 2024",
        domain=RegulationDomain.COURSE_CURRICULUM,
        content_hash="hash-1",
        doc_updated_at="2026-09-01",
        extraction_method=ExtractionMethod.PYMUPDF,
        chunk_count=7,
        last_crawled_at="2026-09-01T10:00:00+07:00",
        last_indexed_at="2026-09-01T10:05:00+07:00",
        status=DocumentStatus.INDEXED,
    )
    return DocumentRecord(**{**defaults, **overrides})


class TestDocuments:
    def test_creates_its_own_directory_and_schema(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "db.sqlite"
        SQLiteDocumentRegistry(str(path))
        assert path.exists()

    def test_unknown_url_returns_none(self, registry):
        assert registry.get("https://nope") is None

    def test_round_trips_every_field(self, registry):
        record = make_record()
        registry.upsert(record)
        loaded = registry.get(record.source_url)
        assert loaded == record

    def test_upsert_is_idempotent_and_updates(self, registry):
        registry.upsert(make_record())
        registry.upsert(make_record(content_hash="hash-2", chunk_count=9))
        assert len(registry.list_documents()) == 1
        loaded = registry.get("https://drive.google.com/file/d/a/view")
        assert loaded.content_hash == "hash-2"
        assert loaded.chunk_count == 9

    def test_missing_extraction_method_round_trips_as_none(self, registry):
        registry.upsert(make_record(extraction_method=None))
        assert registry.get("https://drive.google.com/file/d/a/view").extraction_method is None

    def test_lists_documents_sorted_by_title(self, registry):
        registry.upsert(make_record(source_url="u1", doc_id="d1", title="Zeta"))
        registry.upsert(make_record(source_url="u2", doc_id="d2", title="alpha"))
        assert [d.title for d in registry.list_documents()] == ["alpha", "Zeta"]

    def test_unknown_stored_enum_does_not_crash(self, registry, tmp_path):
        """Stored strings are data — a stale value must degrade, not raise."""
        registry.upsert(make_record())
        with sqlite3.connect(registry._path) as connection:
            connection.execute("UPDATE documents SET domain = 'LEGACY_DOMAIN'")
        assert registry.get("https://drive.google.com/file/d/a/view").domain is (
            RegulationDomain.OTHER
        )


class TestRuns:
    def test_start_run_returns_increasing_ids(self, registry):
        first = registry.start_run(RunTrigger.MANUAL)
        second = registry.start_run(RunTrigger.SCHEDULE)
        assert second > first

    def test_finish_run_is_recorded(self, registry):
        run_id = registry.start_run(RunTrigger.SCHEDULE)
        registry.finish_run(
            run_id,
            status=RunStatus.COMPLETED,
            scraped_count=22,
            changed_count=3,
            indexed_chunks=140,
        )
        run = registry.index_status(0).recent_runs[0]
        assert run.id == run_id
        assert run.status is RunStatus.COMPLETED
        assert run.trigger is RunTrigger.SCHEDULE
        assert (run.scraped_count, run.changed_count, run.indexed_chunks) == (22, 3, 140)
        assert run.finished_at

    def test_failed_run_keeps_its_error(self, registry):
        run_id = registry.start_run(RunTrigger.MANUAL)
        registry.finish_run(run_id, status=RunStatus.FAILED, error="page did not load")
        assert registry.index_status(0).recent_runs[0].error == "page did not load"

    def test_recent_runs_are_newest_first_and_capped(self, registry):
        for _ in range(12):
            registry.finish_run(registry.start_run(RunTrigger.MANUAL), status=RunStatus.COMPLETED)
        runs = registry.index_status(0).recent_runs
        assert len(runs) == 10
        assert runs[0].id > runs[-1].id


class TestIndexStatus:
    def test_empty_registry_reports_zeroes(self, registry):
        status = registry.index_status(0)
        assert status.total_documents == 0
        assert status.documents_per_domain == {}
        assert status.last_crawled_at == ""

    def test_counts_by_status_and_domain(self, registry):
        registry.upsert(make_record(source_url="u1", doc_id="d1"))
        registry.upsert(
            make_record(
                source_url="u2",
                doc_id="d2",
                domain=RegulationDomain.SCHOLARSHIP,
                status=DocumentStatus.FAILED,
            )
        )
        registry.upsert(
            make_record(
                source_url="u3",
                doc_id="d3",
                domain=RegulationDomain.SCHOLARSHIP,
                status=DocumentStatus.INDEXED,
            )
        )
        status = registry.index_status(total_chunks=99)
        assert status.total_documents == 3
        assert status.indexed_documents == 2
        assert status.failed_documents == 1
        assert status.documents_per_domain == {"COURSE_CURRICULUM": 1, "SCHOLARSHIP": 2}

    def test_chunk_count_comes_from_the_caller(self, registry):
        """The vector store owns that number, not the registry."""
        registry.upsert(make_record())
        assert registry.index_status(total_chunks=1234).total_chunks == 1234

    def test_reports_latest_timestamps(self, registry):
        registry.upsert(make_record(source_url="u1", doc_id="d1", last_crawled_at="2026-01-01"))
        registry.upsert(make_record(source_url="u2", doc_id="d2", last_crawled_at="2026-09-09"))
        assert registry.index_status(0).last_crawled_at == "2026-09-09"
