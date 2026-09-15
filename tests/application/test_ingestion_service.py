"""Behavioral contract for IngestionService.

Coverage:
  happy path   — crawl to indexed, registry updated, counts reported
  change detection — unchanged content hash is skipped without re-embedding
  re-index     — a changed document's old chunks are deleted before upsert
  isolation    — one bad document does not abort the run
  run auditing — every run is opened and closed in the registry
  identity     — doc_id / chunk_id scheme, domain classification
  upload       — admin PDF path skips the crawler
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.ingestion_service import IngestionService
from domain.exceptions import PDFDownloadError, ScraperError
from domain.models import (
    DocumentStatus,
    IngestionRequest,
    RegulationDomain,
    RunStatus,
    RunTrigger,
)


def make_service(
    mock_scraper,
    mock_downloader,
    mock_extractor,
    mock_chunker,
    mock_embedder,
    mock_vector_store,
    mock_llm,
    mock_registry,
) -> IngestionService:
    return IngestionService(
        scraper=mock_scraper,
        downloader=mock_downloader,
        extractor=mock_extractor,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm=mock_llm,
        registry=mock_registry,
    )


@pytest.fixture
def service(
    mock_scraper,
    mock_downloader,
    mock_extractor,
    mock_chunker,
    mock_embedder,
    mock_vector_store,
    mock_llm,
    mock_registry,
):
    return make_service(
        mock_scraper,
        mock_downloader,
        mock_extractor,
        mock_chunker,
        mock_embedder,
        mock_vector_store,
        mock_llm,
        mock_registry,
    )


# ── Happy path ────────────────────────────────────────────────────────────────


class TestHappyPath:
    async def test_indexes_every_scraped_document(self, service, sample_sources):
        result = await service.run(IngestionRequest())
        assert result.scraped_count == len(sample_sources)
        assert result.changed_count == len(sample_sources)
        assert result.failed_titles == []

    async def test_counts_indexed_chunks(self, service, sample_text_chunks, sample_sources):
        result = await service.run(IngestionRequest())
        assert result.indexed_chunks == len(sample_text_chunks) * len(sample_sources)

    async def test_registry_marks_documents_indexed(
        self, service, mock_registry, sample_sources
    ):
        await service.run(IngestionRequest())
        for source in sample_sources:
            record = mock_registry.get(source.link)
            assert record is not None
            assert record.status is DocumentStatus.INDEXED
            assert record.content_hash
            assert record.chunk_count == 2
            assert record.last_indexed_at

    async def test_embeddings_are_attached_before_upsert(self, service, mock_vector_store):
        await service.run(IngestionRequest())
        chunks = mock_vector_store.upsert.await_args.args[0]
        assert chunks
        assert all(chunk.embedding for chunk in chunks)

    async def test_collection_is_ensured_once(self, service, mock_vector_store):
        await service.run(IngestionRequest())
        mock_vector_store.ensure_collection.assert_awaited_once()


# ── Change detection ──────────────────────────────────────────────────────────


class TestChangeDetection:
    async def test_unchanged_document_is_skipped(
        self, service, mock_registry, mock_embedder, mock_vector_store, indexed_record,
        sample_sources, mock_downloader,
    ):
        """Same content hash plus indexed status means no work to redo."""
        for source in sample_sources:
            pdf = await mock_downloader.download(source.link)
            digest = hashlib.sha256(pdf).hexdigest()
            mock_registry.upsert(indexed_record(source.link, digest))
        mock_embedder.embed_documents.reset_mock()
        mock_vector_store.upsert.reset_mock()

        result = await service.run(IngestionRequest(only_new=True))

        assert result.skipped_count == len(sample_sources)
        assert result.changed_count == 0
        mock_embedder.embed_documents.assert_not_awaited()
        mock_vector_store.upsert.assert_not_awaited()

    async def test_changed_hash_triggers_reindex(
        self, service, mock_registry, mock_vector_store, indexed_record, sample_sources
    ):
        for source in sample_sources:
            mock_registry.upsert(indexed_record(source.link, "stale-hash"))
        result = await service.run(IngestionRequest(only_new=True))
        assert result.changed_count == len(sample_sources)
        mock_vector_store.upsert.assert_awaited()

    async def test_only_new_false_reindexes_everything(
        self, service, mock_registry, indexed_record, sample_sources, mock_downloader
    ):
        for source in sample_sources:
            pdf = await mock_downloader.download(source.link)
            digest = hashlib.sha256(pdf).hexdigest()
            mock_registry.upsert(indexed_record(source.link, digest))
        result = await service.run(IngestionRequest(only_new=False))
        assert result.skipped_count == 0
        assert result.changed_count == len(sample_sources)

    async def test_previously_failed_document_is_retried(
        self, service, mock_registry, indexed_record, sample_sources, mock_downloader
    ):
        """A matching hash must not mask a document that never finished indexing."""
        for source in sample_sources:
            pdf = await mock_downloader.download(source.link)
            record = indexed_record(source.link, hashlib.sha256(pdf).hexdigest())
            record.status = DocumentStatus.FAILED
            mock_registry.upsert(record)
        result = await service.run(IngestionRequest(only_new=True))
        assert result.changed_count == len(sample_sources)
        assert result.skipped_count == 0


# ── Re-index hygiene ──────────────────────────────────────────────────────────


class TestReindexHygiene:
    async def test_old_chunks_deleted_before_upsert(self, service, mock_vector_store):
        """Otherwise a smaller chunk count leaves orphaned rows behind."""
        order: list[str] = []
        mock_vector_store.delete_by_doc_id.side_effect = lambda doc_id: order.append("delete")
        mock_vector_store.upsert.side_effect = lambda chunks: (
            order.append("upsert") or len(chunks)
        )
        await service.run(IngestionRequest())
        assert order[:2] == ["delete", "upsert"]

    async def test_delete_targets_the_documents_own_id(
        self, service, mock_vector_store, sample_sources
    ):
        await service.run(IngestionRequest())
        deleted = {
            call.args[0] if call.args else call.kwargs["doc_id"]
            for call in mock_vector_store.delete_by_doc_id.await_args_list
        }
        expected = {
            hashlib.sha1(source.link.encode("utf-8")).hexdigest() for source in sample_sources
        }
        assert deleted == expected


# ── Identity and metadata ─────────────────────────────────────────────────────


class TestIdentity:
    async def test_chunk_ids_follow_doc_id_index_scheme(
        self, service, mock_vector_store, sample_sources
    ):
        await service.run(IngestionRequest())
        chunks = mock_vector_store.upsert.await_args_list[0].args[0]
        doc_id = hashlib.sha1(sample_sources[0].link.encode("utf-8")).hexdigest()
        assert [c.chunk_id for c in chunks] == [f"{doc_id}:0", f"{doc_id}:1"]
        assert all(chunk.doc_id == doc_id for chunk in chunks)

    async def test_doc_title_is_carried_onto_every_chunk(
        self, service, mock_vector_store, sample_sources
    ):
        """The legacy bug: chunks were indexed with an empty doc_title."""
        await service.run(IngestionRequest())
        chunks = mock_vector_store.upsert.await_args_list[0].args[0]
        assert all(chunk.doc_title == sample_sources[0].title for chunk in chunks)

    async def test_page_and_section_are_preserved(self, service, mock_vector_store):
        await service.run(IngestionRequest())
        chunks = mock_vector_store.upsert.await_args_list[0].args[0]
        assert [c.page_start for c in chunks] == [1, 2]
        assert [c.section for c in chunks] == ["Điều 1", "Điều 2"]

    async def test_domain_classified_once_per_document(
        self, service, mock_llm, mock_vector_store, sample_sources
    ):
        await service.run(IngestionRequest())
        assert mock_llm.classify_domain.await_count == len(sample_sources)
        chunks = mock_vector_store.upsert.await_args_list[0].args[0]
        assert all(c.domain is RegulationDomain.COURSE_CURRICULUM for c in chunks)

    async def test_classification_failure_falls_back_to_other(
        self, service, mock_llm, mock_vector_store
    ):
        mock_llm.classify_domain.side_effect = RuntimeError("429")
        result = await service.run(IngestionRequest())
        assert result.failed_titles == []
        chunks = mock_vector_store.upsert.await_args_list[0].args[0]
        assert all(c.domain is RegulationDomain.OTHER for c in chunks)


# ── Failure isolation ─────────────────────────────────────────────────────────


class TestFailureIsolation:
    async def test_one_bad_download_does_not_abort_the_run(
        self, service, mock_downloader, sample_sources
    ):
        def download(url: str) -> bytes:
            if url == sample_sources[0].link:
                raise PDFDownloadError("403 Forbidden")
            return b"%PDF-1.4 ok"

        mock_downloader.download.side_effect = download
        result = await service.run(IngestionRequest())
        assert result.failed_titles == [sample_sources[0].title]
        assert result.changed_count == 1

    async def test_failed_document_is_recorded_as_failed(
        self, service, mock_downloader, mock_registry, sample_sources
    ):
        mock_downloader.download.side_effect = PDFDownloadError("403")
        await service.run(IngestionRequest())
        for source in sample_sources:
            assert mock_registry.get(source.link).status is DocumentStatus.FAILED

    async def test_document_with_no_extractable_text_fails_cleanly(
        self, service, mock_chunker, sample_sources
    ):
        mock_chunker.chunk.return_value = []
        result = await service.run(IngestionRequest())
        assert sorted(result.failed_titles) == sorted(s.title for s in sample_sources)
        assert result.indexed_chunks == 0

    async def test_scraper_failure_propagates(self, service, mock_scraper):
        mock_scraper.scrape.side_effect = ScraperError("page did not load")
        with pytest.raises(ScraperError):
            await service.run(IngestionRequest())

    async def test_scraper_failure_closes_the_run_as_failed(
        self, service, mock_scraper, mock_registry
    ):
        mock_scraper.scrape.side_effect = ScraperError("page did not load")
        with pytest.raises(ScraperError):
            await service.run(IngestionRequest())
        assert mock_registry.finish_run.call_args.kwargs["status"] is RunStatus.FAILED


# ── Run auditing ──────────────────────────────────────────────────────────────


class TestRunAuditing:
    async def test_run_is_opened_with_its_trigger(self, service, mock_registry):
        await service.run(IngestionRequest(trigger=RunTrigger.SCHEDULE))
        assert mock_registry.start_run.call_args.args[0] is RunTrigger.SCHEDULE

    async def test_run_is_closed_as_completed(self, service, mock_registry):
        await service.run(IngestionRequest())
        kwargs = mock_registry.finish_run.call_args.kwargs
        assert kwargs["status"] is RunStatus.COMPLETED
        assert kwargs["indexed_chunks"] == 4

    async def test_limit_caps_documents_processed(self, service, mock_registry):
        result = await service.run(IngestionRequest(limit=1))
        assert result.changed_count == 1


# ── Admin upload ──────────────────────────────────────────────────────────────


class TestUpload:
    async def test_upload_indexes_without_crawling(
        self, service, mock_scraper, mock_vector_store, mock_registry
    ):
        count = await service.ingest_pdf(
            source_url="upload://quy-dinh-hoc-bong.pdf",
            title="Quy định học bổng 2026",
            pdf_bytes=b"%PDF-1.4 upload",
            domain=RegulationDomain.SCHOLARSHIP,
        )
        assert count == 2
        mock_scraper.scrape.assert_not_awaited()
        record = mock_registry.get("upload://quy-dinh-hoc-bong.pdf")
        assert record.status is DocumentStatus.INDEXED
        assert record.domain is RegulationDomain.SCHOLARSHIP

    async def test_explicit_domain_skips_classification(self, service, mock_llm):
        await service.ingest_pdf(
            source_url="upload://a.pdf",
            title="A",
            pdf_bytes=b"%PDF-1.4",
            domain=RegulationDomain.DISCIPLINARY,
        )
        mock_llm.classify_domain.assert_not_awaited()

    async def test_upload_without_domain_is_classified(self, service, mock_llm):
        await service.ingest_pdf(
            source_url="upload://b.pdf", title="B", pdf_bytes=b"%PDF-1.4", domain=None
        )
        mock_llm.classify_domain.assert_awaited_once()
