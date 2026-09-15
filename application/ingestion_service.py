"""Document ingestion: crawl → change-detect → extract → chunk → classify → index.

Idempotent and resumable. A document whose content hash is unchanged is skipped
without re-embedding; a changed one has its old chunks deleted before the new
ones are written, so altering the chunk size cannot leave orphaned rows behind.
One failing document never aborts the run.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date

from domain.exceptions import EmbeddingError, PDFExtractionError
from domain.models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    ExtractionResult,
    IngestionRequest,
    IngestionResult,
    RegulationDomain,
    RegulationSource,
    RunStatus,
    RunTrigger,
    utc_now_iso,
)
from domain.ports import (
    ChunkerPort,
    DocumentRegistryPort,
    EmbedderPort,
    LLMPort,
    PDFDownloaderPort,
    PDFExtractorPort,
    ScraperPort,
    VectorStorePort,
)

logger = logging.getLogger(__name__)

_CLASSIFY_SAMPLE_CHARS = 2000


def doc_id_for(source_url: str) -> str:
    """Stable document identity. Chunk ids are '{doc_id}:{index}'."""
    return hashlib.sha1(source_url.encode("utf-8")).hexdigest()


class IngestionService:
    def __init__(
        self,
        scraper: ScraperPort,
        downloader: PDFDownloaderPort,
        extractor: PDFExtractorPort,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        registry: DocumentRegistryPort,
        *,
        delay_between_requests: float = 0.0,
    ) -> None:
        self._scraper = scraper
        self._downloader = downloader
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._registry = registry
        self._delay = delay_between_requests

    async def run(self, request: IngestionRequest) -> IngestionResult:
        run_id = self._registry.start_run(request.trigger)
        result = IngestionResult()

        try:
            await self._vector_store.ensure_collection()
            sources = _dedupe_by_link(await self._scraper.scrape())
        except Exception as exc:
            logger.error("Ingestion run aborted: %s", exc, exc_info=True)
            self._registry.finish_run(run_id, status=RunStatus.FAILED, error=str(exc))
            raise

        result.scraped_count = len(sources)
        if request.limit is not None:
            sources = sources[: request.limit]
        logger.info("Crawled %d documents, processing %d", result.scraped_count, len(sources))

        for source in sources:
            try:
                written = await self._process(source, only_new=request.only_new)
            except Exception as exc:
                logger.warning("Document failed | %r: %s", source.title, exc, exc_info=True)
                result.failed_titles.append(source.title)
                self._mark_failed(source.link, source.title)
            else:
                if written is None:
                    result.skipped_count += 1
                else:
                    result.changed_count += 1
                    result.indexed_chunks += written
            if self._delay:
                await asyncio.sleep(self._delay)

        self._registry.finish_run(
            run_id,
            status=RunStatus.COMPLETED,
            scraped_count=result.scraped_count,
            changed_count=result.changed_count,
            indexed_chunks=result.indexed_chunks,
        )
        logger.info(
            "Ingestion done | changed=%d skipped=%d chunks=%d failed=%d",
            result.changed_count,
            result.skipped_count,
            result.indexed_chunks,
            len(result.failed_titles),
        )
        return result

    async def ingest_pdf(
        self,
        *,
        source_url: str,
        title: str,
        pdf_bytes: bytes,
        domain: RegulationDomain | None = None,
        doc_updated_at: str = "",
    ) -> int:
        """Index one PDF supplied directly — the admin upload path (US-08)."""
        run_id = self._registry.start_run(RunTrigger.UPLOAD)
        try:
            await self._vector_store.ensure_collection()
            written = await self._index(
                source_url=source_url,
                title=title,
                pdf_bytes=pdf_bytes,
                content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
                domain=domain,
                doc_updated_at=doc_updated_at,
            )
        except Exception as exc:
            logger.error("Upload failed | %r: %s", title, exc, exc_info=True)
            self._mark_failed(source_url, title)
            self._registry.finish_run(run_id, status=RunStatus.FAILED, error=str(exc))
            raise

        self._registry.finish_run(
            run_id,
            status=RunStatus.COMPLETED,
            scraped_count=1,
            changed_count=1,
            indexed_chunks=written,
        )
        return written

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _process(self, source: RegulationSource, *, only_new: bool) -> int | None:
        """Index one crawled document. Returns None when nothing had changed."""
        pdf_bytes = await self._downloader.download(source.link)
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        existing = self._registry.get(source.link)

        unchanged = (
            only_new
            and existing is not None
            and existing.content_hash == content_hash
            and existing.status is DocumentStatus.INDEXED
        )
        if unchanged:
            logger.debug("Unchanged, skipping | %r", source.title)
            existing.last_crawled_at = utc_now_iso()
            self._registry.upsert(existing)
            return None

        return await self._index(
            source_url=source.link,
            title=source.title,
            pdf_bytes=pdf_bytes,
            content_hash=content_hash,
            domain=None,
        )

    async def _index(
        self,
        *,
        source_url: str,
        title: str,
        pdf_bytes: bytes,
        content_hash: str,
        domain: RegulationDomain | None,
        doc_updated_at: str = "",
    ) -> int:
        extraction = await self._extractor.extract(pdf_bytes, title)
        text_chunks = self._chunker.chunk(extraction.pages)
        if not text_chunks:
            raise PDFExtractionError(f"No extractable text in {title!r}")

        if domain is None:
            domain = await self._classify(title, extraction)

        doc_id = doc_id_for(source_url)
        # The content hash last changed today, which is the freshest "last
        # updated" the source actually gives us (US-07).
        updated_at = doc_updated_at or date.today().isoformat()

        chunks = [
            Chunk(
                chunk_id=f"{doc_id}:{index}",
                doc_id=doc_id,
                text=text_chunk.text,
                source_url=source_url,
                doc_title=title,
                domain=domain,
                chunk_index=index,
                page_start=text_chunk.page_start,
                page_end=text_chunk.page_end,
                section=text_chunk.section,
                doc_updated_at=updated_at,
            )
            for index, text_chunk in enumerate(text_chunks)
        ]

        embeddings = await self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise EmbeddingError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks"
            )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Replace, never merge: chunk boundaries shift when the document or the
        # chunk size changes, and stale rows would keep being retrieved.
        await self._vector_store.delete_by_doc_id(doc_id)
        written = await self._vector_store.upsert(chunks)

        now = utc_now_iso()
        self._registry.upsert(
            DocumentRecord(
                source_url=source_url,
                doc_id=doc_id,
                title=title,
                domain=domain,
                content_hash=content_hash,
                doc_updated_at=updated_at,
                extraction_method=extraction.method,
                chunk_count=written,
                last_crawled_at=now,
                last_indexed_at=now,
                status=DocumentStatus.INDEXED,
            )
        )
        logger.info("Indexed %d chunks | %r (%s)", written, title, domain.value)
        return written

    async def _classify(self, title: str, extraction: ExtractionResult) -> RegulationDomain:
        """Assign a domain. A classifier outage must not block indexing."""
        try:
            return await self._llm.classify_domain(
                title, extraction.full_text[:_CLASSIFY_SAMPLE_CHARS]
            )
        except Exception as exc:
            logger.warning("Domain classification failed for %r: %s", title, exc)
            return RegulationDomain.OTHER

    def _mark_failed(self, source_url: str, title: str) -> None:
        record = self._registry.get(source_url) or DocumentRecord(
            source_url=source_url, doc_id=doc_id_for(source_url), title=title
        )
        record.status = DocumentStatus.FAILED
        record.last_crawled_at = utc_now_iso()
        self._registry.upsert(record)


def _dedupe_by_link(sources: list[RegulationSource]) -> list[RegulationSource]:
    seen: dict[str, RegulationSource] = {}
    for source in sources:
        seen.setdefault(source.link, source)
    return list(seen.values())
