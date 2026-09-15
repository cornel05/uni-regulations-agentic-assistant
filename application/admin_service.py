"""Admin operations: index status, manual crawl, manual PDF upload.

Thin orchestration over the registry, the vector store, and the ingestion
pipeline — it exists so the HTTP layer holds no business logic.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from domain.models import IndexStatus, IngestionRequest, IngestionResult, RegulationDomain, RunTrigger
from domain.ports import DocumentRegistryPort, VectorStorePort

from .ingestion_service import IngestionService

logger = logging.getLogger(__name__)

_UPLOAD_SCHEME = "upload://"


class AdminService:
    def __init__(
        self,
        registry: DocumentRegistryPort,
        vector_store: VectorStorePort,
        ingestion: IngestionService,
    ) -> None:
        self._registry = registry
        self._vector_store = vector_store
        self._ingestion = ingestion

    async def status(self) -> IndexStatus:
        """Dashboard payload. The live chunk count comes from the vector store."""
        try:
            total_chunks = await self._vector_store.count()
        except Exception as exc:
            # The dashboard stays useful even when the index is unreachable.
            logger.warning("Could not read chunk count from the vector store: %s", exc)
            total_chunks = 0
        return self._registry.index_status(total_chunks)

    async def crawl(
        self,
        *,
        trigger: RunTrigger = RunTrigger.MANUAL,
        only_new: bool = True,
        limit: int | None = None,
    ) -> IngestionResult:
        return await self._ingestion.run(
            IngestionRequest(only_new=only_new, limit=limit, trigger=trigger)
        )

    async def upload(
        self,
        *,
        filename: str,
        title: str,
        pdf_bytes: bytes,
        domain: RegulationDomain | None = None,
    ) -> int:
        """Index an operator-supplied PDF immediately (US-08).

        Uploads get an ``upload://`` pseudo-URL as their identity, so re-uploading
        the same filename replaces that document rather than duplicating it.
        """
        source_url = upload_source_url(filename)
        logger.info("Admin upload | %r as %s", title, source_url)
        return await self._ingestion.ingest_pdf(
            source_url=source_url, title=title, pdf_bytes=pdf_bytes, domain=domain
        )


def upload_source_url(filename: str) -> str:
    """Identity of an uploaded document.

    Public because the HTTP layer needs the same value to read the resulting
    record back, and because re-uploading a filename must replace that document
    rather than create a second one.
    """
    return _UPLOAD_SCHEME + _slugify(filename)


def _slugify(filename: str) -> str:
    """Filesystem- and URL-safe stem, so a pseudo-URL stays stable and printable."""
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip() or "document.pdf"
    normalized = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip("-._")
    return slug or "document.pdf"
