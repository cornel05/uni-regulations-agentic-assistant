"""Dependency-inversion contracts for every external dependency.

Rules:
  1. ``application/`` imports only from this file and ``domain/models.py``.
  2. Adapters implement these Protocols structurally — they never inherit from
     them, so infrastructure keeps no dependency on domain classes.
  3. ``runtime_checkable`` enables isinstance() checks in tests.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    Chunk,
    DocumentRecord,
    ExtractionResult,
    GroundedAnswer,
    IndexStatus,
    PageText,
    RegulationDomain,
    RegulationSource,
    RetrievedChunk,
    RunStatus,
    RunTrigger,
    TextChunk,
    Turn,
)


@runtime_checkable
class EmbedderPort(Protocol):
    """Text to dense vectors. Query and document use different task types."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed one search string (task type ``RETRIEVAL_QUERY``).

        Raises:
            EmbeddingError: On API failure or quota exhaustion.
        """
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for indexing (task type ``RETRIEVAL_DOCUMENT``).

        Returns vectors in the same order as ``texts``.

        Raises:
            EmbeddingError: On API failure or quota exhaustion.
        """
        ...


@runtime_checkable
class RetrieverPort(Protocol):
    """Hybrid retrieval over the indexed chunks."""

    async def hybrid_search(
        self,
        query_text: str,
        query_embeddings: list[list[float]],
        top_k: int,
        domain: RegulationDomain | None = None,
    ) -> list[RetrievedChunk]:
        """Find the most relevant chunks, fusing dense and lexical matches.

        Args:
            query_text: Raw query, used for the lexical (BM25) branch.
            query_embeddings: One or more query vectors. More than one is used
                for cross-lingual search (e.g. the English query plus its
                Vietnamese translation); results are fused across all of them.
            top_k: Maximum number of chunks to return.
            domain: When set, restricts results to that regulatory domain.

        Returns:
            Chunks sorted by descending fused score. Empty list when nothing
            matches — not an exception.

        Raises:
            RetrievalError: On connection or query failure.
        """
        ...


@runtime_checkable
class VectorStorePort(Protocol):
    """Write side of the vector index."""

    async def ensure_collection(self) -> None:
        """Create the collection and its indexes if absent, then load it.

        Raises:
            VectorStoreError: On connection or schema failure.
        """
        ...

    async def upsert(self, chunks: list[Chunk]) -> int:
        """Insert or replace chunks. Every chunk must carry an embedding.

        Returns:
            Number of chunks written.

        Raises:
            VectorStoreError: On connection or schema violation.
        """
        ...

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove every chunk of one document.

        Called before re-indexing a changed document so that changing the chunk
        size cannot leave orphaned rows behind.

        Raises:
            VectorStoreError: On delete failure.
        """
        ...

    async def count(self) -> int:
        """Total number of indexed chunks.

        Raises:
            VectorStoreError: On query failure.
        """
        ...


@runtime_checkable
class LLMPort(Protocol):
    """Instruction-following generation."""

    async def answer(self, system_prompt: str, user_message: str) -> GroundedAnswer:
        """Generate a grounded answer as structured output.

        The adapter is responsible for requesting JSON and parsing it into a
        GroundedAnswer, so the application never sees raw model text.

        Raises:
            LLMUnavailableError: On API failure, quota exhaustion, or unparsable
                output.
        """
        ...

    async def classify_domain(self, title: str, sample_text: str) -> RegulationDomain:
        """Assign one regulatory domain to a document.

        Returns ``RegulationDomain.OTHER`` when the model is unsure; never raises
        for an unrecognized label.

        Raises:
            LLMUnavailableError: On API failure.
        """
        ...

    async def translate_to_vietnamese(self, text: str) -> str:
        """Translate a query into Vietnamese for cross-lingual retrieval.

        Raises:
            LLMUnavailableError: On API failure.
        """
        ...


@runtime_checkable
class ScraperPort(Protocol):
    """Crawler for the regulation listing page."""

    async def scrape(self) -> list[RegulationSource]:
        """Load the listing page and extract every document link.

        Returns an empty list when the page loads but lists nothing.

        Raises:
            ScraperError: If the page fails to load or the DOM is unrecognized.
        """
        ...


@runtime_checkable
class PDFDownloaderPort(Protocol):
    async def download(self, url: str) -> bytes:
        """Download raw PDF bytes, following Google Drive download flows.

        Raises:
            PDFDownloadError: On HTTP error, timeout, or a non-PDF response.
        """
        ...


@runtime_checkable
class PDFExtractorPort(Protocol):
    async def extract(self, pdf_bytes: bytes, title: str) -> ExtractionResult:
        """Extract per-page text, falling back to a multimodal model on poor quality.

        Raises:
            PDFExtractionError: If no usable text can be produced.
        """
        ...


@runtime_checkable
class ChunkerPort(Protocol):
    def chunk(self, pages: list[PageText]) -> list[TextChunk]:
        """Split page texts into overlapping chunks, preserving provenance.

        Each returned chunk carries the page range it spans and the nearest
        enclosing section heading, so answers can cite a page and a section.

        Args:
            pages: Per-page text, in document order.

        Returns:
            Ordered chunks. Empty list when the pages hold no text.
        """
        ...


@runtime_checkable
class DocumentRegistryPort(Protocol):
    """Durable record of what has been crawled and indexed."""

    def get(self, source_url: str) -> DocumentRecord | None:
        """Return the stored record for a URL, or None if unseen.

        Raises:
            RegistryError: On read failure.
        """
        ...

    def upsert(self, record: DocumentRecord) -> None:
        """Insert or replace a document record, keyed by ``source_url``.

        Raises:
            RegistryError: On write failure.
        """
        ...

    def list_documents(self) -> list[DocumentRecord]:
        """Every known document.

        Raises:
            RegistryError: On read failure.
        """
        ...

    def index_status(self, total_chunks: int) -> IndexStatus:
        """Aggregate view for the admin dashboard.

        Args:
            total_chunks: Live chunk count from the vector store, which owns that
                number; the registry only knows its own per-document totals.

        Raises:
            RegistryError: On read failure.
        """
        ...

    def start_run(self, trigger: RunTrigger) -> int:
        """Record the start of an ingestion run and return its id.

        Raises:
            RegistryError: On write failure.
        """
        ...

    def finish_run(
        self,
        run_id: int,
        status: RunStatus,
        scraped_count: int = 0,
        changed_count: int = 0,
        indexed_chunks: int = 0,
        error: str = "",
    ) -> None:
        """Close out an ingestion run.

        Raises:
            RegistryError: On write failure.
        """
        ...


@runtime_checkable
class SessionStorePort(Protocol):
    """Multi-turn conversation context with a TTL."""

    def create(self) -> str:
        """Start a session and return its id."""
        ...

    def history(self, session_id: str) -> list[Turn]:
        """Turns for a session, oldest first. Empty for unknown or expired ids."""
        ...

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Record one exchange, trimming to the configured turn limit."""
        ...

    def clear(self, session_id: str) -> None:
        """Forget a session's history. Idempotent."""
        ...
