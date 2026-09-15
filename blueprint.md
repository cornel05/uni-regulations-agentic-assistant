# HCMUT Regulation Bot — Complete Top-Down Rebuild Blueprint

---

## Table of Contents

1. [Part 0: Architecture Overview](#part-0-architecture-overview)
2. [Part 1 (Step 1): Interfaces & Contracts](#part-1-step-1-interfaces--contracts)
3. [Part 2 (Step 2): Test-Driven Development](#part-2-step-2-test-driven-development)
4. [Part 3 (Step 3): Core Logic Implementation](#part-3-step-3-core-logic-implementation)
5. [Part 4 (Step 4): Production Ready](#part-4-step-4-production-ready)

---

## Part 0: Architecture Overview

### Guiding Principles

The existing codebase has three structural problems this rebuild fixes:

| Problem | Fix |
|---|---|
| Scraping + Discord + LLM calls tangled in one file | Strict layer separation via Ports (Dependency Inversion Principle) |
| Two competing RAG implementations (`rag_engine.py` vs `rag_engine_langchain.py`) | One canonical `RAGService` in the application layer |
| No tests, `print()` everywhere, brittle error handling | TDD-first, `logging` module, domain exceptions |

### Dependency Flow

```
presentation/       ──▶  application/    ──▶  domain/ports.py
(Discord Bot)            (RAGService)         (Protocols only)
                                                    ▲
infrastructure/     ─────────────────────────────────
(Gemini, Milvus)    (implements the Protocols)
```

**Critical rule**: `application/` imports **only** from `domain/`. It never touches `infrastructure/` or `presentation/`. All wiring happens in `main.py` (the Composition Root).

### Final Folder Structure

```
hcmut_reg_bot/
├── config/
│   └── settings.py              # Pydantic BaseSettings — single source of truth
├── domain/
│   ├── models.py                # Pydantic schemas: shared language between all layers
│   ├── exceptions.py            # Custom domain exception hierarchy
│   └── ports.py                 # typing.Protocol contracts for every external dependency
├── application/
│   ├── rag_service.py           # Core RAG orchestration (agnostic of Discord & DB)
│   ├── monitor_service.py       # Regulation monitoring pipeline
│   └── ingestion_service.py     # Document crawl → embed → index pipeline
├── infrastructure/
│   ├── llm/
│   │   └── gemini_llm.py        # Concrete: Google Gemini LLM adapter
│   ├── embeddings/
│   │   └── gemini_embedder.py   # Concrete: Google Gemini Embedder adapter
│   ├── vector_db/
│   │   └── milvus_adapter.py    # Concrete: Milvus/Zilliz adapter
│   ├── scrapers/
│   │   └── hcmut_scraper.py     # Concrete: Selenium scraper
│   ├── pdf/
│   │   ├── pymupdf_extractor.py # Concrete: PyMuPDF text extractor
│   │   └── gemini_extractor.py  # Concrete: Gemini PDF extractor (fallback)
│   ├── notifiers/
│   │   └── discord_webhook.py   # Concrete: Discord webhook sender
│   └── cache/
│       └── jsonl_cache.py       # Concrete: JSONL file cache + state tracker
├── presentation/
│   └── discord_bot/
│       ├── bot.py               # Discord bot factory
│       └── cogs/
│           └── rag_cog.py       # !ask command handler
├── tests/
│   ├── conftest.py
│   └── application/
│       └── test_rag_service.py
└── main.py                      # Composition root — wires all dependencies
```

---

## Part 1 (Step 1): Interfaces & Contracts

> **Rule**: No business logic. Every function body is `pass` or `raise NotImplementedError`.
> These files define the *language* of the system — the contracts every layer must honor.

---

### `config/settings.py`

```python
"""
Centralized application configuration loaded from environment variables / .env file.
Single source of truth — no other module reads os.environ directly.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    discord_bot_token: str = Field(..., description="Discord bot token")
    discord_webhook_url: str | None = Field(
        None, description="Discord webhook URL for Phase 1 monitor alerts"
    )

    # ── LLM Configuration ─────────────────────────────────────────────────────
    llm_model: str = Field("gemini-2.0-flash", description="Primary generation model")
    fallback_model: str = Field("gemini-2.5-flash", description="Fallback on quota error")
    embedding_model: str = Field("models/gemini-embedding-001")
    llm_temperature: float = Field(0.2, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(3072, ge=256, le=8192)

    # ── Vector Database ───────────────────────────────────────────────────────
    milvus_uri: str = Field(
        "http://localhost:19530",
        description="Milvus local URI or Zilliz Cloud endpoint",
    )
    milvus_api_key: str | None = Field(None, description="Zilliz Cloud API key")
    milvus_collection_name: str = Field("hcmut_regulations")

    # ── RAG Parameters ────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(5, ge=1, le=20)

    # ── Document Processing ───────────────────────────────────────────────────
    chunk_size: int = Field(1200, ge=200, le=4000)
    chunk_overlap: int = Field(200, ge=0)
    extraction_mode: str = Field(
        "hybrid",
        description="PDF extraction strategy: 'pymupdf' | 'gemini' | 'hybrid' | 'pymupdf_only'",
    )
    pymupdf_quality_threshold: float = Field(0.75, ge=0.0, le=1.0)

    # ── Cache & State ─────────────────────────────────────────────────────────
    cache_dir: str = Field("cache")
    crawled_cache_file: str = Field("cache/processed_records.jsonl")
    processor_state_file: str = Field("processor_seen_laws.json")

    # ── Phase 1 Monitor ───────────────────────────────────────────────────────
    regulation_page_url: str = Field("https://hcmut.edu.vn/dao-tao/quy-che-quy-dinh")
    monitor_state_file: str = Field("seen_laws.json")
    headless_mode: bool = Field(True)
    wait_time: int = Field(8, description="Seconds to wait for JS on regulation page")
    delay_between_requests: int = Field(5)
    pdf_download_timeout: int = Field(60)

    # ── Embedding Retry ───────────────────────────────────────────────────────
    embedding_retry_count: int = Field(3)
    embedding_retry_delay: float = Field(1.5, description="Seconds between retry attempts")
```

---

### `domain/models.py`

```python
"""
Canonical data schemas for the HCMUT Regulation Bot.

These Pydantic models are the *shared language* between all layers.
No layer imports from another layer's internals — only from here.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class ExtractionMethod(str, Enum):
    """Strategy used to extract text from a PDF."""
    PYMUPDF = "pymupdf"
    GEMINI = "gemini"
    HYBRID = "hybrid"


# ── Phase 1: Monitor ───────────────────────────────────────────────────────────

class RegulationSource(BaseModel):
    """A single regulation entry scraped from the HCMUT website."""
    title: str = Field(..., description="Human-readable title of the regulation document")
    link: str = Field(..., description="Google Drive or direct URL to the PDF")


class SummaryRequest(BaseModel):
    """Input for the PDF summarization port."""
    pdf_bytes: bytes = Field(..., description="Raw PDF binary content")
    title: str = Field(..., description="Document title for context in the prompt")


class SummaryResult(BaseModel):
    """Output of the PDF summarization port."""
    summary: str = Field(..., description="AI-generated summary text (max 4000 chars)")
    model_used: str = Field(..., description="Name of the model that generated this summary")


class NotificationPayload(BaseModel):
    """Data required to send a Discord alert embed for a new regulation."""
    title: str
    link: str
    summary: str = Field(..., max_length=4000)
    color: int = Field(0x003F87, description="Embed sidebar color (HCMUT blue)")


class MonitorResult(BaseModel):
    """Outcome of a single monitoring cycle run."""
    new_count: int = Field(..., description="Total new regulations discovered")
    notified_count: int = Field(..., description="Regulations successfully notified via Discord")
    failed_titles: list[str] = Field(
        default_factory=list,
        description="Titles of regulations that failed to process",
    )


# ── Phase 2: Ingestion ────────────────────────────────────────────────────────

class ProcessedChunk(BaseModel):
    """A single text chunk extracted from a regulation document, ready for vector indexing."""
    chunk_id: str = Field(..., description="Unique ID in format '{doc_id}:{chunk_index}'")
    text: str = Field(..., description="The chunk's text content")
    doc_id: str = Field(..., description="SHA-1 hash of the source URL")
    source_url: str
    doc_title: str
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=1)
    extraction_method: ExtractionMethod


class VectorRecord(BaseModel):
    """A record ready to be upserted into the vector database."""
    id: str = Field(..., description="Primary key — matches chunk_id")
    text: str
    embedding: list[float] | None = Field(
        None, description="Populated just before upsert; None when cached to disk"
    )
    source_url: str
    doc_title: str
    chunk_id: str


class IngestionRequest(BaseModel):
    """Configuration for a single ingestion pipeline run."""
    only_new: bool = Field(True, description="Skip already-indexed documents")
    force_reindex: bool = Field(False, description="Ignore cache and re-embed everything")


class IngestionResult(BaseModel):
    """Outcome of a single ingestion pipeline run."""
    scraped_count: int
    processed_count: int
    indexed_count: int
    failed_titles: list[str] = Field(default_factory=list)


# ── Phase 2: RAG ──────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned from the vector similarity search with its relevance score."""
    text: str
    source_url: str
    doc_title: str
    score: float = Field(..., description="Similarity score (higher = more relevant)")
    chunk_id: str


class RAGRequest(BaseModel):
    """Input to the RAG service."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class RAGResponse(BaseModel):
    """Complete output of the RAG pipeline."""
    query: str = Field(..., description="The original user query, echoed back")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved context")
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Context chunks used to generate the answer",
    )
    is_empty_context: bool = Field(
        False,
        description="True if no relevant chunks were found; answer is a polite decline",
    )
```

---

### `domain/exceptions.py`

```python
"""
Custom exception hierarchy for the HCMUT Regulation Bot.

Design rule:
  - Every layer catches bare exceptions from external libraries and
    re-raises one of these domain exceptions using `raise X from exc`.
  - The presentation layer (Discord bot) catches domain exceptions
    and maps them to user-friendly Vietnamese messages.
  - Never catch HCMUTBotError directly in infrastructure — let it propagate.
"""
from __future__ import annotations


class HCMUTBotError(Exception):
    """Base class for all application-level errors. Never raise this directly."""


# ── Infrastructure Errors ─────────────────────────────────────────────────────

class EmbeddingError(HCMUTBotError):
    """
    Raised when the embedding service fails (quota exceeded, network timeout, etc.).
    Callers: RAGService, IngestionService
    """


class LLMUnavailableError(HCMUTBotError):
    """
    Raised when the LLM is unavailable, rate-limited, or returns an empty response.
    Callers: RAGService, MonitorService
    """


class RetrievalError(HCMUTBotError):
    """
    Raised when the vector database search fails.
    Callers: RAGService
    """


class VectorStoreError(HCMUTBotError):
    """
    Raised when an upsert or index operation on the vector database fails.
    Callers: IngestionService
    """


class PDFDownloadError(HCMUTBotError):
    """
    Raised when a PDF cannot be fetched (HTTP 4xx/5xx, invalid Google Drive link, timeout).
    Callers: MonitorService, IngestionService
    """


class PDFExtractionError(HCMUTBotError):
    """
    Raised when text cannot be extracted from a PDF.
    Callers: IngestionService
    """


class ScraperError(HCMUTBotError):
    """
    Raised when the Selenium scraper cannot load or parse the regulation page.
    Callers: MonitorService, IngestionService
    """


class NotificationError(HCMUTBotError):
    """
    Raised when a Discord webhook or API call fails to deliver a notification.
    Callers: MonitorService
    """


class CacheError(HCMUTBotError):
    """
    Raised when a cache read or write operation fails (corrupted JSONL, disk full, etc.).
    Callers: IngestionService
    """


class ConfigurationError(HCMUTBotError):
    """
    Raised at startup if required environment variables are missing or invalid.
    Callers: main.py composition root
    """
```

---

### `domain/ports.py`

```python
"""
Dependency Inversion contracts for every external dependency.

Design rules:
  1. The application layer imports ONLY from this file (plus domain/models.py).
  2. Infrastructure adapters implement these Protocols structurally —
     they never inherit from them (no coupling).
  3. runtime_checkable enables isinstance() checks in tests.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    ExtractionMethod,
    NotificationPayload,
    ProcessedChunk,
    RegulationSource,
    RetrievedChunk,
    SummaryRequest,
    SummaryResult,
    VectorRecord,
)


@runtime_checkable
class EmbedderPort(Protocol):
    """Contract for converting text into dense vector representations."""

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Uses task_type='retrieval_query' internally — this produces a different
        vector space than embed_documents, improving asymmetric retrieval accuracy.

        Args:
            text: The user's question or search string.

        Returns:
            A dense float vector (e.g., 3072-dim for gemini-embedding-001).

        Raises:
            EmbeddingError: On API failure or quota exhaustion.
        """
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document chunks for indexing.

        Uses task_type='retrieval_document' internally.

        Args:
            texts: List of chunk texts to embed.

        Returns:
            List of float vectors, one per input text, in the same order.

        Raises:
            EmbeddingError: On API failure or quota exhaustion.
        """
        ...


@runtime_checkable
class RetrieverPort(Protocol):
    """Contract for fetching semantically similar chunks from the vector database."""

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Find the top-k most relevant chunks for a given query vector.

        Args:
            query_embedding: The dense float vector of the user's query.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk sorted by descending relevance score.
            Returns an empty list (not an exception) when no results are found.

        Raises:
            RetrievalError: On database connection failure or query error.
        """
        ...


@runtime_checkable
class VectorStorePort(Protocol):
    """Contract for writing vector records to the database."""

    async def upsert(self, records: list[VectorRecord]) -> int:
        """
        Insert or update a batch of vector records.

        Args:
            records: List of VectorRecord objects. Each must have a populated
                     `embedding` field before being passed here.

        Returns:
            The number of records successfully written.

        Raises:
            VectorStoreError: On connection failure or schema violation.
        """
        ...

    async def get_indexed_ids(self) -> set[str]:
        """
        Return the set of all primary key IDs already present in the collection.

        Used by IngestionService to skip re-embedding of existing chunks.

        Returns:
            Set of string chunk_id values.

        Raises:
            VectorStoreError: On query failure.
        """
        ...


@runtime_checkable
class LLMPort(Protocol):
    """Contract for generating text completions from an instruction-following LLM."""

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 3072,
    ) -> str:
        """
        Generate a text response given a system instruction and a user message.

        Args:
            system_prompt: Instructions for the model (role, constraints, language).
            user_message: The combined context + user question, formatted by RAGService.
            max_tokens: Hard upper limit on response length.

        Returns:
            The generated text response as a plain string.

        Raises:
            LLMUnavailableError: On API failure, quota exhaustion, or empty response.
        """
        ...


@runtime_checkable
class SummarizerPort(Protocol):
    """Contract for summarizing a full PDF document using a multimodal AI model."""

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """
        Upload and summarize a PDF using a multimodal LLM (e.g., Gemini).

        Args:
            request: SummaryRequest containing raw PDF bytes and the document title.

        Returns:
            SummaryResult with the generated summary and the model name used.

        Raises:
            LLMUnavailableError: On API failure.
            PDFExtractionError: If the model cannot process the file format.
        """
        ...


@runtime_checkable
class ScraperPort(Protocol):
    """Contract for crawling the HCMUT regulations listing page."""

    async def scrape(self) -> list[RegulationSource]:
        """
        Load the HCMUT regulation index page and extract all document links.

        Returns:
            List of RegulationSource objects. Returns empty list if the page
            loads successfully but contains no links — not an exception.

        Raises:
            ScraperError: If the page fails to load or the DOM is unrecognized.
        """
        ...


@runtime_checkable
class PDFDownloaderPort(Protocol):
    """Contract for downloading a PDF from a given URL."""

    async def download(self, url: str) -> bytes:
        """
        Download raw PDF bytes from a URL (including Google Drive links).

        Args:
            url: The direct or Google Drive URL to the PDF file.

        Returns:
            Raw binary content of the PDF file.

        Raises:
            PDFDownloadError: On HTTP error, timeout, or invalid file response.
        """
        ...


@runtime_checkable
class PDFExtractorPort(Protocol):
    """Contract for extracting plain text from a PDF binary."""

    def extract(self, pdf_bytes: bytes) -> tuple[str, ExtractionMethod]:
        """
        Extract clean text content from a PDF file.

        Args:
            pdf_bytes: Raw binary PDF content.

        Returns:
            Tuple of (extracted_text, extraction_method_used).

        Raises:
            PDFExtractionError: If extraction fails or quality is below threshold.
        """
        ...


@runtime_checkable
class ChunkerPort(Protocol):
    """Contract for splitting a long document text into overlapping chunks."""

    def chunk(
        self,
        text: str,
        doc_id: str,
        source_url: str,
        title: str,
    ) -> list[ProcessedChunk]:
        """
        Split document text into overlapping chunks suitable for vector indexing.

        Args:
            text: The full normalized text of the document.
            doc_id: SHA-1 hash of the source URL, used to build chunk IDs.
            source_url: Original URL of the source document.
            title: Human-readable document title for metadata.

        Returns:
            Ordered list of ProcessedChunk objects. At least one chunk is guaranteed
            if text is non-empty.

        Raises:
            ValueError: If text is empty.
        """
        ...


@runtime_checkable
class NotifierPort(Protocol):
    """Contract for sending a notification to an external channel (e.g., Discord)."""

    async def notify(self, payload: NotificationPayload) -> None:
        """
        Deliver a notification for a newly discovered regulation.

        Args:
            payload: The structured data to render in the notification embed.

        Raises:
            NotificationError: If the delivery fails after retries.
        """
        ...


@runtime_checkable
class CachePort(Protocol):
    """Contract for a local persistent cache of processed vector records."""

    def get(self, key: str) -> list[VectorRecord] | None:
        """
        Retrieve cached vector records for a source URL.

        Args:
            key: The source_url used as the cache key.

        Returns:
            List of VectorRecord if cached, None on cache miss.

        Raises:
            CacheError: On file read or parse failure.
        """
        ...

    def put(self, key: str, records: list[VectorRecord]) -> None:
        """
        Persist vector records for a source URL to the cache.

        Note: Embeddings are stripped before writing to keep the file size manageable.
        They are re-generated at index time if needed.

        Args:
            key: The source_url to use as the cache key.
            records: List of VectorRecord objects to persist.

        Raises:
            CacheError: On file write failure.
        """
        ...


@runtime_checkable
class StatePort(Protocol):
    """Contract for tracking which regulation URLs have been processed."""

    def get_seen_urls(self) -> set[str]:
        """
        Return the set of all previously processed regulation URLs.

        Returns:
            Set of URL strings. Returns empty set on first run.

        Raises:
            CacheError: If the state file is corrupted.
        """
        ...

    def mark_seen(self, urls: list[str]) -> None:
        """
        Persist one or more URLs as processed, preventing future reprocessing.

        Args:
            urls: List of source URL strings to mark as seen.

        Raises:
            CacheError: If the state file cannot be written.
        """
        ...
```

---

### `application/rag_service.py` (Skeleton)

```python
"""
Core RAG (Retrieval-Augmented Generation) orchestration service.

Intentionally agnostic of:
  - The specific LLM provider (Gemini, OpenAI, etc.)
  - The specific vector database (Milvus, Pinecone, etc.)
  - The presentation layer (Discord, REST API, CLI, etc.)

All external dependencies are injected as Protocol implementations.
"""
from __future__ import annotations

from hcmut_reg_bot.domain.exceptions import EmbeddingError, LLMUnavailableError, RetrievalError
from hcmut_reg_bot.domain.models import RAGRequest, RAGResponse, RetrievedChunk
from hcmut_reg_bot.domain.ports import EmbedderPort, LLMPort, RetrieverPort


class RAGService:
    """
    Orchestrates the full RAG pipeline: embed → retrieve → augment → generate.

    Dependencies are injected via the constructor, enabling complete mocking in tests.
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        retriever: RetrieverPort,
        llm: LLMPort,
    ) -> None:
        """
        Args:
            embedder: Converts query text to a dense vector.
            retriever: Searches the vector database for relevant chunks.
            llm: Generates a grounded answer from context + query.
        """
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm

    async def answer(self, request: RAGRequest) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user question.

        Pipeline:
            1. Embed the query via EmbedderPort.
            2. Retrieve top-k similar chunks via RetrieverPort.
            3. Select the system prompt based on context availability.
            4. Format retrieved chunks into a structured user message.
            5. Generate an answer via LLMPort.
            6. Return a validated RAGResponse.

        Args:
            request: RAGRequest containing the query string and top_k parameter.

        Returns:
            RAGResponse with the generated answer, source attributions,
            and a flag indicating whether context was found.

        Raises:
            EmbeddingError: If the query cannot be embedded.
            RetrievalError: If the vector search fails.
            LLMUnavailableError: If the LLM fails to generate a response.
        """
        raise NotImplementedError

    def _build_system_prompt(
        self, has_context: bool, is_comparison_query: bool
    ) -> str:
        """
        Select the appropriate system prompt based on query characteristics.

        Args:
            has_context: True if the retriever returned at least one chunk.
            is_comparison_query: True if the query asks to compare old vs. new rules.

        Returns:
            The system prompt string to pass to the LLM.
        """
        raise NotImplementedError

    def _build_user_message(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> str:
        """
        Format retrieved chunks and the user's question into a single LLM message.

        When context_chunks is empty, the message must contain the Vietnamese phrase
        "không tìm thấy" so the LLM produces a polite decline rather than hallucinating.

        Args:
            query: The original user question.
            context_chunks: List of retrieved chunks (may be empty).

        Returns:
            Formatted string combining numbered context blocks and the question.
        """
        raise NotImplementedError

    def _is_comparison_query(self, query: str) -> bool:
        """
        Detect if a query asks to compare two versions of a regulation.

        Checks for Vietnamese and English comparison keywords such as:
        'khác', 'so sánh', 'cũ', 'mới', 'thay đổi', 'compare', 'difference'.

        Args:
            query: The raw user query string.

        Returns:
            True if a comparison keyword is detected (case-insensitive).
        """
        raise NotImplementedError
```

---

### `application/monitor_service.py` (Skeleton)

```python
"""
Regulation monitoring service.

Responsibilities:
  1. Scrape the HCMUT regulation page for new document links.
  2. Download and summarize each new PDF.
  3. Send a Discord notification for each.
  4. Persist processed URLs to prevent duplicate alerts.
"""
from __future__ import annotations

from hcmut_reg_bot.domain.models import MonitorResult, NotificationPayload, SummaryRequest
from hcmut_reg_bot.domain.ports import (
    NotifierPort,
    PDFDownloaderPort,
    ScraperPort,
    StatePort,
    SummarizerPort,
)


class MonitorService:
    """Orchestrates one complete monitor cycle: scrape → download → summarize → notify."""

    def __init__(
        self,
        scraper: ScraperPort,
        downloader: PDFDownloaderPort,
        summarizer: SummarizerPort,
        notifier: NotifierPort,
        state: StatePort,
    ) -> None:
        """
        Args:
            scraper: Crawls the HCMUT regulation listing page.
            downloader: Downloads raw PDF bytes from a URL.
            summarizer: Generates a concise AI summary of a PDF.
            notifier: Sends a Discord embed notification.
            state: Persists processed URLs between runs.
        """
        self._scraper = scraper
        self._downloader = downloader
        self._summarizer = summarizer
        self._notifier = notifier
        self._state = state

    async def run(self) -> MonitorResult:
        """
        Execute one full monitoring cycle.

        Returns:
            MonitorResult with counts of new/notified regulations and
            a list of titles that failed to process.

        Raises:
            ScraperError: If the regulation page is inaccessible (non-retryable).
        """
        raise NotImplementedError
```

---

### `application/ingestion_service.py` (Skeleton)

```python
"""
Document ingestion service.

Responsibilities:
  1. Scrape the HCMUT regulation page for source documents.
  2. For each new document: download PDF → extract text → chunk → embed → index.
  3. Use a local JSONL cache to survive interruptions without re-downloading.
  4. Track which URLs have been indexed to enable incremental updates.
"""
from __future__ import annotations

from hcmut_reg_bot.domain.models import IngestionRequest, IngestionResult, VectorRecord
from hcmut_reg_bot.domain.ports import (
    CachePort,
    ChunkerPort,
    EmbedderPort,
    PDFDownloaderPort,
    PDFExtractorPort,
    ScraperPort,
    StatePort,
    VectorStorePort,
)


class IngestionService:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(
        self,
        scraper: ScraperPort,
        downloader: PDFDownloaderPort,
        extractor: PDFExtractorPort,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        cache: CachePort,
        state: StatePort,
    ) -> None:
        """
        Args:
            scraper: Crawls the HCMUT regulation listing page.
            downloader: Downloads raw PDF bytes.
            extractor: Extracts plain text from PDF bytes.
            chunker: Splits document text into overlapping chunks.
            embedder: Converts chunk texts to dense vectors.
            vector_store: Upserts vector records into the database.
            cache: Persists processed records to avoid re-embedding on restart.
            state: Tracks which source URLs have been indexed.
        """
        self._scraper = scraper
        self._downloader = downloader
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._cache = cache
        self._state = state

    async def run(self, request: IngestionRequest) -> IngestionResult:
        """
        Execute one full ingestion cycle.

        Args:
            request: Configuration controlling which documents to process.

        Returns:
            IngestionResult with processing statistics.

        Raises:
            ScraperError: If the regulation page is inaccessible.
        """
        raise NotImplementedError

    async def _process_single(
        self,
        source_url: str,
        title: str,
        force_reindex: bool,
    ) -> list[VectorRecord]:
        """
        Process a single regulation document end-to-end.

        Args:
            source_url: The Google Drive or direct PDF URL.
            title: Human-readable document title.
            force_reindex: If True, bypass the local JSONL cache.

        Returns:
            List of VectorRecord objects ready for upsert (with embeddings populated).

        Raises:
            PDFDownloadError: If the PDF cannot be fetched.
            PDFExtractionError: If text extraction fails.
            EmbeddingError: If embedding the chunks fails.
        """
        raise NotImplementedError
```

---

## Part 2 (Step 2): Test-Driven Development

> **Rule**: No real network calls. Every external dependency is an `AsyncMock`.
> Tests run fully offline in milliseconds. The test suite defines the **behavioral contract**
> that the Step 3 implementation must satisfy.

---

### `tests/conftest.py`

```python
"""
Shared pytest fixtures available to all test files automatically.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from hcmut_reg_bot.domain.models import RetrievedChunk


# ── Reusable data fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_query() -> str:
    return "Điểm trung bình tích lũy tối thiểu để không bị cảnh báo học vụ là bao nhiêu?"


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    """Two realistic chunks as would be returned by a live Milvus search."""
    return [
        RetrievedChunk(
            text=(
                "Điều 5. Điểm trung bình tích lũy (GPA): Sinh viên phải đạt GPA "
                "tối thiểu 2.0/4.0 để không bị cảnh báo học vụ."
            ),
            source_url="https://drive.google.com/file/d/abc123/view",
            doc_title="Quy chế đào tạo đại học HCMUT 2024",
            score=0.92,
            chunk_id="docabc:0",
        ),
        RetrievedChunk(
            text=(
                "Điều 6. Cảnh báo học vụ: Sinh viên bị cảnh báo khi GPA dưới 1.5 "
                "trong hai học kỳ liên tiếp."
            ),
            source_url="https://drive.google.com/file/d/abc123/view",
            doc_title="Quy chế đào tạo đại học HCMUT 2024",
            score=0.87,
            chunk_id="docabc:1",
        ),
    ]


@pytest.fixture
def fake_embedding() -> list[float]:
    """Deterministic 3072-dimension vector for testing."""
    return [0.1] * 3072


# ── Port mocks ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder(fake_embedding):
    embedder = AsyncMock()
    embedder.embed_query.return_value = fake_embedding
    embedder.embed_documents.return_value = [fake_embedding]
    return embedder


@pytest.fixture
def mock_retriever(sample_chunks):
    retriever = AsyncMock()
    retriever.similarity_search.return_value = sample_chunks
    return retriever


@pytest.fixture
def mock_empty_retriever():
    """A retriever that finds nothing — simulates a query with no matching regulations."""
    retriever = AsyncMock()
    retriever.similarity_search.return_value = []
    return retriever


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate.return_value = (
        "Theo Điều 5 Quy chế đào tạo HCMUT 2024, điểm GPA tối thiểu là 2.0/4.0."
    )
    return llm
```

---

### `tests/application/test_rag_service.py`

```python
"""
Comprehensive test suite for RAGService.

Coverage matrix:
  [HAPPY PATH]
  ✓ Returns a valid RAGResponse instance
  ✓ response.query echoes the original query verbatim
  ✓ response.answer is a non-empty string
  ✓ response.sources matches retrieved chunks exactly
  ✓ is_empty_context is False when chunks are found

  [ORCHESTRATION]
  ✓ Embedder receives the exact query string
  ✓ Retriever receives the embedding vector from the embedder
  ✓ top_k parameter is forwarded unchanged to the retriever
  ✓ LLM is called exactly once per answer() invocation
  ✓ LLM receives a non-empty system_prompt
  ✓ LLM user_message contains the original query
  ✓ LLM user_message contains all retrieved chunk texts

  [EDGE CASE 1: Empty context]
  ✓ is_empty_context is True when retriever returns []
  ✓ sources is an empty list
  ✓ LLM is still called (to produce a polite decline)
  ✓ user_message signals empty context ("không tìm thấy")
  ✓ RAGResponse schema is valid even with empty context

  [EDGE CASE 2: API failures]
  ✓ Embedder failure → EmbeddingError (with original __cause__)
  ✓ Embedder failure → retriever never called
  ✓ Retriever failure → RetrievalError (with original __cause__)
  ✓ Retriever failure → LLM never called
  ✓ LLM failure → LLMUnavailableError (with original __cause__)
  ✓ Domain exceptions chain the original cause

  [EDGE CASE 3: Comparison queries]
  ✓ Comparison keywords trigger a different system prompt
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from hcmut_reg_bot.application.rag_service import RAGService
from hcmut_reg_bot.domain.exceptions import (
    EmbeddingError,
    LLMUnavailableError,
    RetrievalError,
)
from hcmut_reg_bot.domain.models import RAGRequest, RAGResponse


# ── Helper ─────────────────────────────────────────────────────────────────────

def make_service(embedder, retriever, llm) -> RAGService:
    return RAGService(embedder=embedder, retriever=retriever, llm=llm)


# ══════════════════════════════════════════════════════════════════════════════
# HAPPY PATH
# ══════════════════════════════════════════════════════════════════════════════

class TestHappyPath:

    @pytest.mark.asyncio
    async def test_returns_rag_response_instance(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """The return value must be a valid RAGResponse Pydantic model."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert isinstance(response, RAGResponse)

    @pytest.mark.asyncio
    async def test_response_echoes_original_query(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """response.query must equal the original request query verbatim."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert response.query == sample_query

    @pytest.mark.asyncio
    async def test_response_answer_is_non_empty_string(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """The answer field must be a non-empty string."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    @pytest.mark.asyncio
    async def test_response_sources_match_retrieved_chunks(
        self, mock_embedder, mock_retriever, mock_llm, sample_query, sample_chunks
    ):
        """Sources in the response must correspond exactly to retrieved chunks."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert len(response.sources) == len(sample_chunks)
        for source, chunk in zip(response.sources, sample_chunks):
            assert source.chunk_id == chunk.chunk_id
            assert source.doc_title == chunk.doc_title
            assert source.source_url == chunk.source_url

    @pytest.mark.asyncio
    async def test_is_empty_context_false_when_chunks_found(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """is_empty_context must be False when chunks were retrieved."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert response.is_empty_context is False


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION: CORRECT PORT CALL CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestration:

    @pytest.mark.asyncio
    async def test_embedder_receives_exact_query_string(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """embed_query must be called exactly once with the raw user query."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        mock_embedder.embed_query.assert_called_once_with(sample_query)

    @pytest.mark.asyncio
    async def test_retriever_receives_embedding_from_embedder(
        self, mock_embedder, mock_retriever, mock_llm, fake_embedding, sample_query
    ):
        """The exact vector from embed_query must be forwarded to similarity_search."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query, top_k=3))
        mock_retriever.similarity_search.assert_called_once_with(
            query_embedding=fake_embedding,
            top_k=3,
        )

    @pytest.mark.asyncio
    async def test_top_k_forwarded_to_retriever(
        self, mock_embedder, mock_retriever, mock_llm, fake_embedding
    ):
        """top_k from RAGRequest must reach the retriever unchanged."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        for k in [1, 5, 10, 20]:
            mock_retriever.similarity_search.reset_mock()
            await service.answer(RAGRequest(query="test", top_k=k))
            mock_retriever.similarity_search.assert_called_once_with(
                query_embedding=fake_embedding, top_k=k
            )

    @pytest.mark.asyncio
    async def test_llm_called_exactly_once(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """LLM must be called exactly once per answer() invocation."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_receives_non_empty_system_prompt(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """The system_prompt passed to the LLM must be a non-empty string."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        call_kwargs = mock_llm.generate.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs.args[0]
        assert isinstance(system_prompt, str) and len(system_prompt) > 0

    @pytest.mark.asyncio
    async def test_llm_user_message_contains_query(
        self, mock_embedder, mock_retriever, mock_llm, sample_query
    ):
        """The user_message sent to the LLM must contain the original query."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        call_kwargs = mock_llm.generate.call_args
        user_message = call_kwargs.kwargs.get("user_message") or call_kwargs.args[1]
        assert sample_query in user_message

    @pytest.mark.asyncio
    async def test_llm_user_message_contains_all_chunk_texts(
        self, mock_embedder, mock_retriever, mock_llm, sample_query, sample_chunks
    ):
        """All retrieved chunk texts must appear in the user_message."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        call_kwargs = mock_llm.generate.call_args
        user_message = call_kwargs.kwargs.get("user_message") or call_kwargs.args[1]
        for chunk in sample_chunks:
            assert chunk.text in user_message


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASE 1: EMPTY CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptyContext:
    """When the retriever returns no chunks, the service must degrade gracefully."""

    @pytest.mark.asyncio
    async def test_empty_context_sets_flag_true(
        self, mock_embedder, mock_empty_retriever, mock_llm, sample_query
    ):
        """is_empty_context must be True when the retriever returns an empty list."""
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert response.is_empty_context is True

    @pytest.mark.asyncio
    async def test_empty_context_sources_is_empty_list(
        self, mock_embedder, mock_empty_retriever, mock_llm, sample_query
    ):
        """sources must be an empty list when no chunks are retrieved."""
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_empty_context_still_calls_llm(
        self, mock_embedder, mock_empty_retriever, mock_llm, sample_query
    ):
        """The LLM must still be called even with no context — to produce a polite decline."""
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_context_user_message_signals_no_data(
        self, mock_embedder, mock_empty_retriever, mock_llm, sample_query
    ):
        """When context is empty, the user_message must signal this so the LLM
        does not hallucinate. Checks for 'không tìm thấy' or equivalent markers."""
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        await service.answer(RAGRequest(query=sample_query))
        call_kwargs = mock_llm.generate.call_args
        user_message = call_kwargs.kwargs.get("user_message") or call_kwargs.args[1]
        has_signal = (
            "không tìm thấy" in user_message.lower()
            or "no context" in user_message.lower()
            or "no relevant" in user_message.lower()
        )
        assert has_signal, (
            f"user_message must signal empty context. Got: {user_message!r}"
        )

    @pytest.mark.asyncio
    async def test_empty_context_response_schema_valid(
        self, mock_embedder, mock_empty_retriever, mock_llm, sample_query
    ):
        """RAGResponse schema must be valid even with empty context."""
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=sample_query))
        assert isinstance(response, RAGResponse)
        assert isinstance(response.answer, str) and len(response.answer) > 0


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASE 2: EXTERNAL API FAILURES
# ══════════════════════════════════════════════════════════════════════════════

class TestExternalFailures:
    """Each infrastructure failure must be caught and re-raised as a domain exception."""

    @pytest.mark.asyncio
    async def test_embedder_failure_raises_embedding_error(
        self, mock_retriever, mock_llm, sample_query
    ):
        """Any exception from embed_query must become EmbeddingError."""
        failing_embedder = AsyncMock()
        failing_embedder.embed_query.side_effect = RuntimeError("Quota exceeded: 429")
        service = make_service(failing_embedder, mock_retriever, mock_llm)
        with pytest.raises(EmbeddingError):
            await service.answer(RAGRequest(query=sample_query))

    @pytest.mark.asyncio
    async def test_embedder_failure_does_not_call_retriever(
        self, mock_retriever, mock_llm, sample_query
    ):
        """If embedding fails, the retriever must never be called."""
        failing_embedder = AsyncMock()
        failing_embedder.embed_query.side_effect = RuntimeError("Network timeout")
        service = make_service(failing_embedder, mock_retriever, mock_llm)
        with pytest.raises(EmbeddingError):
            await service.answer(RAGRequest(query=sample_query))
        mock_retriever.similarity_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_retriever_failure_raises_retrieval_error(
        self, mock_embedder, mock_llm, sample_query
    ):
        """Any exception from similarity_search must become RetrievalError."""
        failing_retriever = AsyncMock()
        failing_retriever.similarity_search.side_effect = ConnectionError(
            "Milvus: connection refused on port 19530"
        )
        service = make_service(mock_embedder, failing_retriever, mock_llm)
        with pytest.raises(RetrievalError) as exc_info:
            await service.answer(RAGRequest(query=sample_query))
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    @pytest.mark.asyncio
    async def test_retriever_failure_does_not_call_llm(
        self, mock_embedder, mock_llm, sample_query
    ):
        """If retrieval fails, the LLM must never be called."""
        failing_retriever = AsyncMock()
        failing_retriever.similarity_search.side_effect = ConnectionError("DB down")
        service = make_service(mock_embedder, failing_retriever, mock_llm)
        with pytest.raises(RetrievalError):
            await service.answer(RAGRequest(query=sample_query))
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_raises_llm_unavailable_error(
        self, mock_embedder, mock_retriever, sample_query
    ):
        """Any exception from LLM.generate must become LLMUnavailableError."""
        failing_llm = AsyncMock()
        failing_llm.generate.side_effect = TimeoutError("LLM API: 503 Service Unavailable")
        service = make_service(mock_embedder, mock_retriever, failing_llm)
        with pytest.raises(LLMUnavailableError) as exc_info:
            await service.answer(RAGRequest(query=sample_query))
        assert isinstance(exc_info.value.__cause__, TimeoutError)

    @pytest.mark.asyncio
    async def test_domain_exceptions_chain_original_cause(
        self, mock_retriever, mock_llm, sample_query
    ):
        """Domain exceptions must preserve the original exception as __cause__
        so the full traceback is available for logging."""
        original_error = RuntimeError("Original API error")
        failing_embedder = AsyncMock()
        failing_embedder.embed_query.side_effect = original_error
        service = make_service(failing_embedder, mock_retriever, mock_llm)
        with pytest.raises(EmbeddingError) as exc_info:
            await service.answer(RAGRequest(query=sample_query))
        assert exc_info.value.__cause__ is original_error


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASE 3: COMPARISON QUERY ROUTING
# ══════════════════════════════════════════════════════════════════════════════

class TestComparisonQueryDetection:
    """Queries comparing old vs. new regulations must trigger a distinct system prompt."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("comparison_query", [
        "Quy định mới khác gì so với quy định cũ?",
        "So sánh quy chế đào tạo 2022 và 2024",
        "Sự thay đổi trong quy định học bổng là gì?",
        "What is the difference between the old and new regulations?",
        "Compare the two regulation versions",
    ])
    async def test_comparison_keywords_trigger_different_prompt(
        self, mock_embedder, mock_retriever, mock_llm, comparison_query
    ):
        """Comparison queries must produce a different system prompt than normal queries."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)

        await service.answer(RAGRequest(query=comparison_query))
        await service.answer(RAGRequest(query="GPA tối thiểu là bao nhiêu?"))

        calls = mock_llm.generate.call_args_list
        comparison_prompt = calls[0].kwargs.get("system_prompt") or calls[0].args[0]
        normal_prompt = calls[1].kwargs.get("system_prompt") or calls[1].args[0]

        assert comparison_prompt != normal_prompt, (
            "Comparison query must produce a different system prompt than a normal query"
        )
```

---

## Part 3 (Step 3): Core Logic Implementation

> **Rule**: Make every test from Step 2 pass. No extra features.
> The RAGService must remain 100% agnostic of infrastructure.

---

### `application/rag_service.py` (Complete)

```python
"""
RAGService — complete implementation.

Design decisions:
  - Three distinct system prompts defined at module level (immutable, easily audited).
  - Comparison detection is a pure function (no I/O) — fast and trivially testable.
  - All infrastructure exceptions caught at the boundary of each port call
    and re-raised as typed domain exceptions with `raise ... from exc`
    to preserve the original traceback.
"""
from __future__ import annotations

from hcmut_reg_bot.domain.exceptions import EmbeddingError, LLMUnavailableError, RetrievalError
from hcmut_reg_bot.domain.models import RAGRequest, RAGResponse, RetrievedChunk
from hcmut_reg_bot.domain.ports import EmbedderPort, LLMPort, RetrieverPort

# ── System Prompts ─────────────────────────────────────────────────────────────

_PROMPT_WITH_CONTEXT = """\
Bạn là trợ lý AI chuyên về quy chế và quy định của Trường Đại học Bách Khoa TP.HCM (HCMUT).
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên dựa HOÀN TOÀN vào các đoạn văn bản \
được cung cấp trong ngữ cảnh bên dưới.

Quy tắc:
1. Chỉ sử dụng thông tin có trong ngữ cảnh. Không được tự bịa đặt.
2. Trả lời bằng tiếng Việt, rõ ràng và súc tích.
3. Trích dẫn số điều khoản hoặc tên quy định nếu có trong ngữ cảnh.
4. Nếu ngữ cảnh không đủ để trả lời đầy đủ, hãy nói rõ điều đó.\
"""

_PROMPT_COMPARISON = """\
Bạn là trợ lý AI chuyên về quy chế và quy định của HCMUT.
Câu hỏi này yêu cầu SO SÁNH giữa các phiên bản quy định khác nhau.

Hướng dẫn:
1. Nêu rõ điểm khác biệt giữa quy định cũ và mới dựa trên ngữ cảnh.
2. Trình bày dưới dạng bảng hoặc danh sách để dễ so sánh.
3. Chỉ sử dụng thông tin có trong ngữ cảnh. Không được tự bịa đặt.
4. Nếu ngữ cảnh chỉ có một phiên bản, hãy nói rõ điều đó.\
"""

_PROMPT_EMPTY_CONTEXT = """\
Bạn là trợ lý AI về quy chế HCMUT.
Hệ thống không tìm thấy thông tin liên quan trong cơ sở dữ liệu quy định.
Hãy thông báo lịch sự rằng bạn không tìm thấy thông tin liên quan và gợi ý \
sinh viên liên hệ Phòng Đào tạo hoặc kiểm tra trực tiếp trên website HCMUT.\
"""

_COMPARISON_KEYWORDS: frozenset[str] = frozenset([
    "khác", "so sánh", "cũ", "mới", "thay đổi", "trước", "sau",
    "difference", "compare", "old", "new", "changed",
])


class RAGService:

    def __init__(
        self,
        embedder: EmbedderPort,
        retriever: RetrieverPort,
        llm: LLMPort,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm

    async def answer(self, request: RAGRequest) -> RAGResponse:
        # 1. Embed
        try:
            query_embedding = await self._embedder.embed_query(request.query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

        # 2. Retrieve
        try:
            chunks = await self._retriever.similarity_search(
                query_embedding=query_embedding,
                top_k=request.top_k,
            )
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        # 3. Build prompt
        is_empty = len(chunks) == 0
        system_prompt = self._build_system_prompt(
            has_context=not is_empty,
            is_comparison_query=self._is_comparison_query(request.query),
        )
        user_message = self._build_user_message(request.query, chunks)

        # 4. Generate
        try:
            answer_text = await self._llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except Exception as exc:
            raise LLMUnavailableError(f"LLM generation failed: {exc}") from exc

        return RAGResponse(
            query=request.query,
            answer=answer_text,
            sources=chunks,
            is_empty_context=is_empty,
        )

    def _build_system_prompt(
        self, has_context: bool, is_comparison_query: bool
    ) -> str:
        if not has_context:
            return _PROMPT_EMPTY_CONTEXT
        if is_comparison_query:
            return _PROMPT_COMPARISON
        return _PROMPT_WITH_CONTEXT

    def _build_user_message(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> str:
        if not context_chunks:
            return (
                f"Câu hỏi: {query}\n\n"
                "[Không tìm thấy ngữ cảnh liên quan trong cơ sở dữ liệu quy định]"
            )
        parts = [
            f"[Nguồn {i}: {c.doc_title}]\n{c.text}"
            for i, c in enumerate(context_chunks, 1)
        ]
        return "Ngữ cảnh:\n\n" + "\n\n---\n\n".join(parts) + f"\n\n---\n\nCâu hỏi: {query}"

    def _is_comparison_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(kw in lowered for kw in _COMPARISON_KEYWORDS)
```

---

### `application/monitor_service.py` (Complete)

```python
from __future__ import annotations

from hcmut_reg_bot.domain.exceptions import HCMUTBotError
from hcmut_reg_bot.domain.models import MonitorResult, NotificationPayload, SummaryRequest
from hcmut_reg_bot.domain.ports import (
    NotifierPort, PDFDownloaderPort, ScraperPort, StatePort, SummarizerPort,
)

_MAX_SUMMARY_CHARS = 4000


class MonitorService:

    def __init__(
        self,
        scraper: ScraperPort,
        downloader: PDFDownloaderPort,
        summarizer: SummarizerPort,
        notifier: NotifierPort,
        state: StatePort,
    ) -> None:
        self._scraper = scraper
        self._downloader = downloader
        self._summarizer = summarizer
        self._notifier = notifier
        self._state = state

    async def run(self) -> MonitorResult:
        all_regulations = await self._scraper.scrape()

        seen_urls = self._state.get_seen_urls()
        new_regulations = [r for r in all_regulations if r.link not in seen_urls]

        notified_count = 0
        failed_titles: list[str] = []

        for reg in new_regulations:
            try:
                pdf_bytes = await self._downloader.download(reg.link)
                summary_result = await self._summarizer.summarize(
                    SummaryRequest(pdf_bytes=pdf_bytes, title=reg.title)
                )
                payload = NotificationPayload(
                    title=reg.title,
                    link=reg.link,
                    summary=summary_result.summary[:_MAX_SUMMARY_CHARS],
                )
                await self._notifier.notify(payload)
                self._state.mark_seen([reg.link])
                notified_count += 1
            except HCMUTBotError:
                failed_titles.append(reg.title)

        return MonitorResult(
            new_count=len(new_regulations),
            notified_count=notified_count,
            failed_titles=failed_titles,
        )
```

---

### `application/ingestion_service.py` (Complete)

```python
from __future__ import annotations

import hashlib

from hcmut_reg_bot.domain.exceptions import HCMUTBotError
from hcmut_reg_bot.domain.models import IngestionRequest, IngestionResult, VectorRecord
from hcmut_reg_bot.domain.ports import (
    CachePort, ChunkerPort, EmbedderPort, PDFDownloaderPort,
    PDFExtractorPort, ScraperPort, StatePort, VectorStorePort,
)


class IngestionService:

    def __init__(
        self,
        scraper: ScraperPort,
        downloader: PDFDownloaderPort,
        extractor: PDFExtractorPort,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        cache: CachePort,
        state: StatePort,
    ) -> None:
        self._scraper = scraper
        self._downloader = downloader
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._cache = cache
        self._state = state

    async def run(self, request: IngestionRequest) -> IngestionResult:
        all_regulations = await self._scraper.scrape()

        seen_urls = self._state.get_seen_urls()
        to_process = (
            [r for r in all_regulations if r.link not in seen_urls]
            if request.only_new
            else all_regulations
        )

        processed_count = 0
        indexed_count = 0
        failed_titles: list[str] = []

        for reg in to_process:
            try:
                records = await self._process_single(
                    source_url=reg.link,
                    title=reg.title,
                    force_reindex=request.force_reindex,
                )
                n = await self._vector_store.upsert(records)
                indexed_count += n
                self._state.mark_seen([reg.link])
                processed_count += 1
            except HCMUTBotError:
                failed_titles.append(reg.title)

        return IngestionResult(
            scraped_count=len(all_regulations),
            processed_count=processed_count,
            indexed_count=indexed_count,
            failed_titles=failed_titles,
        )

    async def _process_single(
        self,
        source_url: str,
        title: str,
        force_reindex: bool,
    ) -> list[VectorRecord]:
        # Cache hit
        if not force_reindex:
            cached = self._cache.get(source_url)
            if cached is not None:
                return cached

        # Download + extract
        pdf_bytes = await self._downloader.download(source_url)
        text, method = self._extractor.extract(pdf_bytes)

        # Chunk
        doc_id = hashlib.sha1(source_url.encode()).hexdigest()
        chunks = self._chunker.chunk(
            text=text, doc_id=doc_id, source_url=source_url, title=title
        )

        # Embed
        embeddings = await self._embedder.embed_documents([c.text for c in chunks])

        records = [
            VectorRecord(
                id=chunk.chunk_id,
                text=chunk.text,
                embedding=emb,
                source_url=chunk.source_url,
                doc_title=chunk.doc_title,
                chunk_id=chunk.chunk_id,
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        # Cache (strip embeddings to keep file size small)
        records_for_cache = [r.model_copy(update={"embedding": None}) for r in records]
        self._cache.put(source_url, records_for_cache)

        return records
```

---

### Infrastructure Adapters (Concrete Implementations)

These are the only files that import third-party libraries.

#### `infrastructure/llm/gemini_llm.py`

```python
from __future__ import annotations

import google.generativeai as genai
from hcmut_reg_bot.config.settings import Settings


class GeminiLLM:
    """Concrete LLMPort backed by Google Gemini."""

    def __init__(self, settings: Settings) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._settings = settings

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 3072,
    ) -> str:
        model = genai.GenerativeModel(
            model_name=self._settings.llm_model,
            generation_config=genai.GenerationConfig(
                temperature=self._settings.llm_temperature,
                max_output_tokens=max_tokens,
            ),
            system_instruction=system_prompt,
        )
        response = await model.generate_content_async(user_message)
        return response.text
```

#### `infrastructure/embeddings/gemini_embedder.py`

```python
from __future__ import annotations

import google.generativeai as genai
from hcmut_reg_bot.config.settings import Settings


class GeminiEmbedder:
    """Concrete EmbedderPort backed by Google Gemini Embeddings."""

    def __init__(self, settings: Settings) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model

    async def embed_query(self, text: str) -> list[float]:
        result = await genai.embed_content_async(
            model=self._model, content=text, task_type="retrieval_query"
        )
        return result["embedding"]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            result = await genai.embed_content_async(
                model=self._model, content=text, task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings
```

#### `infrastructure/vector_db/milvus_adapter.py`

```python
from __future__ import annotations

from pymilvus import MilvusClient
from hcmut_reg_bot.config.settings import Settings
from hcmut_reg_bot.domain.models import RetrievedChunk, VectorRecord


class MilvusAdapter:
    """Concrete RetrieverPort + VectorStorePort backed by Milvus / Zilliz Cloud."""

    def __init__(self, settings: Settings) -> None:
        connect_kwargs: dict = {"uri": settings.milvus_uri}
        if settings.milvus_api_key:
            connect_kwargs["token"] = settings.milvus_api_key
        self._client = MilvusClient(**connect_kwargs)
        self._collection = settings.milvus_collection_name

    async def similarity_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        results = self._client.search(
            collection_name=self._collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "source_url", "doc_title", "chunk_id"],
        )
        return [
            RetrievedChunk(
                text=hit["entity"]["text"],
                source_url=hit["entity"]["source_url"],
                doc_title=hit["entity"]["doc_title"],
                score=float(hit["distance"]),
                chunk_id=hit["entity"]["chunk_id"],
            )
            for hit in results[0]
        ]

    async def upsert(self, records: list[VectorRecord]) -> int:
        data = [
            {
                "id": r.id,
                "embedding": r.embedding,
                "text": r.text,
                "source_url": r.source_url,
                "doc_title": r.doc_title,
                "chunk_id": r.chunk_id,
            }
            for r in records
            if r.embedding is not None
        ]
        result = self._client.upsert(collection_name=self._collection, data=data)
        return result.get("upsert_count", len(data))

    async def get_indexed_ids(self) -> set[str]:
        results = self._client.query(
            collection_name=self._collection,
            filter="id != ''",
            output_fields=["id"],
            limit=16384,
        )
        return {r["id"] for r in results}
```

---

### `main.py` — Composition Root

```python
"""
Application entry point.

This is the ONLY file that imports from both application/ and infrastructure/.
All other modules import only from domain/ or their own layer.
Swapping any adapter (e.g., Milvus → Pinecone) requires changing only this file.
"""
from __future__ import annotations

import logging

import discord
from discord.ext import commands

from hcmut_reg_bot.application.rag_service import RAGService
from hcmut_reg_bot.config.settings import Settings
from hcmut_reg_bot.infrastructure.embeddings.gemini_embedder import GeminiEmbedder
from hcmut_reg_bot.infrastructure.llm.gemini_llm import GeminiLLM
from hcmut_reg_bot.infrastructure.vector_db.milvus_adapter import MilvusAdapter
from hcmut_reg_bot.presentation.discord_bot.cogs.rag_cog import RAGCog


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    settings = Settings()  # Raises pydantic.ValidationError on missing keys

    # Wire infrastructure
    embedder = GeminiEmbedder(settings)
    milvus = MilvusAdapter(settings)
    llm = GeminiLLM(settings)

    # Wire application
    rag_service = RAGService(embedder=embedder, retriever=milvus, llm=llm)

    # Wire presentation
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    async def setup_hook() -> None:
        await bot.add_cog(RAGCog(bot=bot, rag_service=rag_service))

    bot.setup_hook = setup_hook
    bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
```

---

## Part 4 (Step 4): Production Ready

> **Additions**: structured `logging` at every pipeline stage, quota-aware retry logic,
> proper `async/await` throughout, user-facing error mapping in the Discord cog.

---

### Final `application/rag_service.py` — With Full Logging

```python
from __future__ import annotations

import logging

from hcmut_reg_bot.domain.exceptions import EmbeddingError, LLMUnavailableError, RetrievalError
from hcmut_reg_bot.domain.models import RAGRequest, RAGResponse, RetrievedChunk
from hcmut_reg_bot.domain.ports import EmbedderPort, LLMPort, RetrieverPort

logger = logging.getLogger(__name__)

_PROMPT_WITH_CONTEXT = """\
Bạn là trợ lý AI chuyên về quy chế và quy định của Trường Đại học Bách Khoa TP.HCM (HCMUT).
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên dựa HOÀN TOÀN vào các đoạn văn bản \
được cung cấp trong ngữ cảnh bên dưới.

Quy tắc:
1. Chỉ sử dụng thông tin có trong ngữ cảnh. Không được tự bịa đặt.
2. Trả lời bằng tiếng Việt, rõ ràng và súc tích.
3. Trích dẫn số điều khoản hoặc tên quy định nếu có trong ngữ cảnh.
4. Nếu ngữ cảnh không đủ để trả lời đầy đủ, hãy nói rõ điều đó.\
"""

_PROMPT_COMPARISON = """\
Bạn là trợ lý AI chuyên về quy chế HCMUT. Câu hỏi yêu cầu SO SÁNH các phiên bản quy định.

Hướng dẫn:
1. Nêu rõ điểm khác biệt giữa quy định cũ và mới dựa trên ngữ cảnh.
2. Trình bày dưới dạng bảng hoặc danh sách để dễ so sánh.
3. Chỉ sử dụng thông tin có trong ngữ cảnh. Không được tự bịa đặt.\
"""

_PROMPT_EMPTY_CONTEXT = """\
Bạn là trợ lý AI về quy chế HCMUT.
Hệ thống không tìm thấy thông tin liên quan. Hãy thông báo lịch sự và gợi ý sinh viên \
liên hệ Phòng Đào tạo hoặc kiểm tra trực tiếp trên website HCMUT.\
"""

_COMPARISON_KEYWORDS: frozenset[str] = frozenset([
    "khác", "so sánh", "cũ", "mới", "thay đổi", "trước", "sau",
    "difference", "compare", "old", "new", "changed",
])


class RAGService:

    def __init__(
        self,
        embedder: EmbedderPort,
        retriever: RetrieverPort,
        llm: LLMPort,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm

    async def answer(self, request: RAGRequest) -> RAGResponse:
        logger.info(
            "RAG pipeline started | query_length=%d top_k=%d",
            len(request.query), request.top_k,
        )

        # 1. Embed
        try:
            query_embedding = await self._embedder.embed_query(request.query)
            logger.debug("Query embedded | vector_dim=%d", len(query_embedding))
        except Exception as exc:
            logger.error("Embedding failed | error=%s", exc, exc_info=True)
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

        # 2. Retrieve
        try:
            chunks = await self._retriever.similarity_search(
                query_embedding=query_embedding, top_k=request.top_k
            )
            logger.info("Retrieval complete | chunks_found=%d", len(chunks))
        except Exception as exc:
            logger.error("Vector retrieval failed | error=%s", exc, exc_info=True)
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        # 3. Build prompt
        is_empty = len(chunks) == 0
        is_comparison = self._is_comparison_query(request.query)
        if is_empty:
            logger.warning("No relevant chunks found | query=%r", request.query[:80])
        if is_comparison:
            logger.debug("Comparison query detected")

        system_prompt = self._build_system_prompt(not is_empty, is_comparison)
        user_message = self._build_user_message(request.query, chunks)

        # 4. Generate
        try:
            answer_text = await self._llm.generate(
                system_prompt=system_prompt, user_message=user_message
            )
            logger.info(
                "LLM generation complete | answer_length=%d is_empty_context=%s",
                len(answer_text), is_empty,
            )
        except Exception as exc:
            logger.error("LLM generation failed | error=%s", exc, exc_info=True)
            raise LLMUnavailableError(f"LLM generation failed: {exc}") from exc

        response = RAGResponse(
            query=request.query,
            answer=answer_text,
            sources=chunks,
            is_empty_context=is_empty,
        )
        logger.info(
            "RAG pipeline complete | sources=%d empty_context=%s",
            len(chunks), is_empty,
        )
        return response

    def _build_system_prompt(self, has_context: bool, is_comparison_query: bool) -> str:
        if not has_context:
            return _PROMPT_EMPTY_CONTEXT
        if is_comparison_query:
            return _PROMPT_COMPARISON
        return _PROMPT_WITH_CONTEXT

    def _build_user_message(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        if not context_chunks:
            return (
                f"Câu hỏi: {query}\n\n"
                "[Không tìm thấy ngữ cảnh liên quan trong cơ sở dữ liệu quy định]"
            )
        parts = [
            f"[Nguồn {i}: {c.doc_title}]\n{c.text}"
            for i, c in enumerate(context_chunks, 1)
        ]
        return "Ngữ cảnh:\n\n" + "\n\n---\n\n".join(parts) + f"\n\n---\n\nCâu hỏi: {query}"

    def _is_comparison_query(self, query: str) -> bool:
        return any(kw in query.lower() for kw in _COMPARISON_KEYWORDS)
```

---

### Final `infrastructure/llm/gemini_llm.py` — With Quota-Aware Retry

```python
from __future__ import annotations

import asyncio
import logging

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from hcmut_reg_bot.config.settings import Settings
from hcmut_reg_bot.domain.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
_RETRY_DELAY_SECONDS = 20
_MAX_ATTEMPTS = 2


class GeminiLLM:
    """
    Gemini LLM adapter with quota-aware retry.

    On ResourceExhausted (429) or ServiceUnavailable (503), waits
    _RETRY_DELAY_SECONDS before one retry. If the retry also fails,
    raises LLMUnavailableError.
    """

    def __init__(self, settings: Settings) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._settings = settings

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 3072,
    ) -> str:
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                logger.debug(
                    "LLM call | attempt=%d/%d model=%s",
                    attempt, _MAX_ATTEMPTS, self._settings.llm_model,
                )
                model = genai.GenerativeModel(
                    model_name=self._settings.llm_model,
                    generation_config=genai.GenerationConfig(
                        temperature=self._settings.llm_temperature,
                        max_output_tokens=max_tokens,
                    ),
                    system_instruction=system_prompt,
                )
                response = await model.generate_content_async(user_message)
                text = response.text
                if not text:
                    raise LLMUnavailableError("LLM returned an empty response")
                return text

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == _MAX_ATTEMPTS:
                    logger.error(
                        "LLM unavailable after %d attempts | error=%s",
                        _MAX_ATTEMPTS, exc,
                    )
                    raise LLMUnavailableError(
                        f"LLM unavailable after {_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "LLM rate-limited, retrying in %ds | attempt=%d error=%s",
                    _RETRY_DELAY_SECONDS, attempt, exc,
                )
                await asyncio.sleep(_RETRY_DELAY_SECONDS)

        raise LLMUnavailableError("LLM generation failed") from last_exc
```

---

### Final `infrastructure/embeddings/gemini_embedder.py` — With Retry

```python
from __future__ import annotations

import asyncio
import logging

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from hcmut_reg_bot.config.settings import Settings
from hcmut_reg_bot.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Gemini Embedder with configurable quota-aware retry."""

    def __init__(self, settings: Settings) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model
        self._retry_count = settings.embedding_retry_count
        self._retry_delay = settings.embedding_retry_delay

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed(text, task_type="retrieval_query")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            emb = await self._embed(text, task_type="retrieval_document")
            results.append(emb)
        return results

    async def _embed(self, text: str, task_type: str) -> list[float]:
        last_exc: Exception | None = None

        for attempt in range(1, self._retry_count + 1):
            try:
                result = await genai.embed_content_async(
                    model=self._model, content=text, task_type=task_type
                )
                return result["embedding"]

            except ResourceExhausted as exc:
                last_exc = exc
                logger.warning(
                    "Embedding quota exceeded | attempt=%d/%d retrying in %.1fs",
                    attempt, self._retry_count, self._retry_delay,
                )
                await asyncio.sleep(self._retry_delay)

            except Exception as exc:
                logger.error("Embedding API error | error=%s", exc, exc_info=True)
                raise EmbeddingError(f"Embedding call failed: {exc}") from exc

        raise EmbeddingError(
            f"Embedding failed after {self._retry_count} attempts"
        ) from last_exc
```

---

### Final `presentation/discord_bot/cogs/rag_cog.py`

```python
"""
Discord presentation layer for the RAG chatbot.

Responsibilities:
  - Accept !ask commands.
  - Forward to RAGService (injected — no direct knowledge of Gemini or Milvus).
  - Map domain exceptions to user-friendly Vietnamese error messages.
  - Chunk long responses to respect Discord's 2000-char message limit.
  - Attach oversized responses as a .txt file.
"""
from __future__ import annotations

import io
import logging

import discord
from discord.ext import commands

from hcmut_reg_bot.application.rag_service import RAGService
from hcmut_reg_bot.domain.exceptions import (
    EmbeddingError,
    HCMUTBotError,
    LLMUnavailableError,
    RetrievalError,
)
from hcmut_reg_bot.domain.models import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)

_MAX_MSG_CHARS = 1900
_MAX_SOURCES_CHARS = 1900

_ERROR_MESSAGES: dict[type[HCMUTBotError], str] = {
    EmbeddingError: "⚠️ Có lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại.",
    RetrievalError: "⚠️ Hệ thống tìm kiếm tạm thời gặp sự cố. Vui lòng thử lại sau.",
    LLMUnavailableError: (
        "⚠️ Hệ thống AI tạm thời không khả dụng (có thể do giới hạn quota). "
        "Vui lòng thử lại sau vài phút."
    ),
}


class RAGCog(commands.Cog, name="RAG"):
    """Commands for querying the HCMUT regulation knowledge base."""

    def __init__(self, bot: commands.Bot, rag_service: RAGService) -> None:
        self._bot = bot
        self._rag_service = rag_service

    @commands.command(
        name="ask",
        help="Đặt câu hỏi về quy chế HCMUT. Ví dụ: !ask GPA tối thiểu là bao nhiêu?",
    )
    async def ask(self, ctx: commands.Context, *, question: str) -> None:
        """Handle the !ask command: embed → retrieve → generate → reply."""
        log_ctx = {
            "user_id": ctx.author.id,
            "guild_id": ctx.guild.id if ctx.guild else "DM",
            "query_length": len(question),
        }
        logger.info("Received !ask | %s", log_ctx)

        async with ctx.typing():
            try:
                request = RAGRequest(query=question)
                response = await self._rag_service.answer(request)
                await self._send_response(ctx, response)

            except HCMUTBotError as exc:
                user_msg = _ERROR_MESSAGES.get(
                    type(exc),
                    "❌ Đã xảy ra lỗi không mong đợi. Vui lòng thử lại.",
                )
                logger.warning(
                    "Domain error in !ask | type=%s error=%s ctx=%s",
                    type(exc).__name__, exc, log_ctx,
                )
                await ctx.reply(user_msg)

            except Exception as exc:
                logger.exception("Unexpected error in !ask | error=%s ctx=%s", exc, log_ctx)
                await ctx.reply(
                    "❌ Đã xảy ra lỗi không mong đợi. Vui lòng thử lại hoặc liên hệ admin."
                )

    async def _send_response(
        self, ctx: commands.Context, response: RAGResponse
    ) -> None:
        # Send answer in chunks
        answer = response.answer
        answer_chunks = [
            answer[i : i + _MAX_MSG_CHARS]
            for i in range(0, len(answer), _MAX_MSG_CHARS)
        ]
        for i, chunk in enumerate(answer_chunks):
            if i == 0:
                await ctx.reply(chunk)
            else:
                await ctx.send(chunk)

        if not response.sources:
            return

        # Build sources block
        source_lines = [
            f"**{i + 1}.** {s.doc_title}\n<{s.source_url}>"
            for i, s in enumerate(response.sources)
        ]
        sources_text = "**Nguồn tham khảo:**\n" + "\n\n".join(source_lines)

        if len(sources_text) <= _MAX_SOURCES_CHARS:
            await ctx.send(sources_text)
        else:
            await ctx.send(
                "**Nguồn tham khảo** (xem file đính kèm):",
                file=discord.File(
                    fp=io.BytesIO(sources_text.encode("utf-8")),
                    filename="sources.txt",
                ),
            )
```

---

### `config/logging.yaml` — Structured Logging Configuration

Load this in `main.py` with `logging.config.dictConfig(yaml.safe_load(open("config/logging.yaml")))`.

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: logs/hcmut_bot.log
    maxBytes: 10485760   # 10 MB
    backupCount: 5
    encoding: utf-8

loggers:
  hcmut_reg_bot:
    level: DEBUG
    handlers: [console, file]
    propagate: false

  discord:
    level: WARNING
    handlers: [console]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

---

## Architectural Decision Log

| Decision | Rationale |
|---|---|
| `typing.Protocol` over ABC | Structural subtyping — infrastructure adapters don't inherit from domain, preventing coupling. Swap any adapter without touching its import tree. |
| Pydantic for all I/O schemas | Runtime validation at system boundaries catches bad data (e.g., empty embeddings, oversized queries) before it reaches business logic. |
| Three distinct system prompts | Comparison queries need tabular output formatting; empty context must explicitly prevent hallucination. A single prompt cannot serve all three cases well. |
| `raise X from exc` everywhere | Preserves original traceback (`__cause__`) in logs while surfacing clean typed domain exceptions to callers. Never lose the root cause. |
| Composition root in `main.py` | One file to swap any adapter (e.g., Milvus → Pinecone, Gemini → OpenAI) by changing a single line. No other file is touched. |
| Retry only in infrastructure | Business logic (`RAGService`) stays clean and synchronous in its error model. Retry is an infrastructure concern — it belongs in the adapter. |
| `async with ctx.typing()` | Shows a Discord typing indicator during the ~3–5s RAG pipeline. Prevents users from thinking the bot is unresponsive. |
| Cache-before-embed in ingestion | Gemini embedding quota is the primary bottleneck. The JSONL cache lets interrupted ingestion runs resume without re-embedding already-processed chunks. |
| Strip embeddings from cache | Embedding vectors are 3072 floats (~24 KB each). Storing them in the JSONL cache would make it unmanageably large. Re-generate only when needed. |
