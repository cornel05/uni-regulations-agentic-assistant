"""Canonical schemas — the shared language between all layers.

No layer imports another layer's internals; everything crossing a boundary is
one of these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ── Enums ─────────────────────────────────────────────────────────────────────


class RegulationDomain(str, Enum):
    """The five regulatory domains in scope, plus a catch-all."""

    COURSE_CURRICULUM = "COURSE_CURRICULUM"
    GRADUATION = "GRADUATION"
    SCHOLARSHIP = "SCHOLARSHIP"
    DISCIPLINARY = "DISCIPLINARY"
    ADMISSION_ENROLLMENT = "ADMISSION_ENROLLMENT"
    OTHER = "OTHER"


class Language(str, Enum):
    VI = "vi"
    EN = "en"


class ConfidenceBand(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ExtractionMethod(str, Enum):
    PYMUPDF = "pymupdf"
    GEMINI = "gemini"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class RunStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class RunTrigger(str, Enum):
    SCHEDULE = "schedule"
    MANUAL = "manual"
    UPLOAD = "upload"


# ── Ingestion ─────────────────────────────────────────────────────────────────


class RegulationSource(BaseModel):
    """One regulation entry as listed on the university portal."""

    title: str
    link: str


class PageText(BaseModel):
    """Text of a single PDF page. Page numbers are 1-based."""

    page: int = Field(..., ge=1)
    text: str


class ExtractionResult(BaseModel):
    """Output of the PDF text extractor."""

    pages: list[PageText]
    method: ExtractionMethod
    quality: dict[str, float] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


class TextChunk(BaseModel):
    """A chunk as produced by the chunker — text plus where it came from.

    Identity and document metadata are attached later by the ingestion service.
    """

    text: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    section: str = Field("", description="e.g. 'Điều 5', 'Chương II', 'Phụ lục 1'")


class Chunk(BaseModel):
    """A fully identified chunk, ready to embed and index."""

    chunk_id: str = Field(..., description="'{doc_id}:{chunk_index}'")
    doc_id: str = Field(..., description="sha1 of the source URL")
    text: str
    source_url: str
    doc_title: str
    domain: RegulationDomain
    chunk_index: int = Field(..., ge=0)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    section: str = ""
    doc_updated_at: str = Field("", description="ISO date of the source document")
    embedding: list[float] | None = None


class IngestionRequest(BaseModel):
    only_new: bool = Field(True, description="Skip documents whose content hash is unchanged")
    limit: int | None = Field(None, ge=1, description="Cap on documents processed this run")
    trigger: RunTrigger = RunTrigger.MANUAL


class IngestionResult(BaseModel):
    scraped_count: int = 0
    changed_count: int = 0
    skipped_count: int = 0
    indexed_chunks: int = 0
    failed_titles: list[str] = Field(default_factory=list)


# ── Registry ──────────────────────────────────────────────────────────────────


class DocumentRecord(BaseModel):
    source_url: str
    doc_id: str
    title: str
    domain: RegulationDomain = RegulationDomain.OTHER
    content_hash: str = ""
    doc_updated_at: str = ""
    extraction_method: ExtractionMethod | None = None
    chunk_count: int = 0
    last_crawled_at: str = ""
    last_indexed_at: str = ""
    status: DocumentStatus = DocumentStatus.PENDING


class IngestionRunRecord(BaseModel):
    id: int
    started_at: str
    finished_at: str = ""
    trigger: RunTrigger
    status: RunStatus
    scraped_count: int = 0
    changed_count: int = 0
    indexed_chunks: int = 0
    error: str = ""


class IndexStatus(BaseModel):
    """Payload behind the admin dashboard."""

    total_documents: int = 0
    indexed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    documents_per_domain: dict[str, int] = Field(default_factory=dict)
    last_crawled_at: str = ""
    last_indexed_at: str = ""
    recent_runs: list[IngestionRunRecord] = Field(default_factory=list)


# ── RAG ───────────────────────────────────────────────────────────────────────


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, with its fused and dense scores."""

    chunk_id: str
    text: str
    source_url: str
    doc_title: str
    domain: RegulationDomain = RegulationDomain.OTHER
    section: str = ""
    page_start: int = 1
    page_end: int = 1
    doc_updated_at: str = ""
    score: float = Field(..., description="Fused ranking score (RRF)")
    dense_score: float = Field(
        0.0, description="Cosine similarity from the dense branch; feeds confidence"
    )


class GroundedAnswer(BaseModel):
    """Structured LLM output: the answer plus its own grounding self-rating."""

    answer: str
    grounding: float = Field(
        0.0, ge=0.0, le=1.0, description="How fully the answer is supported by the context"
    )
    used_sources: list[int] = Field(
        default_factory=list, description="1-based indices of the context blocks actually used"
    )


class Confidence(BaseModel):
    band: ConfidenceBand
    score: float = Field(..., ge=0.0, le=1.0)


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    domain: RegulationDomain | None = Field(None, description="Scope retrieval to one domain")
    history: list[Turn] = Field(default_factory=list, description="Earlier turns, oldest first")


class RAGResponse(BaseModel):
    query: str
    answer: str
    language: Language
    confidence: Confidence
    sources: list[RetrievedChunk] = Field(default_factory=list)
    is_empty_context: bool = False


def utc_now_iso() -> str:
    """Timestamp helper so every layer writes the same format."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
