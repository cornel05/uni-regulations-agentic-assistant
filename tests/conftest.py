"""Shared fixtures. Every external dependency is a mock — tests run offline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.models import (
    DocumentRecord,
    DocumentStatus,
    ExtractionMethod,
    ExtractionResult,
    GroundedAnswer,
    PageText,
    RegulationDomain,
    RegulationSource,
    RetrievedChunk,
    TextChunk,
)

EMBEDDING_DIM = 3072


# ── Data ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def vi_query() -> str:
    return "Điểm trung bình tích lũy tối thiểu để không bị cảnh báo học vụ là bao nhiêu?"


@pytest.fixture
def en_query() -> str:
    return "What GPA do I need to keep my scholarship?"


@pytest.fixture
def fake_embedding() -> list[float]:
    return [0.1] * EMBEDDING_DIM


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    """Two chunks as hybrid retrieval would return them."""
    return [
        RetrievedChunk(
            chunk_id="docabc:0",
            text=(
                "Điều 5. Điểm trung bình tích lũy (GPA): Sinh viên phải đạt GPA "
                "tối thiểu 2.0/4.0 để không bị cảnh báo học vụ."
            ),
            source_url="https://drive.google.com/file/d/abc123/view",
            doc_title="Quy chế đào tạo đại học HCMUT 2024",
            domain=RegulationDomain.COURSE_CURRICULUM,
            section="Điều 5",
            page_start=3,
            page_end=3,
            doc_updated_at="2024-08-01",
            score=0.032,
            dense_score=0.92,
        ),
        RetrievedChunk(
            chunk_id="docabc:1",
            text=(
                "Điều 6. Cảnh báo học vụ: Sinh viên bị cảnh báo khi GPA dưới 1.5 "
                "trong hai học kỳ liên tiếp."
            ),
            source_url="https://drive.google.com/file/d/abc123/view",
            doc_title="Quy chế đào tạo đại học HCMUT 2024",
            domain=RegulationDomain.COURSE_CURRICULUM,
            section="Điều 6",
            page_start=4,
            page_end=4,
            doc_updated_at="2024-08-01",
            score=0.016,
            dense_score=0.87,
        ),
    ]


@pytest.fixture
def sample_pages() -> list[PageText]:
    return [
        PageText(page=1, text="Điều 1. Phạm vi áp dụng\n\nQuy chế này áp dụng cho sinh viên."),
        PageText(page=2, text="Điều 2. Đăng ký môn học\n\nSinh viên đăng ký tối đa 24 tín chỉ."),
    ]


@pytest.fixture
def sample_text_chunks() -> list[TextChunk]:
    return [
        TextChunk(text="Điều 1. Phạm vi áp dụng", page_start=1, page_end=1, section="Điều 1"),
        TextChunk(text="Điều 2. Đăng ký môn học", page_start=2, page_end=2, section="Điều 2"),
    ]


@pytest.fixture
def sample_sources() -> list[RegulationSource]:
    return [
        RegulationSource(title="Quy chế đào tạo 2024", link="https://drive.google.com/file/d/a/view"),
        RegulationSource(title="Quy định học bổng", link="https://drive.google.com/file/d/b/view"),
    ]


# ── Port mocks ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_embedder(fake_embedding):
    embedder = AsyncMock()
    embedder.embed_query.return_value = fake_embedding
    embedder.embed_documents.side_effect = lambda texts: [fake_embedding for _ in texts]
    return embedder


@pytest.fixture
def mock_retriever(sample_chunks):
    retriever = AsyncMock()
    retriever.hybrid_search.return_value = sample_chunks
    return retriever


@pytest.fixture
def mock_empty_retriever():
    """Retrieval that finds nothing — no regulation covers the question."""
    retriever = AsyncMock()
    retriever.hybrid_search.return_value = []
    return retriever


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.answer.return_value = GroundedAnswer(
        answer="Theo Điều 5, GPA tối thiểu là 2.0/4.0.",
        grounding=0.9,
        used_sources=[1],
    )
    llm.classify_domain.return_value = RegulationDomain.COURSE_CURRICULUM
    llm.translate_to_vietnamese.return_value = "GPA tối thiểu để giữ học bổng là bao nhiêu?"
    return llm


@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.upsert.side_effect = lambda chunks: len(chunks)
    store.count.return_value = 42
    return store


@pytest.fixture
def mock_scraper(sample_sources):
    scraper = AsyncMock()
    scraper.scrape.return_value = sample_sources
    return scraper


@pytest.fixture
def mock_downloader():
    downloader = AsyncMock()
    downloader.download.side_effect = lambda url: f"%PDF-1.4 {url}".encode()
    return downloader


@pytest.fixture
def mock_extractor(sample_pages):
    extractor = AsyncMock()
    extractor.extract.return_value = ExtractionResult(
        pages=sample_pages,
        method=ExtractionMethod.PYMUPDF,
        quality={"overall": 0.9},
    )
    return extractor


@pytest.fixture
def mock_chunker(sample_text_chunks):
    chunker = MagicMock()
    chunker.chunk.return_value = sample_text_chunks
    return chunker


@pytest.fixture
def mock_registry():
    """In-memory stand-in for the SQLite registry."""
    registry = MagicMock()
    store: dict[str, DocumentRecord] = {}
    registry._store = store
    registry.get.side_effect = store.get
    registry.upsert.side_effect = lambda record: store.__setitem__(record.source_url, record)
    registry.list_documents.side_effect = lambda: list(store.values())
    registry.start_run.return_value = 1
    return registry


@pytest.fixture
def indexed_record():
    """A document already indexed, for hash-comparison tests."""

    def _make(source_url: str, content_hash: str) -> DocumentRecord:
        return DocumentRecord(
            source_url=source_url,
            doc_id="deadbeef",
            title="Quy chế đào tạo 2024",
            domain=RegulationDomain.COURSE_CURRICULUM,
            content_hash=content_hash,
            chunk_count=2,
            status=DocumentStatus.INDEXED,
        )

    return _make
