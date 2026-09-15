"""Centralized configuration loaded from environment variables / .env.

Single source of truth: no other module reads os.environ directly.
Field names double as env var names (case-insensitive), so the legacy .env
keys (GEMINI_API_KEY, MAIN_MODEL, CHUNK_SIZE, ZILLIZ_CLOUD_*) keep working.
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

    # ── Credentials ───────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    zilliz_cloud_endpoint: str = Field(..., description="Zilliz Cloud public endpoint")
    zilliz_cloud_api_key: str = Field(..., description="Zilliz Cloud API key")
    admin_token: str = Field(
        "",
        description="Shared secret for /api/admin/*. Empty disables the admin API.",
    )

    # ── Models ────────────────────────────────────────────────────────────────
    main_model: str = Field("gemini-2.5-flash", description="Primary generation model")
    fallback_model: str = Field("gemini-2.5-flash", description="Used when main model fails")
    embedding_model: str = Field("gemini-embedding-001")
    embedding_dimension: int = Field(3072, ge=1)
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(3072, ge=256, le=65536)

    # ── Vector store ──────────────────────────────────────────────────────────
    milvus_collection_name: str = Field("uni_regulations_v1")
    enable_sparse: bool = Field(
        True, description="Hybrid BM25 retrieval. False falls back to dense-only."
    )
    upsert_batch_size: int = Field(32, ge=1)

    # ── Retrieval / RAG ───────────────────────────────────────────────────────
    retrieval_k: int = Field(5, ge=1, le=50)
    rrf_k: int = Field(60, ge=1, description="Reciprocal-rank-fusion damping constant")
    confidence_dense_weight: float = Field(
        0.5, ge=0.0, le=1.0, description="Weight of retrieval similarity vs LLM self-rating"
    )
    confidence_high_threshold: float = Field(0.8, ge=0.0, le=1.0)
    confidence_medium_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # ── Sessions ──────────────────────────────────────────────────────────────
    session_max_turns: int = Field(5, ge=1)
    session_ttl_seconds: int = Field(3600, ge=60)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    regulation_page_url: str = Field("https://hcmut.edu.vn/dao-tao/quy-che-quy-dinh")
    database_path: str = Field("database/regulations.db")
    chunk_size: int = Field(1200, ge=200, le=8000)
    chunk_overlap: int = Field(200, ge=0)
    extraction_mode: str = Field(
        "hybrid", description="'hybrid' | 'pymupdf_only' | 'gemini'"
    )
    pymupdf_quality_threshold: float = Field(0.75, ge=0.0, le=1.0)
    expected_chars_per_page: int = Field(
        400,
        ge=1,
        description="Characters a text-bearing page should yield; drives extraction scoring",
    )
    extraction_max_output_tokens: int = Field(
        32768, ge=1024, description="Output cap for model-based PDF text extraction"
    )
    embedding_retry_count: int = Field(3, ge=1)
    embedding_retry_delay: float = Field(1.5, ge=0.0)
    llm_retry_count: int = Field(2, ge=1)
    llm_retry_delay: float = Field(
        2.0,
        ge=0.0,
        description=(
            "Base seconds between LLM retries, multiplied by the attempt number. "
            "A chat request waits on this, so it is kept small: the legacy value of "
            "20s meant a transient 503 cost 40s before the answer failed."
        ),
    )

    # ── Crawler ───────────────────────────────────────────────────────────────
    headless_mode: bool = Field(True)
    wait_time: int = Field(20, ge=1, description="Max seconds to wait for the JS-rendered list")
    delay_between_requests: float = Field(5.0, ge=0.0)
    pdf_download_timeout: int = Field(60, ge=1)
    crawl_interval_hours: int = Field(168, ge=1, description="Scheduled re-crawl cadence")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field("127.0.0.1")
    port: int = Field(8000, ge=1, le=65535)
    frontend_dist: str = Field("frontend/dist")
