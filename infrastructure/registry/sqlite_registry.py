"""SQLite document registry — what has been crawled, indexed, or failed.

Replaces the legacy ``seen_laws.json`` / ``processor_seen_laws.json`` /
``processed_records.jsonl`` trio. One durable store, queryable for the admin
dashboard, and idempotent: every write is keyed by ``source_url``.

A fresh connection is opened per operation. SQLite connections have thread
affinity and this registry is called from both the request loop and the
background crawl task; per-operation connections sidestep that entirely, and for
a local file the cost is negligible.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from domain.exceptions import RegistryError
from domain.models import (
    DocumentRecord,
    DocumentStatus,
    ExtractionMethod,
    IndexStatus,
    IngestionRunRecord,
    RegulationDomain,
    RunStatus,
    RunTrigger,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source_url        TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL UNIQUE,
    title             TEXT NOT NULL,
    domain            TEXT NOT NULL DEFAULT 'OTHER',
    content_hash      TEXT NOT NULL DEFAULT '',
    doc_updated_at    TEXT NOT NULL DEFAULT '',
    extraction_method TEXT NOT NULL DEFAULT '',
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    last_crawled_at   TEXT NOT NULL DEFAULT '',
    last_indexed_at   TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS documents_domain_idx ON documents (domain);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT NOT NULL,
    finished_at    TEXT NOT NULL DEFAULT '',
    trigger        TEXT NOT NULL,
    status         TEXT NOT NULL,
    scraped_count  INTEGER NOT NULL DEFAULT 0,
    changed_count  INTEGER NOT NULL DEFAULT 0,
    indexed_chunks INTEGER NOT NULL DEFAULT 0,
    error          TEXT NOT NULL DEFAULT ''
);
"""

_RECENT_RUNS = 10


class SQLiteDocumentRegistry:
    """Concrete DocumentRegistryPort backed by a local SQLite file."""

    def __init__(self, database_path: str) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(self._path, timeout=10)
        except sqlite3.Error as exc:
            raise RegistryError(f"Cannot open registry at {self._path}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("PRAGMA journal_mode=WAL")
                yield connection
        except sqlite3.Error as exc:
            raise RegistryError(f"Registry operation failed: {exc}") from exc
        finally:
            connection.close()

    # ── Documents ─────────────────────────────────────────────────────────────

    def get(self, source_url: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE source_url = ?", (source_url,)
            ).fetchone()
        return _to_record(row) if row else None

    def upsert(self, record: DocumentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    source_url, doc_id, title, domain, content_hash, doc_updated_at,
                    extraction_method, chunk_count, last_crawled_at, last_indexed_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_url) DO UPDATE SET
                    doc_id            = excluded.doc_id,
                    title             = excluded.title,
                    domain            = excluded.domain,
                    content_hash      = excluded.content_hash,
                    doc_updated_at    = excluded.doc_updated_at,
                    extraction_method = excluded.extraction_method,
                    chunk_count       = excluded.chunk_count,
                    last_crawled_at   = excluded.last_crawled_at,
                    last_indexed_at   = excluded.last_indexed_at,
                    status            = excluded.status
                """,
                (
                    record.source_url,
                    record.doc_id,
                    record.title,
                    record.domain.value,
                    record.content_hash,
                    record.doc_updated_at,
                    record.extraction_method.value if record.extraction_method else "",
                    record.chunk_count,
                    record.last_crawled_at,
                    record.last_indexed_at,
                    record.status.value,
                ),
            )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY title COLLATE NOCASE"
            ).fetchall()
        return [_to_record(row) for row in rows]

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def index_status(self, total_chunks: int) -> IndexStatus:
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT
                    COUNT(*)                                             AS total,
                    COALESCE(SUM(status = 'indexed'), 0)                 AS indexed,
                    COALESCE(SUM(status = 'failed'), 0)                  AS failed,
                    COALESCE(MAX(last_crawled_at), '')                   AS last_crawled_at,
                    COALESCE(MAX(last_indexed_at), '')                   AS last_indexed_at
                FROM documents
                """
            ).fetchone()
            per_domain = connection.execute(
                "SELECT domain, COUNT(*) AS n FROM documents GROUP BY domain ORDER BY domain"
            ).fetchall()
            runs = connection.execute(
                "SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT ?", (_RECENT_RUNS,)
            ).fetchall()

        return IndexStatus(
            total_documents=totals["total"],
            indexed_documents=totals["indexed"],
            failed_documents=totals["failed"],
            total_chunks=total_chunks,
            documents_per_domain={row["domain"]: row["n"] for row in per_domain},
            last_crawled_at=totals["last_crawled_at"],
            last_indexed_at=totals["last_indexed_at"],
            recent_runs=[_to_run(row) for row in runs],
        )

    # ── Runs ──────────────────────────────────────────────────────────────────

    def start_run(self, trigger: RunTrigger) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO ingestion_runs (started_at, trigger, status) VALUES (?, ?, ?)",
                (utc_now_iso(), trigger.value, RunStatus.STARTED.value),
            )
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: RunStatus,
        scraped_count: int = 0,
        changed_count: int = 0,
        indexed_chunks: int = 0,
        error: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                   SET finished_at = ?, status = ?, scraped_count = ?,
                       changed_count = ?, indexed_chunks = ?, error = ?
                 WHERE id = ?
                """,
                (
                    utc_now_iso(),
                    status.value,
                    scraped_count,
                    changed_count,
                    indexed_chunks,
                    error[:2000],
                    run_id,
                ),
            )


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        source_url=row["source_url"],
        doc_id=row["doc_id"],
        title=row["title"],
        domain=_enum_or_default(RegulationDomain, row["domain"], RegulationDomain.OTHER),
        content_hash=row["content_hash"],
        doc_updated_at=row["doc_updated_at"],
        extraction_method=_enum_or_default(ExtractionMethod, row["extraction_method"], None),
        chunk_count=row["chunk_count"],
        last_crawled_at=row["last_crawled_at"],
        last_indexed_at=row["last_indexed_at"],
        status=_enum_or_default(DocumentStatus, row["status"], DocumentStatus.PENDING),
    )


def _to_run(row: sqlite3.Row) -> IngestionRunRecord:
    return IngestionRunRecord(
        id=row["id"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        trigger=_enum_or_default(RunTrigger, row["trigger"], RunTrigger.MANUAL),
        status=_enum_or_default(RunStatus, row["status"], RunStatus.FAILED),
        scraped_count=row["scraped_count"],
        changed_count=row["changed_count"],
        indexed_chunks=row["indexed_chunks"],
        error=row["error"],
    )


def _enum_or_default(enum_cls, value, default):
    """Stored strings are data; an unknown one must not crash the dashboard."""
    try:
        return enum_cls(value)
    except ValueError:
        return default
