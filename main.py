"""Composition root and CLI.

The only module that imports both the application and infrastructure layers.
Swapping an adapter — a different vector database, a different model provider —
is a change to `build_container` and nothing else.

    python main.py serve                 # API + frontend
    python main.py ingest --limit 1      # crawl and index
    python main.py status                # what is indexed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

from application.admin_service import AdminService
from application.ingestion_service import IngestionService
from application.rag_service import RAGService
from config.settings import Settings
from domain.models import IngestionRequest, RunTrigger
from infrastructure.chunking.section_chunker import SectionChunker
from infrastructure.embeddings.gemini_embedder import GeminiEmbedder
from infrastructure.llm.gemini_llm import GeminiLLM
from infrastructure.pdf.downloader import PdfDownloader
from infrastructure.pdf.extractor import HybridPdfExtractor
from infrastructure.registry.sqlite_registry import SQLiteDocumentRegistry
from infrastructure.scrapers.hcmut_scraper import HcmutScraper
from infrastructure.sessions.memory_store import MemorySessionStore
from infrastructure.vector_db.milvus_store import MilvusStore
from presentation.api.app import create_app
from presentation.api.deps import Container

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("WDM").setLevel(logging.WARNING)


def build_container(settings: Settings) -> Container:
    """Wire every adapter to its port."""
    embedder = GeminiEmbedder(settings)
    llm = GeminiLLM(settings)
    vector_store = MilvusStore(settings)
    registry = SQLiteDocumentRegistry(settings.database_path)

    rag_service = RAGService(
        embedder=embedder,
        retriever=vector_store,
        llm=llm,
        dense_weight=settings.confidence_dense_weight,
        high_threshold=settings.confidence_high_threshold,
        medium_threshold=settings.confidence_medium_threshold,
    )

    ingestion_service = IngestionService(
        scraper=HcmutScraper(settings),
        downloader=PdfDownloader(settings),
        extractor=HybridPdfExtractor(settings),
        chunker=SectionChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        registry=registry,
        delay_between_requests=settings.delay_between_requests,
    )

    return Container(
        settings=settings,
        rag_service=rag_service,
        ingestion_service=ingestion_service,
        admin_service=AdminService(registry, vector_store, ingestion_service),
        sessions=MemorySessionStore(settings.session_max_turns, settings.session_ttl_seconds),
        vector_store=vector_store,
        registry=registry,
    )


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        raise SystemExit(
            f"Configuration error — missing or invalid: {missing}\n"
            "Copy .env.example to .env and fill in the required values."
        ) from exc


def create_application() -> FastAPI:
    """Entry point for an ASGI server: `uvicorn main:create_application --factory`."""
    settings = load_settings()
    return create_app(build_container(settings))


# ── Commands ──────────────────────────────────────────────────────────────────


def _serve(settings: Settings, container: Container, args: argparse.Namespace) -> int:
    app = create_app(container)
    logger.info("Serving on http://%s:%d", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)
    return 0


def _ingest(settings: Settings, container: Container, args: argparse.Namespace) -> int:
    result = asyncio.run(
        container.ingestion_service.run(
            IngestionRequest(
                only_new=not args.all, limit=args.limit, trigger=RunTrigger.MANUAL
            )
        )
    )
    print(
        f"scraped={result.scraped_count} changed={result.changed_count} "
        f"skipped={result.skipped_count} chunks={result.indexed_chunks}"
    )
    for title in result.failed_titles:
        print(f"  failed: {title}")
    return 1 if result.failed_titles and not result.changed_count else 0


def _status(settings: Settings, container: Container, args: argparse.Namespace) -> int:
    status = asyncio.run(container.admin_service.status())
    print(
        f"documents={status.total_documents} indexed={status.indexed_documents} "
        f"failed={status.failed_documents} chunks={status.total_chunks}"
    )
    print(f"last crawled: {status.last_crawled_at or '-'}")
    print(f"last indexed: {status.last_indexed_at or '-'}")
    for domain, count in sorted(status.documents_per_domain.items()):
        print(f"  {domain}: {count}")
    for record in container.registry.list_documents():
        pages = f"{record.chunk_count} chunks"
        print(f"  [{record.status.value}] {record.title} ({record.domain.value}, {pages})")
    return 0


def main(argv: list[str] | None = None) -> int:
    # SUPPRESS keeps the flag off the namespace unless given, so a subcommand's
    # copy cannot clobber a value already set before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="debug logging",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("serve", help="run the API and frontend", parents=[common])

    ingest = commands.add_parser(
        "ingest", help="crawl and index regulations", parents=[common]
    )
    ingest.add_argument("--limit", type=int, default=None, help="cap documents this run")
    ingest.add_argument(
        "--all", action="store_true", help="re-index everything, ignoring content hashes"
    )

    commands.add_parser("status", help="show what is indexed", parents=[common])

    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))

    settings = load_settings()
    container = build_container(settings)

    handlers = {"serve": _serve, "ingest": _ingest, "status": _status}
    return handlers[args.command](settings, container, args)


if __name__ == "__main__":
    sys.exit(main())
