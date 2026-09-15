"""FastAPI application factory.

The scheduled re-crawl is a plain asyncio task on the app lifespan rather than a
scheduler dependency: the requirement is one weekly pass, and a manual trigger
already exists at POST /api/admin/crawl.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from domain.models import RunTrigger
from presentation.api.deps import Container
from presentation.api.routes import admin, chat

logger = logging.getLogger(__name__)


def create_app(container: Container) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await container.vector_store.ensure_collection()
        except Exception as exc:
            # Chat will surface this per-request; refusing to boot would also
            # take down the admin dashboard that diagnoses it.
            logger.warning("Vector store is not ready: %s", exc)

        crawler = asyncio.create_task(_crawl_periodically(container))
        try:
            yield
        finally:
            crawler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await crawler

    app = FastAPI(
        title="Uni-Regulations Agentic Assistant",
        description="Bilingual, cited, confidence-scored answers over official university regulations.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.container = container

    app.include_router(chat.router)
    app.include_router(admin.router)

    @app.get("/api/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    _mount_frontend(app, container.settings.frontend_dist)
    return app


async def _crawl_periodically(container: Container) -> None:
    """Re-crawl on a fixed cadence; hash comparison makes a no-change run cheap."""
    interval = container.settings.crawl_interval_hours * 3600
    if interval <= 0:
        logger.info("Scheduled crawling is disabled")
        return

    while True:
        await asyncio.sleep(interval)
        try:
            result = await container.admin_service.crawl(trigger=RunTrigger.SCHEDULE)
            logger.info(
                "Scheduled crawl complete | changed=%d skipped=%d chunks=%d failed=%d",
                result.changed_count,
                result.skipped_count,
                result.indexed_chunks,
                len(result.failed_titles),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A failed crawl must not kill the loop; the next pass retries.
            logger.error("Scheduled crawl failed: %s", exc, exc_info=True)


def _mount_frontend(app: FastAPI, dist: str) -> None:
    """Serve the built SPA at / when it exists. Mounted last so /api wins."""
    directory = Path(dist)
    if not (directory / "index.html").is_file():
        logger.info("No frontend build at %s — API only (run: cd frontend && npm run build)", dist)
        return
    # html=True serves index.html for unknown paths, which client-side routing needs.
    app.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
    logger.info("Serving frontend from %s", directory)
