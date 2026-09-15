"""Admin endpoints: index status, manual crawl, manual PDF upload.

Every route sits behind the shared-secret guard. The upload route is a trust
boundary, so the file is checked for type, size, and an actual PDF header before
anything downstream touches it.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from application.admin_service import upload_source_url
from domain.exceptions import UniRegulationsError
from domain.models import (
    DocumentRecord,
    IndexStatus,
    IngestionResult,
    RegulationDomain,
    RunTrigger,
)
from presentation.api.deps import Container, get_container, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)]
)

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_PDF_MAGIC = b"%PDF"
_MAGIC_WINDOW = 1024


class UploadOut(BaseModel):
    title: str
    domain: RegulationDomain
    indexed_chunks: int


@router.get("/status", response_model=IndexStatus)
async def index_status(container: Container = Depends(get_container)) -> IndexStatus:
    return await container.admin_service.status()


@router.get("/documents", response_model=list[DocumentRecord])
async def list_documents(container: Container = Depends(get_container)) -> list[DocumentRecord]:
    return container.registry.list_documents()


@router.post("/crawl", response_model=IngestionResult)
async def trigger_crawl(
    container: Container = Depends(get_container),
    only_new: bool = Query(True, description="Skip documents whose content is unchanged"),
    limit: int | None = Query(None, ge=1, description="Cap documents processed this run"),
) -> IngestionResult:
    try:
        return await container.admin_service.crawl(
            trigger=RunTrigger.MANUAL, only_new=only_new, limit=limit
        )
    except UniRegulationsError as exc:
        logger.warning("Manual crawl failed: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Crawl failed: {exc}") from exc


@router.post("/documents", response_model=UploadOut)
async def upload_document(
    container: Container = Depends(get_container),
    file: UploadFile = File(..., description="The regulation PDF"),
    title: str = Form(..., min_length=1, max_length=500),
    domain: RegulationDomain | None = Form(
        None, description="Omit to have the document classified automatically"
    ),
) -> UploadOut:
    """Index a PDF immediately, outside the crawl cycle (US-08)."""
    payload = await file.read()
    _validate_pdf(payload, file)

    try:
        indexed = await container.admin_service.upload(
            filename=file.filename or "document.pdf",
            title=title.strip(),
            pdf_bytes=payload,
            domain=domain,
        )
    except UniRegulationsError as exc:
        logger.warning("Upload failed for %r: %s", title, exc)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Could not index the document: {exc}"
        ) from exc

    record = container.registry.get(upload_source_url(file.filename or "document.pdf"))
    return UploadOut(
        title=title.strip(),
        domain=record.domain if record else (domain or RegulationDomain.OTHER),
        indexed_chunks=indexed,
    )


def _validate_pdf(payload: bytes, file: UploadFile) -> None:
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(payload) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"The file exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    # Trust the bytes, not the name or the declared type.
    if _PDF_MAGIC not in payload[:_MAGIC_WINDOW]:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{file.filename or 'The upload'} is not a PDF.",
        )
