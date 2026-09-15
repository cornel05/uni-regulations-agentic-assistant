"""Request-scoped access to the composed services, plus the admin guard.

Holds no infrastructure imports: the container is assembled in ``main.py``, the
one place allowed to know both the application and infrastructure layers.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

from application.admin_service import AdminService
from application.ingestion_service import IngestionService
from application.rag_service import RAGService
from config.settings import Settings
from domain.ports import DocumentRegistryPort, SessionStorePort, VectorStorePort


@dataclass(frozen=True)
class Container:
    """Everything the HTTP layer is allowed to reach."""

    settings: Settings
    rag_service: RAGService
    ingestion_service: IngestionService
    admin_service: AdminService
    sessions: SessionStorePort
    vector_store: VectorStorePort
    registry: DocumentRegistryPort


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_settings(container: Container = Depends(get_container)) -> Settings:
    return container.settings


async def require_admin(
    container: Container = Depends(get_container),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """Gate every admin route on a shared secret.

    Fails closed: with no ADMIN_TOKEN configured the admin API is unavailable
    rather than open.
    """
    expected = container.settings.admin_token
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The admin API is disabled. Set ADMIN_TOKEN to enable it.",
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-Admin-Token header."
        )
