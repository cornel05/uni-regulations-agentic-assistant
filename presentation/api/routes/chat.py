"""Chat endpoints: ask a question, drop a session, list the domains."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from domain.exceptions import (
    EmbeddingError,
    LLMUnavailableError,
    RetrievalError,
    UniRegulationsError,
)
from domain.models import ConfidenceBand, Language, RAGRequest, RegulationDomain
from presentation.api.deps import Container, get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

_DOMAIN_LABELS: dict[RegulationDomain, tuple[str, str]] = {
    RegulationDomain.COURSE_CURRICULUM: ("Môn học & chương trình", "Course & Curriculum"),
    RegulationDomain.GRADUATION: ("Tốt nghiệp", "Graduation Requirements"),
    RegulationDomain.SCHOLARSHIP: ("Học bổng & hỗ trợ tài chính", "Scholarship & Financial Aid"),
    RegulationDomain.DISCIPLINARY: ("Kỷ luật & đạo đức học thuật", "Disciplinary & Conduct"),
    RegulationDomain.ADMISSION_ENROLLMENT: ("Tuyển sinh & đăng ký", "Admission & Enrollment"),
}

_ERROR_MESSAGES: dict[type[UniRegulationsError], str] = {
    EmbeddingError: (
        "Không xử lý được câu hỏi, vui lòng thử lại. "
        "/ The question could not be processed, please try again."
    ),
    RetrievalError: (
        "Hệ thống tìm kiếm tạm thời gặp sự cố, vui lòng thử lại sau. "
        "/ Search is temporarily unavailable, please try again shortly."
    ),
    LLMUnavailableError: (
        "Hệ thống AI đang quá tải, vui lòng thử lại sau vài phút. "
        "/ The AI service is busy, please retry in a few minutes."
    ),
}

_LOW_CONFIDENCE_DISCLAIMER = (
    "Độ tin cậy thấp — hãy xác nhận lại với Phòng Đào tạo hoặc cố vấn học tập trước khi "
    "hành động. / Low confidence: please confirm with the Academic Affairs Office or your "
    "academic advisor before acting on this answer."
)


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="Omit to start a new session")
    domain: RegulationDomain | None = Field(None, description="Scope the search to one domain")


class SourceOut(BaseModel):
    doc_title: str
    source_url: str
    section: str
    page_start: int
    page_end: int
    doc_updated_at: str
    score: float


class ConfidenceOut(BaseModel):
    band: ConfidenceBand
    score: float


class ChatOut(BaseModel):
    session_id: str
    answer: str
    language: Language
    confidence: ConfidenceOut
    sources: list[SourceOut]
    is_empty_context: bool
    disclaimer: str | None = Field(None, description="Present when confidence is low")


class DomainOut(BaseModel):
    value: RegulationDomain
    label_vi: str
    label_en: str


@router.post("/chat", response_model=ChatOut)
async def chat(body: ChatBody, container: Container = Depends(get_container)) -> ChatOut:
    sessions = container.sessions
    session_id = body.session_id or sessions.create()

    try:
        response = await container.rag_service.answer(
            RAGRequest(
                query=body.message,
                top_k=container.settings.retrieval_k,
                domain=body.domain,
                history=sessions.history(session_id),
            )
        )
    except UniRegulationsError as exc:
        logger.warning("Chat failed | %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            _ERROR_MESSAGES.get(type(exc), "Đã xảy ra lỗi. / Something went wrong."),
        ) from exc

    sessions.append(session_id, body.message, response.answer)

    return ChatOut(
        session_id=session_id,
        answer=response.answer,
        language=response.language,
        confidence=ConfidenceOut(
            band=response.confidence.band, score=response.confidence.score
        ),
        sources=[
            SourceOut(
                doc_title=source.doc_title,
                source_url=source.source_url,
                section=source.section,
                page_start=source.page_start,
                page_end=source.page_end,
                doc_updated_at=source.doc_updated_at,
                score=round(source.dense_score, 4),
            )
            for source in response.sources
        ],
        is_empty_context=response.is_empty_context,
        disclaimer=(
            _LOW_CONFIDENCE_DISCLAIMER
            if response.confidence.band is ConfidenceBand.LOW
            else None
        ),
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(
    session_id: str, container: Container = Depends(get_container)
) -> None:
    """Forget a conversation — the PRD's 'clear history' control."""
    container.sessions.clear(session_id)


@router.get("/domains", response_model=list[DomainOut])
async def list_domains() -> list[DomainOut]:
    """Domains offered in the filter sidebar. OTHER is deliberately not offered."""
    return [
        DomainOut(value=domain, label_vi=labels[0], label_en=labels[1])
        for domain, labels in _DOMAIN_LABELS.items()
    ]
