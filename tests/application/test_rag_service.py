"""Behavioral contract for RAGService.

Coverage:
  happy path          — response shape, echoed query, sources, confidence band
  orchestration       — what each port receives, and in what order
  empty context       — graceful decline instead of hallucination
  failures            — every port error becomes a typed domain exception
  language routing    — VI answered directly, EN translated for retrieval
  domain filter       — scope forwarded to the retriever
  session context     — earlier turns reach the prompt
  confidence          — banding from retrieval similarity + LLM self-rating
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from application.rag_service import RAGService, detect_language
from domain.exceptions import EmbeddingError, LLMUnavailableError, RetrievalError
from domain.models import (
    ConfidenceBand,
    GroundedAnswer,
    Language,
    RAGRequest,
    RAGResponse,
    RegulationDomain,
    Turn,
)


def make_service(embedder, retriever, llm, **kwargs) -> RAGService:
    return RAGService(embedder=embedder, retriever=retriever, llm=llm, **kwargs)


def llm_call(mock_llm, index: int = 0):
    """The (system_prompt, user_message) of one llm.answer call."""
    call = mock_llm.answer.call_args_list[index]
    system = call.kwargs.get("system_prompt") or call.args[0]
    user = call.kwargs.get("user_message") or call.args[1]
    return system, user


# ── Language detection ────────────────────────────────────────────────────────


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Điểm trung bình tích lũy tối thiểu là bao nhiêu?",
            "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ?",
            "quy che dao tao co gi moi khong",  # Vietnamese without diacritics
        ],
    )
    def test_detects_vietnamese(self, text):
        assert detect_language(text) is Language.VI

    @pytest.mark.parametrize(
        "text",
        [
            "What GPA do I need to keep my scholarship?",
            "How do I apply for a leave of absence?",
            "Is a thesis mandatory for my programme?",
        ],
    )
    def test_detects_english(self, text):
        assert detect_language(text) is Language.EN

    def test_never_raises_on_degenerate_input(self):
        for text in ["", "   ", "GPA", "123", "???"]:
            assert detect_language(text) in (Language.VI, Language.EN)


# ── Happy path ────────────────────────────────────────────────────────────────


class TestHappyPath:
    async def test_returns_rag_response(self, mock_embedder, mock_retriever, mock_llm, vi_query):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert isinstance(response, RAGResponse)

    async def test_echoes_query(self, mock_embedder, mock_retriever, mock_llm, vi_query):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.query == vi_query

    async def test_answer_is_non_empty(self, mock_embedder, mock_retriever, mock_llm, vi_query):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.answer

    async def test_every_source_carries_a_citation(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        """US-02: source link plus page/section on every returned source."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.sources
        for source in response.sources:
            assert source.source_url
            assert source.doc_title
            assert source.page_start >= 1

    async def test_is_empty_context_false_when_chunks_found(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.is_empty_context is False

    async def test_sources_narrow_to_those_the_llm_used(
        self, mock_embedder, mock_retriever, mock_llm, vi_query, sample_chunks
    ):
        """The LLM reports used_sources=[1], so only that chunk is cited."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert [s.chunk_id for s in response.sources] == [sample_chunks[0].chunk_id]

    async def test_all_chunks_cited_when_llm_reports_none(
        self, mock_embedder, mock_retriever, mock_llm, vi_query, sample_chunks
    ):
        mock_llm.answer.return_value = GroundedAnswer(answer="Có.", grounding=0.8, used_sources=[])
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert len(response.sources) == len(sample_chunks)

    async def test_out_of_range_source_indices_are_ignored(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        mock_llm.answer.return_value = GroundedAnswer(
            answer="Có.", grounding=0.8, used_sources=[1, 99, 0, -3]
        )
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert len(response.sources) == 1


# ── Orchestration ─────────────────────────────────────────────────────────────


class TestOrchestration:
    async def test_embeds_the_query(self, mock_embedder, mock_retriever, mock_llm, vi_query):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        mock_embedder.embed_query.assert_awaited_once_with(vi_query)

    async def test_forwards_embedding_and_top_k(
        self, mock_embedder, mock_retriever, mock_llm, fake_embedding, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query, top_k=3))
        kwargs = mock_retriever.hybrid_search.call_args.kwargs
        assert kwargs["query_embeddings"] == [fake_embedding]
        assert kwargs["top_k"] == 3

    async def test_forwards_raw_query_for_lexical_branch(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        assert mock_retriever.hybrid_search.call_args.kwargs["query_text"] == vi_query

    async def test_llm_called_once(self, mock_embedder, mock_retriever, mock_llm, vi_query):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        assert mock_llm.answer.await_count == 1

    async def test_prompt_carries_query_and_all_context(
        self, mock_embedder, mock_retriever, mock_llm, vi_query, sample_chunks
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        system, user = llm_call(mock_llm)
        assert system
        assert vi_query in user
        for chunk in sample_chunks:
            assert chunk.text in user

    async def test_prompt_labels_sources_so_the_llm_can_cite_them(
        self, mock_embedder, mock_retriever, mock_llm, vi_query, sample_chunks
    ):
        """used_sources indices are only meaningful if blocks are numbered."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        _, user = llm_call(mock_llm)
        assert "1" in user and "2" in user
        assert sample_chunks[0].section in user


# ── Domain filter (US-06) ─────────────────────────────────────────────────────


class TestDomainFilter:
    async def test_domain_forwarded_to_retriever(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(
            RAGRequest(query=vi_query, domain=RegulationDomain.SCHOLARSHIP)
        )
        assert (
            mock_retriever.hybrid_search.call_args.kwargs["domain"]
            is RegulationDomain.SCHOLARSHIP
        )

    async def test_no_domain_means_no_scope(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        assert mock_retriever.hybrid_search.call_args.kwargs["domain"] is None


# ── Bilingual routing (US-01, US-05) ──────────────────────────────────────────


class TestBilingual:
    async def test_vietnamese_query_is_not_translated(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        mock_llm.translate_to_vietnamese.assert_not_awaited()
        assert response.language is Language.VI

    async def test_english_query_is_translated_and_dual_embedded(
        self, mock_embedder, mock_retriever, mock_llm, en_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=en_query))
        mock_llm.translate_to_vietnamese.assert_awaited_once()
        assert response.language is Language.EN
        assert len(mock_retriever.hybrid_search.call_args.kwargs["query_embeddings"]) == 2

    async def test_english_query_searches_lexically_in_vietnamese(
        self, mock_embedder, mock_retriever, mock_llm, en_query
    ):
        """BM25 against Vietnamese documents needs the Vietnamese translation."""
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=en_query))
        translated = mock_llm.translate_to_vietnamese.return_value
        assert mock_retriever.hybrid_search.call_args.kwargs["query_text"] == translated

    async def test_english_prompt_requests_an_english_answer(
        self, mock_embedder, mock_retriever, mock_llm, en_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=en_query))
        system, _ = llm_call(mock_llm)
        assert "english" in system.lower()

    async def test_translation_failure_degrades_to_english_only(
        self, mock_embedder, mock_retriever, mock_llm, en_query
    ):
        """A failed translation must not fail the whole query."""
        mock_llm.translate_to_vietnamese.side_effect = RuntimeError("503")
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=en_query))
        assert response.answer
        assert len(mock_retriever.hybrid_search.call_args.kwargs["query_embeddings"]) == 1


# ── Multi-turn session context (US-04) ────────────────────────────────────────


class TestSessionContext:
    async def test_history_reaches_the_prompt(
        self, mock_embedder, mock_retriever, mock_llm
    ):
        history = [
            Turn(role="user", content="GPA tối thiểu là bao nhiêu?"),
            Turn(role="assistant", content="GPA tối thiểu là 2.0/4.0."),
        ]
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(
            RAGRequest(query="Còn nếu tôi bị điểm F thì sao?", history=history)
        )
        _, user = llm_call(mock_llm)
        assert "GPA tối thiểu là bao nhiêu?" in user

    async def test_followup_retrieval_includes_previous_question(
        self, mock_embedder, mock_retriever, mock_llm
    ):
        """A pronoun-only follow-up cannot be retrieved on its own."""
        history = [
            Turn(role="user", content="Điều kiện nhận học bổng khuyến khích là gì?"),
            Turn(role="assistant", content="Cần GPA từ 3.2 trở lên."),
        ]
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query="Thế còn gia hạn?", history=history))
        embedded = mock_embedder.embed_query.await_args.args[0]
        assert "học bổng" in embedded
        assert "gia hạn" in embedded

    async def test_no_history_embeds_the_query_verbatim(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        assert mock_embedder.embed_query.await_args.args[0] == vi_query


# ── Confidence (US-03) ────────────────────────────────────────────────────────


class TestConfidence:
    async def test_strong_retrieval_and_grounding_is_high(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.confidence.band is ConfidenceBand.HIGH

    async def test_weak_grounding_lowers_the_band(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        mock_llm.answer.return_value = GroundedAnswer(
            answer="Có thể là 2.0.", grounding=0.2, used_sources=[]
        )
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.confidence.band is not ConfidenceBand.HIGH

    async def test_weak_retrieval_lowers_the_band(
        self, mock_embedder, mock_retriever, mock_llm, vi_query, sample_chunks
    ):
        for chunk in sample_chunks:
            chunk.dense_score = 0.2
        mock_retriever.hybrid_search.return_value = sample_chunks
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.confidence.band is not ConfidenceBand.HIGH

    async def test_band_is_always_present_and_score_in_range(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert isinstance(response.confidence.band, ConfidenceBand)
        assert 0.0 <= response.confidence.score <= 1.0

    async def test_thresholds_are_configurable(
        self, mock_embedder, mock_retriever, mock_llm, vi_query
    ):
        strict = make_service(
            mock_embedder, mock_retriever, mock_llm, high_threshold=0.99, medium_threshold=0.98
        )
        response = await strict.answer(RAGRequest(query=vi_query))
        assert response.confidence.band is not ConfidenceBand.HIGH


# ── Empty context ─────────────────────────────────────────────────────────────


class TestEmptyContext:
    async def test_flag_and_empty_sources(
        self, mock_embedder, mock_empty_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.is_empty_context is True
        assert response.sources == []

    async def test_confidence_is_low(
        self, mock_embedder, mock_empty_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert response.confidence.band is ConfidenceBand.LOW

    async def test_llm_still_called_to_decline_politely(
        self, mock_embedder, mock_empty_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        mock_llm.answer.assert_awaited_once()

    async def test_prompt_signals_that_nothing_was_found(
        self, mock_embedder, mock_empty_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        await service.answer(RAGRequest(query=vi_query))
        _, user = llm_call(mock_llm)
        assert "không tìm thấy" in user.lower()

    async def test_response_is_still_valid(
        self, mock_embedder, mock_empty_retriever, mock_llm, vi_query
    ):
        service = make_service(mock_embedder, mock_empty_retriever, mock_llm)
        response = await service.answer(RAGRequest(query=vi_query))
        assert isinstance(response, RAGResponse)
        assert response.answer


# ── Failures ──────────────────────────────────────────────────────────────────


class TestFailures:
    async def test_embedder_failure_raises_embedding_error(
        self, mock_retriever, mock_llm, vi_query
    ):
        embedder = AsyncMock()
        embedder.embed_query.side_effect = RuntimeError("Quota exceeded: 429")
        service = make_service(embedder, mock_retriever, mock_llm)
        with pytest.raises(EmbeddingError):
            await service.answer(RAGRequest(query=vi_query))
        mock_retriever.hybrid_search.assert_not_awaited()

    async def test_retriever_failure_raises_retrieval_error(
        self, mock_embedder, mock_llm, vi_query
    ):
        retriever = AsyncMock()
        retriever.hybrid_search.side_effect = ConnectionError("Zilliz: connection refused")
        service = make_service(mock_embedder, retriever, mock_llm)
        with pytest.raises(RetrievalError):
            await service.answer(RAGRequest(query=vi_query))
        mock_llm.answer.assert_not_awaited()

    async def test_llm_failure_raises_llm_unavailable(
        self, mock_embedder, mock_retriever, vi_query
    ):
        llm = AsyncMock()
        llm.answer.side_effect = TimeoutError("503 Service Unavailable")
        service = make_service(mock_embedder, mock_retriever, llm)
        with pytest.raises(LLMUnavailableError):
            await service.answer(RAGRequest(query=vi_query))

    async def test_original_cause_is_preserved(self, mock_retriever, mock_llm, vi_query):
        original = RuntimeError("root cause")
        embedder = AsyncMock()
        embedder.embed_query.side_effect = original
        service = make_service(embedder, mock_retriever, mock_llm)
        with pytest.raises(EmbeddingError) as info:
            await service.answer(RAGRequest(query=vi_query))
        assert info.value.__cause__ is original


# ── Comparison queries ────────────────────────────────────────────────────────


class TestComparisonQueries:
    @pytest.mark.parametrize(
        "query",
        [
            "Quy định mới khác gì so với quy định cũ?",
            "So sánh quy chế đào tạo 2022 và 2024",
            "What is the difference between the old and new regulations?",
        ],
    )
    async def test_comparison_gets_its_own_prompt(
        self, mock_embedder, mock_retriever, mock_llm, query
    ):
        service = make_service(mock_embedder, mock_retriever, mock_llm)
        await service.answer(RAGRequest(query=query))
        await service.answer(RAGRequest(query="GPA tối thiểu là bao nhiêu?"))
        comparison, _ = llm_call(mock_llm, 0)
        normal, _ = llm_call(mock_llm, 1)
        assert comparison != normal
