"""RAG orchestration: detect language, retrieve, generate, score confidence.

Agnostic of the LLM provider, the vector database, and the transport. Every
dependency arrives as a Protocol implementation, so the whole pipeline is
testable with mocks and no network.
"""

from __future__ import annotations

import logging
import re

from domain.exceptions import EmbeddingError, LLMUnavailableError, RetrievalError
from domain.models import (
    Confidence,
    ConfidenceBand,
    Language,
    RAGRequest,
    RAGResponse,
    RetrievedChunk,
    Turn,
)
from domain.ports import EmbedderPort, LLMPort, RetrieverPort

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_JSON_CONTRACT = """
Trả về JSON đúng ba khóa:
- "answer": câu trả lời dành cho sinh viên.
- "grounding": số thực từ 0 đến 1, cho biết câu trả lời được ngữ cảnh hỗ trợ đầy đủ tới mức nào.
- "used_sources": danh sách số thứ tự các nguồn bạn thực sự dùng (theo nhãn [Nguồn n])."""

_PROMPT_GROUNDED = """Bạn là trợ lý ảo về quy chế và quy định của trường đại học.
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên dựa HOÀN TOÀN vào các đoạn ngữ cảnh bên dưới.

Quy tắc:
1. Chỉ dùng thông tin có trong ngữ cảnh. Không suy diễn, không bịa đặt.
2. Trích dẫn số điều khoản hoặc tên mục khi ngữ cảnh có nêu.
3. Nếu ngữ cảnh không đủ để trả lời trọn vẹn, hãy nói rõ phần nào còn thiếu.
4. Trả lời rõ ràng, súc tích, dùng ngôi "mình" và "bạn".
5. Không nhắc nhãn [Nguồn n] trong câu trả lời — nhãn đó chỉ để bạn đối chiếu. Hãy dẫn
   theo tên văn bản và số điều khoản; danh sách nguồn do giao diện hiển thị riêng dựa
   trên "used_sources"."""

_PROMPT_COMPARISON = """Bạn là trợ lý ảo về quy chế và quy định của trường đại học.
Câu hỏi này yêu cầu SO SÁNH giữa các phiên bản quy định khác nhau.

Quy tắc:
1. Chỉ dùng thông tin có trong ngữ cảnh. Không suy diễn, không bịa đặt.
2. Trình bày theo bốn ý, đúng thứ tự: (1) đối tượng áp dụng, (2) khác biệt về cách
   tính hoặc quy đổi, (3) mốc thời gian hiệu lực và chuyển tiếp, (4) kết luận ngắn
   gọn cho sinh viên.
3. Nếu ngữ cảnh chỉ có một phiên bản, hãy nói rõ điều đó thay vì đoán phiên bản còn lại.
4. Không nhắc nhãn [Nguồn n] trong câu trả lời — hãy dẫn theo tên văn bản và số điều khoản."""

_PROMPT_EMPTY = """Bạn là trợ lý ảo về quy chế và quy định của trường đại học.
Hệ thống không tìm thấy quy định liên quan trong cơ sở dữ liệu.

Hãy thông báo lịch sự rằng bạn không tìm thấy thông tin, tuyệt đối không tự đưa ra quy
định nào, và gợi ý sinh viên liên hệ Phòng Đào tạo hoặc kiểm tra trên cổng thông tin
chính thức của trường. Đặt "grounding" bằng 0."""

_ENGLISH_SUFFIX = """

Answer in English, even though the source documents are in Vietnamese. Keep Vietnamese
proper nouns, document titles, and article labels (for example "Điều 5") as they appear
in the context, so the student can find them in the original document.

Never write the Vietnamese context labels such as "[Nguồn 2]" in your answer — they are
scaffolding for you, not for the reader. Cite the document and article instead."""

_COMPARISON_KEYWORDS: frozenset[str] = frozenset(
    [
        "khác",
        "so sánh",
        "cũ",
        "mới",
        "thay đổi",
        "difference",
        "differences",
        "compare",
        "comparison",
        "changed",
    ]
)

# ── Language detection ────────────────────────────────────────────────────────

_VI_DIACRITICS = re.compile(
    "[ăâđêôơư"
    "àáảãạằắẳẵặầấẩẫậ"
    "èéẻẽẹềếểễệ"
    "ìíỉĩị"
    "òóỏõọồốổỗộờớởỡợ"
    "ùúủũụừứửữự"
    "ỳýỷỹỵ]",
    re.IGNORECASE,
)

_VI_TOKENS = frozenset(
    "la gi bao nhieu the nao co khong duoc sinh vien quy che dinh hoc phi tin chi toi"
    " thieu da nhu cua va hoac neu thi con moi cu khac so sanh thay doi diem trung binh"
    " tich luy canh bao vu dao tao gia han truong mon dang ky tot nghiep bong ky luat".split()
)

_EN_TOKENS = frozenset(
    "the a an is are am do does did i my me you your what how why when where which who"
    " of for to and or if need needs must have has can could should will would about"
    " with without per on in at from than then this that these those there many much"
    " requirement requirements policy policies rule rules".split()
)


def detect_language(text: str) -> Language:
    """Infer the query language. Vietnamese is the default for the source corpus.

    Diacritics are decisive. Without them, the word overlap of the two
    vocabularies decides, which is what makes unaccented Vietnamese
    ("quy che dao tao co gi moi khong") resolve correctly.
    """
    if _VI_DIACRITICS.search(text):
        return Language.VI

    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return Language.VI

    vi_hits = sum(token in _VI_TOKENS for token in tokens)
    en_hits = sum(token in _EN_TOKENS for token in tokens)
    if en_hits > vi_hits:
        return Language.EN
    return Language.VI


class RAGService:
    """Orchestrates embed → hybrid retrieve → grounded generation → confidence."""

    def __init__(
        self,
        embedder: EmbedderPort,
        retriever: RetrieverPort,
        llm: LLMPort,
        *,
        dense_weight: float = 0.5,
        high_threshold: float = 0.8,
        medium_threshold: float = 0.5,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm
        self._dense_weight = dense_weight
        self._high_threshold = high_threshold
        self._medium_threshold = medium_threshold

    async def answer(self, request: RAGRequest) -> RAGResponse:
        language = detect_language(request.query)
        retrieval_query = self._retrieval_query(request.query, request.history)
        logger.info(
            "RAG start | lang=%s top_k=%d domain=%s turns=%d",
            language.value,
            request.top_k,
            request.domain.value if request.domain else "all",
            len(request.history),
        )

        embeddings = [await self._embed(retrieval_query)]
        lexical_query = retrieval_query

        # English queries are answered from Vietnamese documents, so the lexical
        # branch needs a Vietnamese string and the dense branch benefits from a
        # second, same-language vector.
        if language is Language.EN:
            translated = await self._translate(retrieval_query)
            if translated:
                lexical_query = translated
                embeddings.append(await self._embed(translated))

        try:
            chunks = await self._retriever.hybrid_search(
                query_text=lexical_query,
                query_embeddings=embeddings,
                top_k=request.top_k,
                domain=request.domain,
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        is_empty = not chunks
        if is_empty:
            logger.warning("No chunks retrieved | query=%r", request.query[:80])

        system_prompt = self._build_system_prompt(
            has_context=not is_empty,
            is_comparison=self._is_comparison_query(request.query),
            language=language,
        )
        user_message = self._build_user_message(request.query, chunks, request.history)

        try:
            grounded = await self._llm.answer(
                system_prompt=system_prompt, user_message=user_message
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            raise LLMUnavailableError(f"LLM generation failed: {exc}") from exc

        sources = self._select_sources(chunks, grounded.used_sources)
        confidence = self._score_confidence(chunks, grounded.grounding)
        logger.info(
            "RAG done | chunks=%d cited=%d band=%s score=%.3f",
            len(chunks),
            len(sources),
            confidence.band.value,
            confidence.score,
        )

        return RAGResponse(
            query=request.query,
            answer=grounded.answer,
            language=language,
            confidence=confidence,
            sources=sources,
            is_empty_context=is_empty,
        )

    # ── Steps ─────────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        try:
            return await self._embedder.embed_query(text)
        except Exception as exc:
            logger.error("Query embedding failed: %s", exc, exc_info=True)
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    async def _translate(self, text: str) -> str:
        """Best-effort translation. A failure degrades retrieval, not the answer."""
        try:
            return (await self._llm.translate_to_vietnamese(text)).strip()
        except Exception as exc:
            logger.warning("Query translation failed, continuing in English: %s", exc)
            return ""

    @staticmethod
    def _retrieval_query(query: str, history: list[Turn]) -> str:
        """Prepend the previous question so pronoun-only follow-ups still retrieve.

        ponytail: cheap query expansion, no extra model call. Swap in an LLM
        rewrite step if follow-up recall proves insufficient on real traffic.
        """
        prior = [turn.content for turn in history if turn.role == "user"]
        if not prior:
            return query
        return f"{prior[-1]}\n{query}"

    @staticmethod
    def _is_comparison_query(query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in _COMPARISON_KEYWORDS)

    def _build_system_prompt(
        self, *, has_context: bool, is_comparison: bool, language: Language
    ) -> str:
        if not has_context:
            base = _PROMPT_EMPTY
        elif is_comparison:
            base = _PROMPT_COMPARISON
        else:
            base = _PROMPT_GROUNDED
        prompt = base + _JSON_CONTRACT
        if language is Language.EN:
            prompt += _ENGLISH_SUFFIX
        return prompt

    @staticmethod
    def _build_user_message(
        query: str, chunks: list[RetrievedChunk], history: list[Turn]
    ) -> str:
        parts: list[str] = []

        if history:
            turns = "\n".join(
                f"{'Sinh viên' if turn.role == 'user' else 'Trợ lý'}: {turn.content}"
                for turn in history
            )
            parts.append(f"Lịch sử hội thoại:\n{turns}")

        if chunks:
            blocks = [
                f"[Nguồn {index}] {chunk.doc_title}"
                f" — {chunk.section or 'không rõ mục'}, trang {chunk.page_start}\n{chunk.text}"
                for index, chunk in enumerate(chunks, 1)
            ]
            parts.append("Ngữ cảnh:\n\n" + "\n\n---\n\n".join(blocks))
        else:
            parts.append("[Không tìm thấy ngữ cảnh liên quan trong cơ sở dữ liệu quy định]")

        parts.append(f"Câu hỏi: {query}")
        return "\n\n".join(parts)

    @staticmethod
    def _select_sources(
        chunks: list[RetrievedChunk], used_sources: list[int]
    ) -> list[RetrievedChunk]:
        """Cite what the model says it used; fall back to everything retrieved."""
        if not chunks:
            return []
        picked = [
            chunks[index - 1]
            for index in dict.fromkeys(used_sources)
            if 1 <= index <= len(chunks)
        ]
        return picked or list(chunks)

    def _score_confidence(
        self, chunks: list[RetrievedChunk], grounding: float
    ) -> Confidence:
        """Blend retrieval similarity with the model's own grounding rating.

        ponytail: the weight and both cut-offs are settings, not constants — they
        need calibrating against the accuracy benchmark before launch.
        """
        if not chunks:
            return Confidence(band=ConfidenceBand.LOW, score=0.0)

        dense_top = _clamp(max(chunk.dense_score for chunk in chunks))
        score = self._dense_weight * dense_top + (1.0 - self._dense_weight) * _clamp(grounding)

        if score >= self._high_threshold:
            band = ConfidenceBand.HIGH
        elif score >= self._medium_threshold:
            band = ConfidenceBand.MEDIUM
        else:
            band = ConfidenceBand.LOW
        return Confidence(band=band, score=round(_clamp(score), 4))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
