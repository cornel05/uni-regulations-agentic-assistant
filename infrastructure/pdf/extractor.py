"""PDF text extraction: PyMuPDF first, a multimodal model as fallback.

Quality scoring decides whether the cheap path succeeded. The legacy score was
built from chunk-size arithmetic, which punished short documents (a valid
300-character regulation scored ~0.49 against a 0.75 threshold and paid for a
needless model call) while saying nothing about the failure it was meant to
catch. It also ran the chunker twice per document just to count chunks.

What actually distinguishes a failed extraction is text *density* — a scanned
page yields almost no characters — and *legibility*, the share of characters that
look like words rather than mojibake. Both are measured directly here, per page,
so document length no longer affects the verdict.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata

import pymupdf
from google import genai
from google.genai import types

from config.settings import Settings
from domain.exceptions import PDFExtractionError
from domain.models import ExtractionMethod, ExtractionResult, PageText

logger = logging.getLogger(__name__)

_MODE_HYBRID = "hybrid"
_MODE_PYMUPDF_ONLY = "pymupdf_only"
_MODE_GEMINI = "gemini"

# Inline request payload ceiling for the generate_content API.
_MAX_INLINE_BYTES = 18 * 1024 * 1024

_PAGE_MARKER = re.compile(r"\[\[\s*page\s*(\d+)\s*\]\]", re.IGNORECASE)

_EXTRACT_PROMPT = """Extract the complete plain text of this university regulation PDF
for search indexing.

Rules:
- Keep the original Vietnamese. Do not translate, summarize, or comment.
- Keep section headings and appendix labels exactly as written (for example "Điều 5",
  "Chương II", "Phụ lục 1"), and keep table contents as readable lines.
- Drop repeated page headers, footers, and bare page numbers.
- Begin every page with a marker on its own line: [[page N]], where N is the page number.

Document title: {title}"""


class HybridPdfExtractor:
    """Concrete PDFExtractorPort.

    Modes (``EXTRACTION_MODE``): ``hybrid`` tries PyMuPDF and falls back on a poor
    score, ``pymupdf_only`` never pays for a model call, ``gemini`` always does.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.main_model
        self._fallback_model = settings.fallback_model
        self._mode = settings.extraction_mode
        self._threshold = settings.pymupdf_quality_threshold
        self._expected_chars_per_page = settings.expected_chars_per_page
        self._max_tokens = settings.extraction_max_output_tokens

    async def extract(self, pdf_bytes: bytes, title: str) -> ExtractionResult:
        if self._mode == _MODE_GEMINI:
            return await self._extract_with_model(pdf_bytes, title)

        try:
            pages = await asyncio.to_thread(_extract_with_pymupdf, pdf_bytes)
        except Exception as exc:
            if self._mode == _MODE_PYMUPDF_ONLY:
                raise PDFExtractionError(f"PyMuPDF failed on {title!r}: {exc}") from exc
            logger.warning("PyMuPDF failed on %r, falling back to the model: %s", title, exc)
            return await self._extract_with_model(pdf_bytes, title)

        quality = score_extraction_quality(
            pages, expected_chars_per_page=self._expected_chars_per_page
        )

        if self._mode == _MODE_PYMUPDF_ONLY:
            if not _has_text(pages):
                raise PDFExtractionError(f"No text extracted from {title!r}")
            return ExtractionResult(
                pages=pages, method=ExtractionMethod.PYMUPDF, quality=quality
            )

        if quality["overall"] >= self._threshold:
            logger.debug("PyMuPDF accepted for %r (score %.2f)", title, quality["overall"])
            return ExtractionResult(
                pages=pages, method=ExtractionMethod.PYMUPDF, quality=quality
            )

        logger.info(
            "PyMuPDF score %.2f below %.2f for %r — using the model",
            quality["overall"],
            self._threshold,
            title,
        )
        return await self._extract_with_model(pdf_bytes, title)

    async def _extract_with_model(self, pdf_bytes: bytes, title: str) -> ExtractionResult:
        if len(pdf_bytes) > _MAX_INLINE_BYTES:
            raise PDFExtractionError(
                f"{title!r} is {len(pdf_bytes) // (1024 * 1024)} MB, above the "
                f"{_MAX_INLINE_BYTES // (1024 * 1024)} MB inline limit; "
                "split the document or index it with EXTRACTION_MODE=pymupdf_only"
            )

        contents = [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            _EXTRACT_PROMPT.format(title=title),
        ]
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=self._max_tokens,
            # No tools are passed; silence the SDK's function-calling notice.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        text = ""
        last_exc: Exception | None = None
        for model in dict.fromkeys([self._model, self._fallback_model]):
            try:
                response = await self._client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("Model %s could not extract %r: %s", model, title, exc)
                continue
            text = (response.text or "").strip()
            if text:
                break

        if not text:
            raise PDFExtractionError(
                f"Model extraction produced no text for {title!r}: {last_exc}"
            ) from last_exc

        pages = _split_marked_pages(text)
        return ExtractionResult(
            pages=pages,
            method=ExtractionMethod.GEMINI,
            quality=score_extraction_quality(
                pages, expected_chars_per_page=self._expected_chars_per_page
            ),
        )


def _extract_with_pymupdf(pdf_bytes: bytes) -> list[PageText]:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [
            PageText(page=number, text=page.get_text("text") or "")
            for number, page in enumerate(document, 1)
        ]
    finally:
        document.close()


def _split_marked_pages(text: str) -> list[PageText]:
    """Split model output on its ``[[page N]]`` markers.

    Without markers the whole document becomes page 1 — citations then name the
    document and section but not a page, which beats inventing one.
    """
    matches = list(_PAGE_MARKER.finditer(text))
    if not matches:
        return [PageText(page=1, text=text)]

    pages: list[PageText] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            pages.append(PageText(page=max(1, int(match.group(1))), text=body))
    return pages or [PageText(page=1, text=text)]


def score_extraction_quality(
    pages: list[PageText], *, expected_chars_per_page: int = 400
) -> dict[str, float]:
    """Rate an extraction on how much legible text it produced per page.

    ``density``    characters per page against what a text-bearing page yields;
                   a scanned page scores near zero.
    ``legibility`` share of characters that are letters, digits, or ordinary
                   punctuation, which collapses on mojibake and control bytes.
    """
    page_count = len(pages)
    text = "".join(page.text for page in pages)
    char_count = len(text.strip())

    if page_count == 0 or char_count == 0:
        return {
            "density": 0.0,
            "legibility": 0.0,
            "overall": 0.0,
            "char_count": float(char_count),
            "page_count": float(page_count),
        }

    expected = max(1, expected_chars_per_page)
    density = min(1.0, (char_count / page_count) / expected)
    legibility = _legibility(text)
    # Weighted evenly on purpose: a page full of mojibake is dense, so a lighter
    # legibility term could never pull it under the acceptance threshold.
    overall = 0.5 * density + 0.5 * legibility

    return {
        "density": round(density, 4),
        "legibility": round(legibility, 4),
        "overall": round(overall, 4),
        "char_count": float(char_count),
        "page_count": float(page_count),
    }


# Whitespace that legitimately appears in extracted text. str.isspace() is too
# generous here: it also accepts the C0 control bytes \x1c-\x1f, which is exactly
# the mojibake this term is meant to detect.
_TEXT_WHITESPACE = frozenset(" \n\t\r")


def _legibility(text: str) -> float:
    """Share of characters that belong to words, numbers, or normal punctuation."""
    sample = text[:20_000]
    if not sample:
        return 0.0
    good = sum(
        1
        for char in sample
        if char.isalnum()
        or char in _TEXT_WHITESPACE
        or unicodedata.category(char).startswith("P")
    )
    return good / len(sample)


def _has_text(pages: list[PageText]) -> bool:
    return any(page.text.strip() for page in pages)
