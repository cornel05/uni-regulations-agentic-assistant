"""Page- and section-aware chunker.

Two things the legacy chunker got wrong, fixed here:

* Overlap applied only when a single paragraph exceeded the chunk size, so the
  configured 200-character overlap did not exist for normal documents. Here every
  chunk after the first is seeded with the tail of its predecessor.
* Provenance was dropped, so answers could not cite a page. Here each chunk
  records the page range it spans and the nearest enclosing section heading.

Because the overlap is prepended, a chunk's text may exceed ``chunk_size`` by up
to ``chunk_overlap`` characters. That is additive context, not a packing bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from domain.models import PageText, TextChunk

_SECTION_PATTERN = re.compile(
    r"^\s*((?:Điều|Chương|Mục|Phần|Phụ\s*lục)\s+[0-9IVXLCDM]+[A-Za-z]?)\b",
    re.IGNORECASE,
)

_PARAGRAPH_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class _Unit:
    """One packing unit: a paragraph (or a slice of an oversized one)."""

    text: str
    page: int
    section: str


class SectionChunker:
    """Splits page texts into overlapping chunks that keep their provenance."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._size = chunk_size
        self._overlap = chunk_overlap

    def chunk(self, pages: list[PageText]) -> list[TextChunk]:
        units = self._units(pages)
        if not units:
            return []

        chunks: list[TextChunk] = []
        buffer: list[_Unit] = []
        buffer_len = 0
        carry = ""

        def flush() -> None:
            nonlocal buffer, buffer_len, carry
            if not buffer:
                return
            body = _PARAGRAPH_SEPARATOR.join(unit.text for unit in buffer)
            text = f"{carry}{_PARAGRAPH_SEPARATOR}{body}" if carry else body
            chunks.append(
                TextChunk(
                    text=text,
                    # Pages come from real units only; the carried overlap belongs
                    # to the previous chunk and carries no page attribution.
                    page_start=buffer[0].page,
                    page_end=buffer[-1].page,
                    section=buffer[0].section,
                )
            )
            carry = self._tail(text)
            buffer = []
            buffer_len = 0

        for unit in units:
            projected = buffer_len + len(_PARAGRAPH_SEPARATOR) + len(unit.text) if buffer else len(unit.text)
            if buffer and projected > self._size:
                flush()
                projected = len(unit.text)
            buffer.append(unit)
            buffer_len = projected

        flush()
        return chunks

    # ── Internals ─────────────────────────────────────────────────────────────

    def _units(self, pages: list[PageText]) -> list[_Unit]:
        """Flatten pages into units no longer than ``chunk_size``, tracking sections."""
        units: list[_Unit] = []
        section = ""
        for page in pages:
            for paragraph in _split_paragraphs(normalize_text(page.text)):
                for piece in self._split_oversized(paragraph):
                    # Per piece, not per paragraph: a long block can span several
                    # headings, and each piece should carry the one it falls under.
                    section = detect_section(piece) or section
                    units.append(_Unit(text=piece, page=page.page, section=section))
        return units

    def _split_oversized(self, text: str) -> list[str]:
        """Slice a paragraph longer than one chunk, preferring word boundaries."""
        if len(text) <= self._size:
            return [text]

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self._size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            pieces.append(text[start:end].strip())
            start = end
        return [piece for piece in pieces if piece]

    def _tail(self, text: str) -> str:
        """The trailing overlap to seed the next chunk, cut at a word boundary."""
        if self._overlap <= 0:
            return ""
        tail = text[-self._overlap :]
        boundary = tail.find(" ")
        if boundary != -1:
            tail = tail[boundary + 1 :]
        return tail.strip()


def normalize_text(raw: str) -> str:
    """Collapse PDF whitespace noise without touching content."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> str:
    """Return the last section heading that opens a line in ``text``, or ''.

    Scanning every line rather than only the start of the block matters on real
    PDFs: PyMuPDF separates lines with a single newline, so an entire page often
    arrives as one paragraph with its headings buried inside it. Matching only at
    position 0 left every chunk with an empty section.
    """
    last = ""
    for line in text.splitlines():
        match = _SECTION_PATTERN.match(line)
        if match:
            last = re.sub(r"\s+", " ", match.group(1)).strip()
    return last


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.split(_PARAGRAPH_SEPARATOR) if part.strip()]
