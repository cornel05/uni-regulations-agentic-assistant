"""Extraction quality scoring and model-output page splitting.

The scoring tests pin down the behaviour the legacy formula got wrong: a short
but perfectly extracted document must not be sent for a paid re-read, while a
scanned or garbled one must be.
"""

from __future__ import annotations

from domain.models import PageText
from infrastructure.pdf.extractor import (
    _split_marked_pages,
    score_extraction_quality,
)

THRESHOLD = 0.75  # the default PYMUPDF_QUALITY_THRESHOLD


def page(number: int, text: str) -> PageText:
    return PageText(page=number, text=text)


class TestQualityScoring:
    def test_no_pages_scores_zero(self):
        assert score_extraction_quality([])["overall"] == 0.0

    def test_blank_pages_score_zero(self):
        assert score_extraction_quality([page(1, "   \n "), page(2, "")])["overall"] == 0.0

    def test_dense_text_scores_full(self):
        pages = [page(n, "Điều 5. Sinh viên phải đạt GPA 2.0. " * 50) for n in range(1, 4)]
        assert score_extraction_quality(pages)["overall"] >= THRESHOLD

    def test_short_but_clean_document_is_accepted(self):
        """A valid 300-character regulation used to score ~0.49 and pay for a re-read."""
        text = "Điều 1. Quy định này áp dụng cho sinh viên chính quy của trường. " * 5
        score = score_extraction_quality([page(1, text[:300])])
        assert score["overall"] >= THRESHOLD

    def test_scanned_document_is_rejected(self):
        """Many pages, almost no extractable text — the real fallback trigger."""
        pages = [page(n, "  \n") for n in range(1, 40)]
        pages[0] = page(1, "QUY CHE")
        assert score_extraction_quality(pages)["overall"] < THRESHOLD

    def test_garbled_text_is_rejected(self):
        pages = [page(1, "".join(chr(index % 30 + 1) for index in range(600)))]
        score = score_extraction_quality(pages)
        assert score["legibility"] < 0.2
        assert score["overall"] < THRESHOLD

    def test_reports_its_inputs(self):
        pages = [page(1, "abc"), page(2, "defg")]
        score = score_extraction_quality(pages)
        assert score["page_count"] == 2.0
        assert score["char_count"] == 7.0

    def test_expected_density_is_configurable(self):
        pages = [page(1, "x" * 200)]
        lenient = score_extraction_quality(pages, expected_chars_per_page=100)
        strict = score_extraction_quality(pages, expected_chars_per_page=4000)
        assert lenient["overall"] > strict["overall"]

    def test_all_components_stay_in_range(self):
        pages = [page(1, "Điều 1. " * 300)]
        score = score_extraction_quality(pages)
        for key in ("density", "legibility", "overall"):
            assert 0.0 <= score[key] <= 1.0


class TestPageMarkerSplitting:
    def test_splits_on_markers(self):
        pages = _split_marked_pages(
            "[[page 1]]\nĐiều 1. Phạm vi\n[[page 2]]\nĐiều 2. Tín chỉ"
        )
        assert [p.page for p in pages] == [1, 2]
        assert pages[0].text == "Điều 1. Phạm vi"
        assert pages[1].text == "Điều 2. Tín chỉ"

    def test_honours_the_reported_page_numbers(self):
        pages = _split_marked_pages("[[page 7]]\nNội dung\n[[page 8]]\nTiếp theo")
        assert [p.page for p in pages] == [7, 8]

    def test_unmarked_output_becomes_one_page(self):
        pages = _split_marked_pages("Điều 1. Không có dấu trang")
        assert len(pages) == 1
        assert pages[0].page == 1
        assert pages[0].text == "Điều 1. Không có dấu trang"

    def test_empty_marked_pages_are_dropped(self):
        pages = _split_marked_pages("[[page 1]]\nNội dung\n[[page 2]]\n   \n[[page 3]]\nCuối")
        assert [p.page for p in pages] == [1, 3]

    def test_tolerates_marker_spacing_and_case(self):
        pages = _split_marked_pages("[[ Page  4 ]]\nNội dung")
        assert pages[0].page == 4

    def test_all_markers_empty_falls_back_to_whole_text(self):
        raw = "[[page 1]]\n   "
        pages = _split_marked_pages(raw)
        assert len(pages) == 1
        assert pages[0].page == 1
