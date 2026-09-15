"""SectionChunker: real overlap, page ranges, section tracking."""

from __future__ import annotations

import pytest

from domain.models import PageText
from infrastructure.chunking.section_chunker import (
    SectionChunker,
    detect_section,
    normalize_text,
)


class TestSectionDetection:
    @pytest.mark.parametrize(
        ("paragraph", "expected"),
        [
            ("Điều 5. Điểm trung bình tích lũy", "Điều 5"),
            ("Chương II. Đào tạo", "Chương II"),
            ("Phụ lục 1: Bảng quy đổi điểm", "Phụ lục 1"),
            ("PHỤ  LỤC 2 — Biểu mẫu", "PHỤ LỤC 2"),
            ("Mục 3. Học bổng", "Mục 3"),
            ("Điều 12a. Trường hợp đặc biệt", "Điều 12a"),
        ],
    )
    def test_detects_headings(self, paragraph, expected):
        assert detect_section(paragraph) == expected

    @pytest.mark.parametrize(
        "paragraph",
        [
            "Sinh viên phải đăng ký tối thiểu 14 tín chỉ.",
            "",
            "Bảng 1. Danh sách môn học",
        ],
    )
    def test_ignores_non_headings(self, paragraph):
        assert detect_section(paragraph) == ""

    def test_finds_a_heading_inside_a_line_separated_block(self):
        """PyMuPDF hands back a whole page as one block with single newlines."""
        block = (
            "TRƯỜNG ĐẠI HỌC BÁCH KHOA\n"
            "Điều 23. Học phí\n"
            "23.1. Sinh viên đăng ký tối đa 24 tín chỉ mỗi học kỳ."
        )
        assert detect_section(block) == "Điều 23"

    def test_last_heading_in_a_block_wins(self):
        assert detect_section("Điều 5. Đánh giá\nnội dung\nĐiều 6. Cảnh báo") == "Điều 6"

    def test_heading_must_open_its_own_line(self):
        assert detect_section("chi tiết xem Điều 9 nêu trên") == ""


class TestNormalization:
    def test_collapses_whitespace_noise(self):
        assert normalize_text("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
        assert normalize_text("a  \t  b") == "a b"
        assert normalize_text("  padded  ") == "padded"


class TestChunking:
    def test_no_text_yields_no_chunks(self):
        chunker = SectionChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk([]) == []
        assert chunker.chunk([PageText(page=1, text="   \n\n  ")]) == []

    def test_short_document_is_one_chunk(self):
        chunker = SectionChunker(chunk_size=1200, chunk_overlap=200)
        pages = [PageText(page=1, text="Điều 1. Phạm vi\n\nQuy chế này áp dụng cho sinh viên.")]
        chunks = chunker.chunk(pages)
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1
        assert chunks[0].section == "Điều 1"

    def test_overlap_is_carried_into_the_next_chunk(self):
        """The legacy chunker never did this for paragraph-packed text."""
        chunker = SectionChunker(chunk_size=60, chunk_overlap=20)
        pages = [PageText(page=1, text=f"{'A' * 50}\n\n{'B' * 50}\n\n{'C' * 50}")]
        chunks = chunker.chunk(pages)
        assert len(chunks) == 3
        assert chunks[0].text == "A" * 50
        assert chunks[1].text.startswith("A" * 20)
        assert "B" * 50 in chunks[1].text
        assert chunks[2].text.startswith("B" * 20)

    def test_first_chunk_has_no_carried_prefix(self):
        chunker = SectionChunker(chunk_size=60, chunk_overlap=20)
        pages = [PageText(page=1, text=f"{'A' * 50}\n\n{'B' * 50}")]
        assert chunker.chunk(pages)[0].text == "A" * 50

    def test_zero_overlap_shares_nothing(self):
        chunker = SectionChunker(chunk_size=60, chunk_overlap=0)
        pages = [PageText(page=1, text=f"{'A' * 50}\n\n{'B' * 50}")]
        chunks = chunker.chunk(pages)
        assert [c.text for c in chunks] == ["A" * 50, "B" * 50]

    def test_page_range_spans_the_pages_a_chunk_covers(self):
        chunker = SectionChunker(chunk_size=1200, chunk_overlap=0)
        pages = [
            PageText(page=1, text="Điều 1. Mở đầu"),
            PageText(page=2, text="Nội dung tiếp theo"),
            PageText(page=3, text="Kết thúc"),
        ]
        chunks = chunker.chunk(pages)
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3

    def test_section_carries_forward_until_the_next_heading(self):
        chunker = SectionChunker(chunk_size=40, chunk_overlap=0)
        pages = [
            PageText(page=1, text="Điều 1. Phạm vi\n\nÁp dụng cho toàn bộ sinh viên chính quy."),
            PageText(page=2, text="Điều 2. Tín chỉ\n\nTối đa 24 tín chỉ mỗi học kỳ."),
        ]
        sections = [chunk.section for chunk in chunker.chunk(pages)]
        assert sections[0] == "Điều 1"
        assert sections[-1] == "Điều 2"
        assert set(sections) == {"Điều 1", "Điều 2"}

    def test_oversized_paragraph_is_split(self):
        chunker = SectionChunker(chunk_size=50, chunk_overlap=10)
        pages = [PageText(page=1, text="X" * 260)]
        chunks = chunker.chunk(pages)
        assert len(chunks) > 1
        # Overlap is additive, so a chunk may exceed chunk_size by up to overlap.
        assert all(len(chunk.text) <= 50 + 10 + 2 for chunk in chunks)

    def test_oversized_paragraph_prefers_word_boundaries(self):
        chunker = SectionChunker(chunk_size=50, chunk_overlap=0)
        words = " ".join(["tinchi"] * 40)
        chunks = chunker.chunk([PageText(page=1, text=words)])
        for chunk in chunks:
            assert all(word == "tinchi" for word in chunk.text.split())

    def test_sections_populate_from_a_line_separated_page(self):
        """The shape PyMuPDF actually produces: one block, single newlines.

        This is the case that used to leave every chunk's section empty.
        """
        page_text = "\n".join(
            [
                "Điều 22. Đăng ký môn học",
                "Sinh viên đăng ký theo kế hoạch của khoa. " * 4,
                "Điều 23. Học phí",
                "Học phí được tính theo số tín chỉ đã đăng ký. " * 4,
            ]
        )
        chunker = SectionChunker(chunk_size=120, chunk_overlap=20)
        sections = [chunk.section for chunk in chunker.chunk([PageText(page=1, text=page_text)])]
        assert sections
        assert all(sections), "every chunk should carry a section"
        assert "Điều 22" in sections
        assert "Điều 23" in sections

    def test_every_chunk_keeps_a_valid_page_number(self):
        chunker = SectionChunker(chunk_size=40, chunk_overlap=10)
        pages = [PageText(page=index, text=f"Trang {index}. " + "nội dung " * 20) for index in range(1, 5)]
        chunks = chunker.chunk(pages)
        assert chunks
        for chunk in chunks:
            assert 1 <= chunk.page_start <= chunk.page_end <= 4


class TestConstructorValidation:
    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(100, 100), (100, 200), (0, 0), (100, -1)],
    )
    def test_rejects_impossible_configuration(self, size, overlap):
        with pytest.raises(ValueError):
            SectionChunker(chunk_size=size, chunk_overlap=overlap)
