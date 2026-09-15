"""Reciprocal rank fusion over the dense and lexical retrieval branches.

Fusion is kept in Python precisely so it can be tested without a server, and so
the dense cosine similarity survives fusion for confidence scoring.
"""

from __future__ import annotations

from domain.models import RegulationDomain
from infrastructure.vector_db.milvus_store import fuse_ranked_lists


def hit(chunk_id: str, distance: float = 0.5, **entity) -> dict:
    payload = {
        "text": f"text of {chunk_id}",
        "doc_id": "doc1",
        "source_url": "https://drive.google.com/file/d/a/view",
        "doc_title": "Quy chế đào tạo 2024",
        "domain": RegulationDomain.COURSE_CURRICULUM.value,
        "section": "Điều 5",
        "page_start": 3,
        "page_end": 3,
        "doc_updated_at": "2026-09-01",
    }
    payload.update(entity)
    return {"id": chunk_id, "distance": distance, "entity": payload}


class TestFusion:
    def test_nothing_in_nothing_out(self):
        assert fuse_ranked_lists([], [], top_k=5) == []
        assert fuse_ranked_lists([[]], [], top_k=5) == []

    def test_single_branch_keeps_its_order(self):
        dense = [[hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]]
        assert [c.chunk_id for c in fuse_ranked_lists(dense, [], top_k=3)] == ["a", "b", "c"]

    def test_top_k_caps_the_result(self):
        dense = [[hit(letter) for letter in "abcdef"]]
        assert len(fuse_ranked_lists(dense, [], top_k=2)) == 2

    def test_agreement_between_branches_wins(self):
        """'b' is second on both branches; 'a' leads only the dense one."""
        dense = [[hit("a", 0.9), hit("b", 0.85)]]
        sparse = [hit("c"), hit("b")]
        ranked = [chunk.chunk_id for chunk in fuse_ranked_lists(dense, sparse, top_k=3)]
        assert ranked[0] == "b"

    def test_dense_similarity_survives_fusion(self):
        dense = [[hit("a", 0.93)]]
        assert fuse_ranked_lists(dense, [], top_k=1)[0].dense_score == 0.93

    def test_best_similarity_across_dense_branches_is_kept(self):
        """The English and Vietnamese query vectors are separate dense branches."""
        dense = [[hit("a", 0.42)], [hit("a", 0.88)]]
        assert fuse_ranked_lists(dense, [], top_k=1)[0].dense_score == 0.88

    def test_negative_cosine_is_clamped(self):
        dense = [[hit("a", -0.3)]]
        assert fuse_ranked_lists(dense, [], top_k=1)[0].dense_score == 0.0

    def test_lexical_only_match_has_no_dense_score(self):
        result = fuse_ranked_lists([], [hit("a", 12.5)], top_k=1)
        assert result[0].chunk_id == "a"
        assert result[0].dense_score == 0.0

    def test_rrf_k_changes_the_weighting(self):
        dense = [[hit("a"), hit("b")]]
        sparse = [hit("b"), hit("a")]
        small = fuse_ranked_lists(dense, sparse, top_k=2, rrf_k=1)
        assert {chunk.chunk_id for chunk in small} == {"a", "b"}


class TestEntityMapping:
    def test_maps_citation_metadata(self):
        chunk = fuse_ranked_lists([[hit("a")]], [], top_k=1)[0]
        assert chunk.doc_title == "Quy chế đào tạo 2024"
        assert chunk.source_url == "https://drive.google.com/file/d/a/view"
        assert chunk.section == "Điều 5"
        assert chunk.page_start == 3
        assert chunk.doc_updated_at == "2026-09-01"
        assert chunk.domain is RegulationDomain.COURSE_CURRICULUM

    def test_unknown_domain_degrades_to_other(self):
        chunk = fuse_ranked_lists([[hit("a", domain="LEGACY")]], [], top_k=1)[0]
        assert chunk.domain is RegulationDomain.OTHER

    def test_missing_entity_yields_safe_defaults(self):
        chunk = fuse_ranked_lists([[{"id": "a", "distance": 0.5}]], [], top_k=1)[0]
        assert chunk.text == ""
        assert chunk.page_start == 1
        assert chunk.domain is RegulationDomain.OTHER

    def test_hits_without_an_id_are_skipped(self):
        dense = [[{"id": "", "distance": 0.9}, hit("b")]]
        assert [c.chunk_id for c in fuse_ranked_lists(dense, [], top_k=5)] == ["b"]

    def test_null_page_numbers_become_page_one(self):
        chunk = fuse_ranked_lists([[hit("a", page_start=None, page_end=None)]], [], top_k=1)[0]
        assert chunk.page_start == 1
        assert chunk.page_end == 1
