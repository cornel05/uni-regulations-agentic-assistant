"""Zilliz Cloud / Milvus adapter — retrieval and indexing in one place.

Hybrid retrieval runs the dense and lexical branches as separate searches and
fuses them here with reciprocal rank fusion. The server-side ranker would do the
fusion in one round trip but returns only the fused score; confidence scoring
needs the raw cosine similarity of the best dense hit, so the branches are kept
separate and merged in Python. Fusion is a pure function, testable without a
server.

Schema differences from the legacy collection, all deliberate:
  * ``doc_title`` is actually written (legacy indexed it empty, which is what the
    separate title-repair script existed to patch up).
  * COSINE instead of L2 — Gemini embeddings are not L2-normalized at truncated
    dimensions, and cosine is what the similarity score is interpreted as.
  * ``domain``, ``section``, ``page_start``/``page_end`` and ``doc_updated_at``
    support domain filtering and page-level citation.
  * a BM25 function field for the lexical branch.

pymilvus is synchronous, so every call runs off the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import re

from pymilvus import DataType, Function, FunctionType, MilvusClient

from config.settings import Settings
from domain.exceptions import RetrievalError, VectorStoreError
from domain.models import Chunk, RegulationDomain, RetrievedChunk

logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = [
    "text",
    "doc_id",
    "source_url",
    "doc_title",
    "domain",
    "section",
    "page_start",
    "page_end",
    "doc_updated_at",
]

_DENSE_FIELD = "dense"
_SPARSE_FIELD = "sparse"
_BM25_FUNCTION = "text_bm25"

_TEXT_MAX_LENGTH = 65535
_SAFE_ID = re.compile(r"^[A-Za-z0-9_:.-]+$")

# Milvus builds differ in which analyzers they ship. ICU segments Vietnamese
# better; the standard analyzer is the universally available fallback.
_ANALYZER_CANDIDATES = ({"tokenizer": "icu"}, {"type": "standard"})


class MilvusStore:
    """Concrete RetrieverPort and VectorStorePort."""

    def __init__(self, settings: Settings) -> None:
        self._uri = settings.zilliz_cloud_endpoint
        self._token = settings.zilliz_cloud_api_key
        self._connection: MilvusClient | None = None
        self._collection = settings.milvus_collection_name
        self._dimension = settings.embedding_dimension
        self._sparse_enabled = settings.enable_sparse
        self._batch_size = settings.upsert_batch_size
        self._rrf_k = settings.rrf_k
        self._ready = False

    @property
    def _client(self) -> MilvusClient:
        """Connect on first use, not at construction.

        Connecting in __init__ would make every entry point fail when the index
        is unreachable — including `main.py status`, which only reads SQLite, and
        the API server, which is meant to start and report the problem per
        request rather than refuse to boot.
        """
        if self._connection is None:
            try:
                self._connection = MilvusClient(uri=self._uri, token=self._token)
            except Exception as exc:
                raise VectorStoreError(
                    f"Cannot reach the vector database at {self._uri}: {exc}. "
                    "A serverless cluster that has been idle may be suspended — "
                    "check its status in the Zilliz console and resume it."
                ) from exc
        return self._connection

    # ── Write side ────────────────────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        if self._ready:
            return
        try:
            await asyncio.to_thread(self._ensure_collection_blocking)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not prepare {self._collection!r}: {exc}") from exc
        self._ready = True

    def _ensure_collection_blocking(self) -> None:
        if self._client.has_collection(self._collection):
            self._client.load_collection(self._collection)
            logger.info("Loaded existing collection %r", self._collection)
            return

        analyzers = _ANALYZER_CANDIDATES if self._sparse_enabled else (None,)
        last_exc: Exception | None = None

        for analyzer_params in analyzers:
            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    schema=self._build_schema(analyzer_params),
                    index_params=self._build_index_params(),
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("Collection creation failed with analyzer %s: %s", analyzer_params, exc)
                continue
            # load_collection is what the legacy cloud adapter never called, so
            # searching a freshly created collection always failed.
            self._client.load_collection(self._collection)
            logger.info(
                "Created collection %r (analyzer=%s, sparse=%s)",
                self._collection,
                analyzer_params,
                self._sparse_enabled,
            )
            return

        if self._sparse_enabled:
            raise VectorStoreError(
                f"Could not create {self._collection!r} with a text analyzer: {last_exc}. "
                "Set ENABLE_SPARSE=false to index dense-only on this deployment."
            ) from last_exc
        raise VectorStoreError(f"Could not create {self._collection!r}: {last_exc}") from last_exc

    def _build_schema(self, analyzer_params: dict | None):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(_DENSE_FIELD, DataType.FLOAT_VECTOR, dim=self._dimension)

        if self._sparse_enabled:
            schema.add_field(_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(
                "text",
                DataType.VARCHAR,
                max_length=_TEXT_MAX_LENGTH,
                enable_analyzer=True,
                analyzer_params=analyzer_params,
            )
        else:
            schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX_LENGTH)

        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("source_url", DataType.VARCHAR, max_length=1024)
        schema.add_field("doc_title", DataType.VARCHAR, max_length=1024)
        schema.add_field("domain", DataType.VARCHAR, max_length=32)
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_updated_at", DataType.VARCHAR, max_length=32)
        schema.add_field("page_start", DataType.INT32)
        schema.add_field("page_end", DataType.INT32)
        schema.add_field("chunk_index", DataType.INT32)

        if self._sparse_enabled:
            schema.add_function(
                Function(
                    name=_BM25_FUNCTION,
                    function_type=FunctionType.BM25,
                    input_field_names="text",
                    output_field_names=_SPARSE_FIELD,
                )
            )
        return schema

    def _build_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=_DENSE_FIELD, index_type="AUTOINDEX", metric_type="COSINE"
        )
        if self._sparse_enabled:
            index_params.add_index(
                field_name=_SPARSE_FIELD, index_type="AUTOINDEX", metric_type="BM25"
            )
        return index_params

    async def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        missing = [chunk.chunk_id for chunk in chunks if not chunk.embedding]
        if missing:
            raise VectorStoreError(f"{len(missing)} chunk(s) have no embedding: {missing[:3]}")

        written = 0
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            rows = [_to_row(chunk) for chunk in batch]
            try:
                result = await asyncio.to_thread(
                    self._client.upsert, collection_name=self._collection, data=rows
                )
            except Exception as exc:
                raise VectorStoreError(f"Upsert failed: {exc}") from exc
            written += int(result.get("upsert_count", len(rows)) or len(rows))
        logger.debug("Upserted %d chunks into %r", written, self._collection)
        return written

    async def delete_by_doc_id(self, doc_id: str) -> None:
        if not _SAFE_ID.match(doc_id):
            raise VectorStoreError(f"Refusing to delete with unsafe doc_id {doc_id!r}")
        try:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection,
                filter=f'doc_id == "{doc_id}"',
            )
        except Exception as exc:
            raise VectorStoreError(f"Delete failed for doc_id {doc_id}: {exc}") from exc

    async def count(self) -> int:
        # count(*) is exact and immediate. get_collection_stats reflects only
        # sealed segments, so it reports 0 for rows that were just upserted and
        # not yet flushed — which made the dashboard read "0 chunks" directly
        # after a successful ingest.
        try:
            rows = await asyncio.to_thread(
                self._client.query,
                collection_name=self._collection,
                filter="",
                output_fields=["count(*)"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not count indexed chunks: {exc}") from exc
        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0) or 0)

    # ── Read side ─────────────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query_text: str,
        query_embeddings: list[list[float]],
        top_k: int,
        domain: RegulationDomain | None = None,
    ) -> list[RetrievedChunk]:
        if not query_embeddings:
            return []

        fetch = max(top_k * 2, top_k)
        filter_expr = f'domain == "{domain.value}"' if domain else ""

        try:
            dense_lists = [
                await asyncio.to_thread(self._search_dense, embedding, fetch, filter_expr)
                for embedding in query_embeddings
            ]
        except Exception as exc:
            raise RetrievalError(f"Dense search failed: {exc}") from exc

        sparse_list: list[dict] = []
        if self._sparse_enabled and query_text.strip():
            try:
                sparse_list = await asyncio.to_thread(
                    self._search_sparse, query_text, fetch, filter_expr
                )
            except Exception as exc:
                # Dense results alone are still a usable answer.
                logger.warning("Lexical branch unavailable, continuing dense-only: %s", exc)

        return fuse_ranked_lists(dense_lists, sparse_list, top_k=top_k, rrf_k=self._rrf_k)

    def _search_dense(self, embedding: list[float], limit: int, filter_expr: str) -> list[dict]:
        results = self._client.search(
            collection_name=self._collection,
            data=[embedding],
            anns_field=_DENSE_FIELD,
            limit=limit,
            filter=filter_expr,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        return list(results[0]) if results else []

    def _search_sparse(self, query_text: str, limit: int, filter_expr: str) -> list[dict]:
        results = self._client.search(
            collection_name=self._collection,
            data=[query_text],
            anns_field=_SPARSE_FIELD,
            limit=limit,
            filter=filter_expr,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "BM25"},
        )
        return list(results[0]) if results else []


def fuse_ranked_lists(
    dense_lists: list[list[dict]],
    sparse_list: list[dict],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion across every branch.

    A chunk's rank contribution is ``1 / (rrf_k + rank)`` per list it appears in,
    which needs no score normalization between cosine and BM25 scales. The dense
    cosine similarity is carried through separately for confidence scoring.
    """
    fused: dict[str, float] = {}
    entities: dict[str, dict] = {}
    dense_best: dict[str, float] = {}

    for hits in dense_lists:
        for rank, hit in enumerate(hits, 1):
            chunk_id = str(hit.get("id", ""))
            if not chunk_id:
                continue
            entities.setdefault(chunk_id, hit.get("entity") or {})
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            similarity = _clamp(float(hit.get("distance", 0.0)))
            dense_best[chunk_id] = max(dense_best.get(chunk_id, 0.0), similarity)

    for rank, hit in enumerate(sparse_list, 1):
        chunk_id = str(hit.get("id", ""))
        if not chunk_id:
            continue
        entities.setdefault(chunk_id, hit.get("entity") or {})
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: max(0, top_k)]
    return [
        _to_chunk(chunk_id, entities.get(chunk_id, {}), score, dense_best.get(chunk_id, 0.0))
        for chunk_id, score in ranked
    ]


def _to_row(chunk: Chunk) -> dict:
    # The sparse field is intentionally absent: the BM25 function derives it
    # server-side from `text`, and supplying it would be rejected.
    return {
        "id": chunk.chunk_id,
        _DENSE_FIELD: chunk.embedding,
        "text": chunk.text[:_TEXT_MAX_LENGTH],
        "doc_id": chunk.doc_id,
        "source_url": chunk.source_url,
        "doc_title": chunk.doc_title,
        "domain": chunk.domain.value,
        "section": chunk.section[:256],
        "doc_updated_at": chunk.doc_updated_at,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
    }


def _to_chunk(chunk_id: str, entity: dict, score: float, dense_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=entity.get("text") or "",
        source_url=entity.get("source_url") or "",
        doc_title=entity.get("doc_title") or "",
        domain=_domain_or_other(entity.get("domain")),
        section=entity.get("section") or "",
        page_start=max(1, int(entity.get("page_start") or 1)),
        page_end=max(1, int(entity.get("page_end") or 1)),
        doc_updated_at=entity.get("doc_updated_at") or "",
        score=score,
        dense_score=dense_score,
    )


def _domain_or_other(value) -> RegulationDomain:
    try:
        return RegulationDomain(value)
    except ValueError:
        return RegulationDomain.OTHER


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
