"""Gemini embeddings adapter.

Two fixes over the legacy implementation:

* Query and document embeddings use different task types. Legacy shared one env
  var, and shipped with it set to ``retrieval_document``, so every query was
  embedded in the wrong vector space.
* Documents are embedded in batches instead of one HTTP call per chunk.
"""

from __future__ import annotations

import asyncio
import logging
import math

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config.settings import Settings
from domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_QUERY_TASK = "RETRIEVAL_QUERY"
_DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"

# The embeddings endpoint caps batch size; 32 stays well inside it and inside
# the per-request token budget for 1200-character chunks.
_MAX_BATCH = 32

_NATIVE_DIMENSION = 3072
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class GeminiEmbedder:
    """Concrete EmbedderPort backed by the Gemini embeddings API."""

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension
        self._retries = settings.embedding_retry_count
        self._delay = settings.embedding_retry_delay

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], _QUERY_TASK)
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = texts[start : start + _MAX_BATCH]
            vectors.extend(await self._embed(batch, _DOCUMENT_TASK))
        return vectors

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self._dimension
        )
        last_exc: Exception | None = None

        for attempt in range(1, self._retries + 1):
            try:
                response = await self._client.aio.models.embed_content(
                    model=self._model, contents=list(texts), config=config
                )
            except (genai_errors.ClientError, genai_errors.ServerError) as exc:
                last_exc = exc
                if getattr(exc, "code", None) not in _RETRYABLE_STATUS:
                    raise EmbeddingError(f"Embedding call rejected: {exc}") from exc
                if attempt == self._retries:
                    break
                wait = self._delay * attempt
                logger.warning(
                    "Embedding throttled (%s), retry %d/%d in %.1fs",
                    getattr(exc, "code", "?"),
                    attempt,
                    self._retries,
                    wait,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                raise EmbeddingError(f"Embedding call failed: {exc}") from exc
            else:
                return self._vectors_from(response, expected=len(texts))

        raise EmbeddingError(
            f"Embedding failed after {self._retries} attempts: {last_exc}"
        ) from last_exc

    def _vectors_from(
        self, response: types.EmbedContentResponse, *, expected: int
    ) -> list[list[float]]:
        embeddings = response.embeddings or []
        vectors = [list(embedding.values or []) for embedding in embeddings]

        if len(vectors) != expected:
            raise EmbeddingError(
                f"Embedder returned {len(vectors)} vectors for {expected} inputs"
            )
        if any(not vector for vector in vectors):
            raise EmbeddingError("Embedder returned an empty vector")

        # Only the native dimension comes back unit-normalized. Truncated output
        # must be re-normalized before it is compared with cosine similarity.
        if self._dimension != _NATIVE_DIMENSION:
            vectors = [_normalize(vector) for vector in vectors]
        return vectors


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
