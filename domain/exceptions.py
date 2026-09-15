"""Domain exception hierarchy.

Rule: infrastructure adapters and the application layer catch third-party
exceptions and re-raise one of these with ``raise X from exc``, so the original
traceback stays available in ``__cause__``. The presentation layer maps these to
user-facing messages.
"""

from __future__ import annotations


class UniRegulationsError(Exception):
    """Base class for all application-level errors. Never raise directly."""


class EmbeddingError(UniRegulationsError):
    """Embedding service failed (quota, network, malformed response)."""


class LLMUnavailableError(UniRegulationsError):
    """LLM is rate-limited, unreachable, or returned an unusable response."""


class RetrievalError(UniRegulationsError):
    """Vector similarity search failed."""


class VectorStoreError(UniRegulationsError):
    """Collection creation, upsert, or delete failed."""


class PDFDownloadError(UniRegulationsError):
    """PDF could not be fetched (HTTP error, timeout, or not a PDF)."""


class PDFExtractionError(UniRegulationsError):
    """Text could not be extracted from a PDF."""


class ScraperError(UniRegulationsError):
    """The regulation listing page could not be loaded or parsed."""


class RegistryError(UniRegulationsError):
    """The document registry (SQLite) could not be read or written."""


class ConfigurationError(UniRegulationsError):
    """Required configuration is missing or invalid."""
