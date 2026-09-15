"""In-process session store for multi-turn context (US-04).

Keeps the last N exchanges per session with a sliding TTL. Nothing is persisted:
conversation logs are the one place a student's own words would be retained, and
the PRD requires a clear-history path, so forgetting on restart is the feature.

ponytail: single-process only — a second worker would not see these sessions.
Swap in Redis behind the same port if the app is ever scaled out.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time

from domain.models import Turn

logger = logging.getLogger(__name__)


class MemorySessionStore:
    """Concrete SessionStorePort backed by a dict with TTL eviction."""

    def __init__(self, max_turns: int = 5, ttl_seconds: int = 3600) -> None:
        self._max_messages = max(1, max_turns) * 2  # one user + one assistant per turn
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[float, list[Turn]]] = {}

    def create(self) -> str:
        session_id = secrets.token_urlsafe(16)
        with self._lock:
            self._sessions[session_id] = (time.monotonic(), [])
        return session_id

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            self._evict_expired()
            entry = self._sessions.get(session_id)
            return list(entry[1]) if entry else []

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            self._evict_expired()
            _, turns = self._sessions.get(session_id, (time.monotonic(), []))
            turns = turns + [
                Turn(role="user", content=user_message),
                Turn(role="assistant", content=assistant_message),
            ]
            self._sessions[session_id] = (
                time.monotonic(),
                turns[-self._max_messages :],
            )

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        """Caller holds the lock."""
        cutoff = time.monotonic() - self._ttl
        expired = [key for key, (touched, _) in self._sessions.items() if touched < cutoff]
        for key in expired:
            del self._sessions[key]
        if expired:
            logger.debug("Evicted %d expired session(s)", len(expired))
