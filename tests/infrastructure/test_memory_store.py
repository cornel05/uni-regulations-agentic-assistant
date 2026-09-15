"""MemorySessionStore: turn retention, trimming, TTL, clearing."""

from __future__ import annotations

from infrastructure.sessions.memory_store import MemorySessionStore


class TestSessions:
    def test_create_returns_distinct_ids(self):
        store = MemorySessionStore()
        assert store.create() != store.create()

    def test_unknown_session_has_empty_history(self):
        assert MemorySessionStore().history("nope") == []

    def test_new_session_has_empty_history(self):
        store = MemorySessionStore()
        assert store.history(store.create()) == []

    def test_append_records_both_roles_in_order(self):
        store = MemorySessionStore()
        session_id = store.create()
        store.append(session_id, "GPA tối thiểu?", "2.0/4.0.")
        history = store.history(session_id)
        assert [turn.role for turn in history] == ["user", "assistant"]
        assert history[0].content == "GPA tối thiểu?"
        assert history[1].content == "2.0/4.0."

    def test_append_to_unknown_id_starts_that_session(self):
        """A client-supplied id must not be silently dropped."""
        store = MemorySessionStore()
        store.append("client-chosen", "hỏi", "đáp")
        assert len(store.history("client-chosen")) == 2

    def test_history_is_trimmed_to_the_turn_limit(self):
        store = MemorySessionStore(max_turns=2)
        session_id = store.create()
        for index in range(5):
            store.append(session_id, f"q{index}", f"a{index}")
        history = store.history(session_id)
        assert len(history) == 4  # 2 turns × (user + assistant)
        assert history[0].content == "q3"
        assert history[-1].content == "a4"

    def test_history_returns_a_copy(self):
        store = MemorySessionStore()
        session_id = store.create()
        store.append(session_id, "q", "a")
        store.history(session_id).clear()
        assert len(store.history(session_id)) == 2

    def test_clear_forgets_the_conversation(self):
        store = MemorySessionStore()
        session_id = store.create()
        store.append(session_id, "q", "a")
        store.clear(session_id)
        assert store.history(session_id) == []

    def test_clear_is_idempotent(self):
        store = MemorySessionStore()
        store.clear("never-existed")
        store.clear("never-existed")

    def test_expired_session_is_forgotten(self):
        store = MemorySessionStore(ttl_seconds=0)
        session_id = store.create()
        store.append(session_id, "q", "a")
        assert store.history(session_id) == []

    def test_activity_refreshes_the_ttl(self):
        store = MemorySessionStore(ttl_seconds=3600)
        session_id = store.create()
        store.append(session_id, "q1", "a1")
        store.append(session_id, "q2", "a2")
        assert len(store.history(session_id)) == 4

    def test_sessions_are_isolated(self):
        store = MemorySessionStore()
        first, second = store.create(), store.create()
        store.append(first, "q", "a")
        assert store.history(second) == []
