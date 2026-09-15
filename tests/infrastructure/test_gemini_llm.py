"""GeminiLLM: model fallthrough, retry policy, and answer parsing.

These decide whether a transient outage or a retired model id costs the student
an answer. A live run hit exactly that: the main model returned 503, and the
configured fallback was a decommissioned id whose 404 aborted the request
instead of moving on.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors as genai_errors

from config.settings import Settings
from domain.exceptions import LLMUnavailableError
from domain.models import RegulationDomain
from infrastructure.llm.gemini_llm import GeminiLLM

ANSWER_JSON = json.dumps(
    {"answer": "GPA tối thiểu là 2.0/4.0.", "grounding": 0.9, "used_sources": [1]}
)


def api_error(status: int) -> genai_errors.ClientError:
    """An SDK error built without depending on its constructor signature."""
    exc = genai_errors.ClientError.__new__(genai_errors.ClientError)
    exc.code = status
    exc.message = f"status {status}"
    return exc


def response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, candidates=[])


def build_llm(**overrides) -> GeminiLLM:
    defaults = dict(
        gemini_api_key="test",
        zilliz_cloud_endpoint="https://example.zillizcloud.com",
        zilliz_cloud_api_key="test",
        main_model="model-main",
        fallback_model="model-fallback",
        llm_retry_count=2,
        llm_retry_delay=0.0,
    )
    return GeminiLLM(Settings(**{**defaults, **overrides}))


def install(llm: GeminiLLM, side_effect) -> AsyncMock:
    """Swap in a fake SDK client so nothing leaves the process."""
    generate = AsyncMock(side_effect=side_effect)
    llm._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    return generate


def models_called(generate: AsyncMock) -> list[str]:
    return [call.kwargs["model"] for call in generate.await_args_list]


class TestModelFallthrough:
    async def test_retired_model_advances_to_the_fallback(self):
        llm = build_llm()
        generate = install(llm, [api_error(404), response(ANSWER_JSON)])
        result = await llm.answer("system", "user")
        assert result.answer
        assert models_called(generate) == ["model-main", "model-fallback"]

    async def test_retired_model_is_not_retried(self):
        """Retrying a 404 is futile — it must cost exactly one attempt."""
        llm = build_llm()
        generate = install(llm, [api_error(404), response(ANSWER_JSON)])
        await llm.answer("system", "user")
        assert generate.await_count == 2

    async def test_retired_everywhere_raises(self):
        llm = build_llm()
        generate = install(llm, [api_error(404), api_error(404)])
        with pytest.raises(LLMUnavailableError):
            await llm.answer("system", "user")
        assert models_called(generate) == ["model-main", "model-fallback"]

    async def test_bad_request_aborts_without_trying_the_fallback(self):
        """A malformed request would fail identically on any model."""
        llm = build_llm()
        generate = install(llm, [api_error(400), response(ANSWER_JSON)])
        with pytest.raises(LLMUnavailableError):
            await llm.answer("system", "user")
        assert generate.await_count == 1

    async def test_rejected_credentials_abort_immediately(self):
        llm = build_llm()
        generate = install(llm, [api_error(403), response(ANSWER_JSON)])
        with pytest.raises(LLMUnavailableError):
            await llm.answer("system", "user")
        assert generate.await_count == 1

    async def test_overload_retries_then_advances(self):
        llm = build_llm()
        generate = install(llm, [api_error(503), api_error(503), response(ANSWER_JSON)])
        result = await llm.answer("system", "user")
        assert result.answer
        assert models_called(generate) == ["model-main", "model-main", "model-fallback"]

    async def test_quota_error_is_retried(self):
        llm = build_llm()
        generate = install(llm, [api_error(429), response(ANSWER_JSON)])
        await llm.answer("system", "user")
        assert models_called(generate) == ["model-main", "model-main"]

    async def test_identical_main_and_fallback_are_tried_once(self):
        llm = build_llm(main_model="same", fallback_model="same")
        generate = install(llm, [api_error(404)])
        with pytest.raises(LLMUnavailableError):
            await llm.answer("system", "user")
        assert generate.await_count == 1


class TestAnswerParsing:
    async def test_parses_structured_output(self):
        llm = build_llm()
        install(llm, [response(ANSWER_JSON)])
        result = await llm.answer("system", "user")
        assert result.answer == "GPA tối thiểu là 2.0/4.0."
        assert result.grounding == 0.9
        assert result.used_sources == [1]

    @pytest.mark.parametrize(("raw", "expected"), [(5, 1.0), (-2, 0.0), ("high", 0.0)])
    async def test_grounding_is_coerced_into_range(self, raw, expected):
        llm = build_llm()
        install(
            llm,
            [response(json.dumps({"answer": "x", "grounding": raw, "used_sources": []}))],
        )
        assert (await llm.answer("s", "u")).grounding == expected

    async def test_ignores_non_integer_source_indices(self):
        llm = build_llm()
        payload = {"answer": "x", "grounding": 1, "used_sources": [1, "two", True, 3]}
        install(llm, [response(json.dumps(payload))])
        assert (await llm.answer("s", "u")).used_sources == [1, 3]

    async def test_unparsable_json_raises(self):
        llm = build_llm()
        install(llm, [response("not json at all")])
        with pytest.raises(LLMUnavailableError, match="unparsable"):
            await llm.answer("s", "u")

    async def test_json_array_instead_of_object_raises(self):
        llm = build_llm()
        install(llm, [response("[1, 2, 3]")])
        with pytest.raises(LLMUnavailableError, match="expected an object"):
            await llm.answer("s", "u")

    async def test_blank_answer_field_raises(self):
        llm = build_llm()
        install(llm, [response(json.dumps({"answer": "  ", "grounding": 1, "used_sources": []}))])
        with pytest.raises(LLMUnavailableError):
            await llm.answer("s", "u")

    async def test_empty_body_raises(self):
        """Usually a safety block or a token cap — worth surfacing, not answering ''."""
        llm = build_llm()
        install(llm, [response("")])
        with pytest.raises(LLMUnavailableError, match="no text"):
            await llm.answer("s", "u")


class TestClassifyAndTranslate:
    async def test_maps_a_known_domain_label(self):
        llm = build_llm()
        install(llm, [response("SCHOLARSHIP")])
        assert await llm.classify_domain("Quy định học bổng", "...") is (
            RegulationDomain.SCHOLARSHIP
        )

    async def test_label_inside_a_sentence_still_maps(self):
        llm = build_llm()
        install(llm, [response("This document is GRADUATION related.")])
        assert await llm.classify_domain("x", "...") is RegulationDomain.GRADUATION

    async def test_unrecognized_label_becomes_other(self):
        llm = build_llm()
        install(llm, [response("SOMETHING_ELSE")])
        assert await llm.classify_domain("x", "...") is RegulationDomain.OTHER

    async def test_translation_is_stripped(self):
        llm = build_llm()
        install(llm, [response("  học bổng  ")])
        assert await llm.translate_to_vietnamese("scholarship") == "học bổng"
