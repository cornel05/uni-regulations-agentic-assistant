"""Gemini generation adapter.

Owns three model interactions: the grounded answer (returned as structured JSON,
so confidence needs no second round-trip), document domain classification, and
query translation for cross-lingual retrieval.

Retry and model fallback live here, not in the application layer, and use
``asyncio.sleep`` — the legacy engine's ``while True: time.sleep(20)`` blocked
the whole event loop on a single quota event.
"""

from __future__ import annotations

import asyncio
import json
import logging

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config.settings import Settings
from domain.exceptions import LLMUnavailableError
from domain.models import GroundedAnswer, RegulationDomain

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# "This model is no longer available" — retire the candidate, not the request.
_MODEL_GONE = 404

# This adapter never passes tools, but the SDK logs an automatic-function-calling
# notice on every generate_content call regardless. Disabling it keeps the logs
# free of a warning that suggests a feature we do not use.
_NO_AFC = types.AutomaticFunctionCallingConfig(disable=True)

_ANSWER_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "grounding": {"type": "NUMBER"},
        "used_sources": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["answer", "grounding", "used_sources"],
}

_CLASSIFY_PROMPT = """Classify this university regulation document into exactly one domain.

Reply with one label and nothing else:
COURSE_CURRICULUM - credit loads, prerequisites, course add/drop, curriculum structure
GRADUATION - graduation requirements, minimum credits, GPA thresholds, thesis rules
SCHOLARSHIP - scholarships, financial aid, eligibility, renewal, appeals
DISCIPLINARY - code of conduct, academic integrity, penalties, disciplinary appeals
ADMISSION_ENROLLMENT - admission, registration windows, status changes, leave of absence
OTHER - anything that fits none of the above

Title: {title}

Excerpt:
{excerpt}"""

_TRANSLATE_PROMPT = """Translate this student question into Vietnamese for searching
Vietnamese university regulation documents. Use the formal administrative vocabulary
those documents use. Reply with the translation only.

Question: {text}"""

_CLASSIFY_MAX_TOKENS = 2048
_TRANSLATE_MAX_TOKENS = 2048


class GeminiLLM:
    """Concrete LLMPort backed by Google Gemini."""

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._models = list(dict.fromkeys([settings.main_model, settings.fallback_model]))
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_output_tokens
        self._retries = settings.llm_retry_count
        self._delay = settings.llm_retry_delay

    async def answer(self, system_prompt: str, user_message: str) -> GroundedAnswer:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            response_mime_type="application/json",
            response_schema=_ANSWER_SCHEMA,
            automatic_function_calling=_NO_AFC,
        )
        text = await self._generate(user_message, config)
        return _parse_answer(text)

    async def classify_domain(self, title: str, sample_text: str) -> RegulationDomain:
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=_CLASSIFY_MAX_TOKENS,
            automatic_function_calling=_NO_AFC,
        )
        prompt = _CLASSIFY_PROMPT.format(title=title, excerpt=sample_text)
        raw = (await self._generate(prompt, config)).strip().upper()

        for domain in RegulationDomain:
            if domain.value in raw:
                return domain
        logger.warning("Unrecognized domain label %r for %r", raw[:60], title)
        return RegulationDomain.OTHER

    async def translate_to_vietnamese(self, text: str) -> str:
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=_TRANSLATE_MAX_TOKENS,
            automatic_function_calling=_NO_AFC,
        )
        return (await self._generate(_TRANSLATE_PROMPT.format(text=text), config)).strip()

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _generate(self, contents: str, config: types.GenerateContentConfig) -> str:
        """Generate text, retrying on throttling and falling back to the backup model."""
        last_exc: Exception | None = None

        for model in self._models:
            for attempt in range(1, self._retries + 1):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                except (genai_errors.ClientError, genai_errors.ServerError) as exc:
                    last_exc = exc
                    status = getattr(exc, "code", None)
                    if status == _MODEL_GONE:
                        # A retired or misspelled model id. Retrying it is futile,
                        # but the next candidate may well work — which is the
                        # whole point of configuring a fallback.
                        logger.warning(
                            "Model %s is unavailable (404) — trying the next candidate", model
                        )
                        break
                    if status not in _RETRYABLE_STATUS:
                        # 400/401/403: our request or our credentials. Another
                        # model would fail identically, so fail fast.
                        raise LLMUnavailableError(f"LLM rejected the request: {exc}") from exc
                    if attempt == self._retries:
                        logger.warning("Model %s exhausted retries (%s)", model, status)
                        break
                    wait = self._delay * attempt
                    logger.warning(
                        "Model %s throttled (%s), retry %d/%d in %.1fs",
                        model,
                        status,
                        attempt,
                        self._retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                except Exception as exc:
                    raise LLMUnavailableError(f"LLM call failed: {exc}") from exc
                else:
                    text = (response.text or "").strip()
                    if not text:
                        # An empty body is usually a safety block or a token cap;
                        # both are worth reporting rather than silently answering "".
                        reason = _finish_reason(response)
                        raise LLMUnavailableError(f"LLM returned no text (finish: {reason})")
                    return text

        raise LLMUnavailableError(f"LLM unavailable: {last_exc}") from last_exc


def _finish_reason(response: types.GenerateContentResponse) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "no candidates"
    return str(getattr(candidates[0], "finish_reason", "unknown"))


def _parse_answer(text: str) -> GroundedAnswer:
    """Parse the structured answer, clamping values the model may overshoot."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(f"LLM returned unparsable JSON: {text[:200]!r}") from exc

    if not isinstance(payload, dict):
        raise LLMUnavailableError(f"LLM returned {type(payload).__name__}, expected an object")

    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise LLMUnavailableError("LLM returned an empty answer field")

    try:
        grounding = float(payload.get("grounding", 0.0))
    except (TypeError, ValueError):
        grounding = 0.0

    used_sources = [
        int(index)
        for index in payload.get("used_sources") or []
        if isinstance(index, (int, float)) and not isinstance(index, bool)
    ]

    return GroundedAnswer(
        answer=answer,
        grounding=max(0.0, min(1.0, grounding)),
        used_sources=used_sources,
    )
