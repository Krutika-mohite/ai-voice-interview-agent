"""
Shared helpers for calling the LLM safely.

- safe_invoke(): retries transient failures and raises a single LLMError with a
  candidate-safe message (no stack traces, keys or prompts are shown in the UI).
- extract_json(): pulls a JSON object out of an LLM reply (local models often
  wrap JSON in prose or ``` fences).
"""

import json
import logging
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import config

logger = logging.getLogger("llm_utils")


# --------------------------------------------------------------------------- #
# Foreground priority
# --------------------------------------------------------------------------- #
# Ollama on a CPU answers one request at a time. Background answer-scoring must never
# delay the next question the candidate is waiting for, so background jobs wait while a
# foreground call (marked with `foreground()`) is running.
_fg_lock = threading.Lock()
_fg_count = 0


@contextmanager
def foreground():
    global _fg_count
    with _fg_lock:
        _fg_count += 1
    try:
        yield
    finally:
        with _fg_lock:
            _fg_count -= 1


def foreground_busy() -> bool:
    return _fg_count > 0


def wait_for_foreground(max_wait: float = 900.0, poll: float = 0.25) -> None:
    """Block a background job while the candidate is waiting for a foreground call."""
    end = time.time() + max_wait
    while foreground_busy() and time.time() < end:
        time.sleep(poll)


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable response. Message is user-safe."""


class LLMBadOutput(LLMError):
    """The LLM answered, but not in the required format (even after a corrective retry)."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


def _user_message_for(exc: Exception) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if config.LLM_PROVIDER == "ollama" and any(
        k in text for k in ("connect", "refused", "unreachable", "failed to establish")
    ):
        return (
            "The interview AI model is not reachable. Please make sure Ollama is running "
            f"(`ollama serve`) and the '{config.OLLAMA_MODEL}' model is pulled, then retry."
        )
    if any(k in text for k in ("timeout", "timed out")):
        return "The interview AI took too long to respond. Please retry."
    if any(k in text for k in ("auth", "401", "api key", "permission")):
        return "The interview AI rejected the credentials. Please check the API key configuration."
    if any(k in text for k in ("tokens per day", "requests per day", "tpd", "rpd")):
        return ("The free AI quota for today has been used up. It resets daily; "
                "please try again later.")
    if any(k in text for k in ("rate", "429", "quota")):
        return "The interview AI is rate-limited right now. Please wait a moment and retry."
    if any(k in text for k in ("connect", "network", "dns", "ssl")):
        return "A network error occurred while contacting the interview AI. Please retry."
    return "The interview AI could not respond. Please retry."


def safe_invoke(prompt: str, retries: Optional[int] = None, max_tokens: Optional[int] = None) -> str:
    """Invoke config.llm with retries. Returns the text content or raises LLMError."""
    attempts = 1 + (config.LLM_MAX_RETRIES if retries is None else retries)
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            res = config.llm_for(max_tokens).invoke(prompt)
            content = getattr(res, "content", res)
            if isinstance(content, list):  # some providers return content parts
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            content = (content or "").strip()
            if content:
                return content
            last_exc = ValueError("empty response")
            logger.warning("LLM returned an empty response (attempt %d/%d)", attempt, attempts)
        except Exception as exc:  # provider-specific exceptions vary widely
            last_exc = exc
            logger.warning(
                "LLM call failed (attempt %d/%d): %s", attempt, attempts, type(exc).__name__
            )
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 4))

    raise LLMError(_user_message_for(last_exc or Exception("unknown")))


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object found in `text`, or None."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for chunk in candidates:
        start = chunk.find("{")
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(chunk)):
                ch = chunk[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(chunk[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
            start = chunk.find("{", start + 1)
    return None


def invoke_json(prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Invoke the LLM expecting a JSON object; one corrective retry if parsing fails."""
    reply = safe_invoke(prompt, max_tokens=max_tokens)
    data = extract_json(reply)
    if data is not None:
        return data
    logger.warning("LLM reply was not valid JSON; asking once more")
    reply = safe_invoke(
        prompt + "\n\nIMPORTANT: Reply with ONE valid JSON object only. No prose, no code fences.",
        max_tokens=max_tokens,
    )
    data = extract_json(reply)
    if data is None:
        raise LLMBadOutput("The interview AI returned an unreadable response. Please retry.", raw=reply)
    return data


# --------------------------------------------------------------------------- #
# Structured output (schema-enforced JSON)
# --------------------------------------------------------------------------- #
def normalize_key(key: str) -> str:
    """'Technical Correctness' / 'technical-correctness' -> 'technical_correctness'."""
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def normalize_keys(data: Any, aliases: Optional[Dict[str, str]] = None) -> Any:
    """Recursively normalise dict keys and apply aliases (e.g. 'clarity' -> 'explanation_quality')."""
    aliases = aliases or {}
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            nk = normalize_key(k)
            nk = aliases.get(nk, nk)
            out[nk] = normalize_keys(v, aliases)
        return out
    if isinstance(data, list):
        return [normalize_keys(v, aliases) for v in data]
    return data


def _schema_hint(model_cls) -> str:
    fields = []
    for name, field in model_cls.model_fields.items():
        desc = field.description or ""
        fields.append(f'  "{name}": {desc}')
    return "{\n" + ",\n".join(fields) + "\n}"


def invoke_structured(prompt: str, model_cls, aliases: Optional[Dict[str, str]] = None,
                      flatten_keys: tuple = (), max_tokens: Optional[int] = None):
    """Return an instance of the pydantic `model_cls` produced by the LLM.

    1) Provider-native structured output (Ollama `format=<json schema>` / OpenAI
       json_schema) - the model cannot emit wrong key names or non-JSON.
    2) Fallback: plain JSON prompt + tolerant parsing (key normalisation, aliases,
       nested dicts flattened) + pydantic validation, with one corrective retry.

    Raises LLMError (service problem) or LLMBadOutput (format problem).
    """
    from pydantic import ValidationError

    model = config.llm_for(max_tokens)
    if config.LLM_STRUCTURED_OUTPUT and hasattr(model, "with_structured_output"):
        for attempt in range(1 + max(0, config.LLM_MAX_RETRIES - 1)):
            try:
                runnable = model.with_structured_output(model_cls, method="json_schema")
                result = runnable.invoke(prompt)
                if isinstance(result, model_cls):
                    return result
                if isinstance(result, dict):
                    return model_cls.model_validate(normalize_keys(result, aliases))
                logger.warning("Structured output returned %s; falling back", type(result).__name__)
                break
            except Exception as exc:  # noqa: BLE001 - provider/parse errors vary
                logger.warning("Structured output failed (%s); attempt %d", type(exc).__name__, attempt + 1)
                if "connect" in str(exc).lower() or "refused" in str(exc).lower():
                    break  # service is down; safe_invoke below produces the proper message
        # fall through to the text-JSON path

    json_prompt = (
        prompt
        + "\n\nReturn ONLY one JSON object with exactly these keys:\n"
        + _schema_hint(model_cls)
    )
    last_raw = ""
    for attempt in range(2):
        reply = safe_invoke(json_prompt, max_tokens=max_tokens)
        last_raw = reply
        data = extract_json(reply)
        if data is not None:
            data = normalize_keys(data, aliases)
            for nested in flatten_keys:  # e.g. {"scores": {...}} -> top level
                if isinstance(data.get(nested), dict):
                    data = {**data.pop(nested), **data}
            try:
                return model_cls.model_validate(data)
            except ValidationError as exc:
                logger.warning("LLM JSON failed validation: %d error(s)", exc.error_count())
        json_prompt = (
            prompt
            + "\n\nYour previous reply was not valid. Return ONLY one valid JSON object, "
              "no prose, with exactly these keys:\n" + _schema_hint(model_cls)
        )
    raise LLMBadOutput("The interview AI returned an unreadable response. Please retry.", raw=last_raw)


def to_score(value: Any, lo: float = 0.0, hi: float = 10.0) -> Optional[float]:
    """Coerce '8', 8, '8/10', 8.5 into a float clamped to [lo, hi]; None if impossible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not m:
            return None
        num = float(m.group(0))
    return max(lo, min(hi, num))
