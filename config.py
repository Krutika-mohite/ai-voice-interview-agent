"""
Central configuration.

The original project created a module-level `llm` (ChatOllama / llama3,
temperature 0) that every agent imports with `from config import llm`.
That is preserved: `llm` is still exported here with the same defaults.

Everything else (provider choice, interview length, voice settings) is read
from environment variables / a local `.env` file. No secrets are hard-coded.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory if present (never committed)


# --------------------------------------------------------------------------- #
# Small env helpers
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Logging (never logs candidate answers or audio - see llm_utils / voice/)
# --------------------------------------------------------------------------- #
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("config")


# --------------------------------------------------------------------------- #
# LLM
#   ollama  local model on this computer (default - the original project setup)
#   groq    Groq's free hosted API (used for free cloud deployment; no Ollama needed)
#   openai  OpenAI, or any OpenAI-compatible API via LLM_BASE_URL
# --------------------------------------------------------------------------- #
LLM_PROVIDER = _env_str("LLM_PROVIDER", "ollama").lower()  # "ollama" | "groq" | "openai"
HOSTED_LLM = LLM_PROVIDER != "ollama"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_MODEL = _env_str("OPENAI_MODEL", "gpt-4o-mini")

# Hosted (OpenAI-compatible) settings. Groq gets sensible defaults; only the key is required.
LLM_BASE_URL = _env_str("LLM_BASE_URL", GROQ_BASE_URL if LLM_PROVIDER == "groq" else "")
LLM_API_KEY = (os.getenv("LLM_API_KEY") or (os.getenv("GROQ_API_KEY") if LLM_PROVIDER == "groq"
                                            else os.getenv("OPENAI_API_KEY")) or "").strip()
# openai/gpt-oss-20b: on Groq's free tier (Llama models left the free tier in Aug 2026)
# and it supports schema-enforced JSON, which the answer scoring relies on.
LLM_MODEL = _env_str("LLM_MODEL", "openai/gpt-oss-20b" if LLM_PROVIDER == "groq" else OPENAI_MODEL)
# gpt-oss models "think" before answering; "low" keeps that short (faster, fewer tokens).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low" if LLM_PROVIDER == "groq" else "").strip()
# Retries inside the HTTP client: they honour the provider's "retry after" on rate limits
# (Groq free tier: 30 requests and 8,000 tokens per minute).
LLM_HTTP_RETRIES = _env_int("LLM_HTTP_RETRIES", 4)

LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
# Local: generous, a request can wait in Ollama's queue behind another one on a CPU.
# Hosted: answers take seconds, so fail fast and let the retries handle hiccups.
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 120 if HOSTED_LLM else 600)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
LLM_SEED = _env_int("LLM_SEED", 42)  # reproducible scoring with Ollama
# IMPORTANT: Ollama's default context window is only 2048 tokens and it silently
# drops the START of longer prompts (the JD / question). 8192 = llama3's full window.
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 8192)
# Use the provider's schema-enforced JSON output for scoring (much more reliable).
LLM_STRUCTURED_OUTPUT = _env_bool("LLM_STRUCTURED_OUTPUT", True)


# Speed: keep the model in memory between calls (Ollama unloads it after 5 idle
# minutes by default, and reloading llama3 costs 10-30 s).
OLLAMA_KEEP_ALIVE = _env_str("OLLAMA_KEEP_ALIVE", "30m")
# Output caps (tokens) per kind of call. Local: small, because writing tokens is the
# slowest part on a CPU. Hosted: larger, because reasoning models spend part of the
# budget thinking and a cut-off JSON answer would be useless (hosted output is fast).
LLM_MAX_TOKENS_PARSE = _env_int("LLM_MAX_TOKENS_PARSE", 1200 if HOSTED_LLM else 350)
LLM_MAX_TOKENS_SCREEN = _env_int("LLM_MAX_TOKENS_SCREEN", 1000 if HOSTED_LLM else 300)
LLM_MAX_TOKENS_QUESTION = _env_int("LLM_MAX_TOKENS_QUESTION", 1500 if HOSTED_LLM else 400)
LLM_MAX_TOKENS_EVAL = _env_int("LLM_MAX_TOKENS_EVAL", 1500 if HOSTED_LLM else 450)
LLM_MAX_TOKENS_REPORT = _env_int("LLM_MAX_TOKENS_REPORT", 1500 if HOSTED_LLM else 500)


def _build_llm(max_tokens=None):
    if HOSTED_LLM:
        if not LLM_API_KEY:
            key_name = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "OPENAI_API_KEY (or LLM_API_KEY)"
            raise RuntimeError(
                f"LLM_PROVIDER={LLM_PROVIDER} but {key_name} is not set. Add it to your .env file "
                "locally, or to the app's Secrets when deployed (see DEPLOY.md)."
            )
        from langchain_openai import ChatOpenAI

        kwargs = {}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        if LLM_REASONING_EFFORT:
            kwargs["reasoning_effort"] = LLM_REASONING_EFFORT
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_HTTP_RETRIES,
            **kwargs,
        )

    # Default: local Ollama, exactly as in the original project.
    try:
        from langchain_ollama import ChatOllama  # maintained package

        kwargs = {"num_predict": max_tokens} if max_tokens else {}
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
            num_ctx=OLLAMA_NUM_CTX,
            seed=LLM_SEED,
            keep_alive=OLLAMA_KEEP_ALIVE,
            client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
            **kwargs,
        )
    except ImportError:
        # Fallback to the original (deprecated) import so old installs still work.
        from langchain_community.chat_models import ChatOllama

        kwargs = {"num_predict": max_tokens} if max_tokens else {}
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
            num_ctx=OLLAMA_NUM_CTX,
            keep_alive=OLLAMA_KEEP_ALIVE,
            **kwargs,
        )


llm = _build_llm()
_BASE_LLM = llm
_LLM_BY_BUDGET = {}


def llm_for(max_tokens=None):
    """The shared model, with an output cap when one is given (cached per cap).

    If `llm` was replaced (tests, custom setups) that object is always returned as-is.
    """
    if not max_tokens or llm is not _BASE_LLM:
        return llm
    if max_tokens not in _LLM_BY_BUDGET:
        try:
            _LLM_BY_BUDGET[max_tokens] = _build_llm(max_tokens)
        except Exception:  # noqa: BLE001 - never fail just because a cap can't be applied
            _LLM_BY_BUDGET[max_tokens] = llm
    return _LLM_BY_BUDGET[max_tokens]


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
# Used only when the screening LLM gives a match score but no explicit decision.
SCREENING_PASS_SCORE = _env_int("SCREENING_PASS_SCORE", 60)


# --------------------------------------------------------------------------- #
# Interview
# --------------------------------------------------------------------------- #
MAX_INTERVIEW_QUESTIONS = _env_int("MAX_INTERVIEW_QUESTIONS", 5)
# Bounds for the "number of questions" control in the UI.
INTERVIEW_QUESTIONS_MIN = _env_int("INTERVIEW_QUESTIONS_MIN", 3)
INTERVIEW_QUESTIONS_MAX = _env_int("INTERVIEW_QUESTIONS_MAX", 10)
INTERVIEW_START_DIFFICULTY = _env_str("INTERVIEW_START_DIFFICULTY", "medium").lower()

# Adaptive difficulty: score (0-10) at/above which the next question gets harder,
# and below which it gets easier / becomes a clarifying follow-up.
ADAPT_HARDER_AT = _env_float("ADAPT_HARDER_AT", 8.0)
ADAPT_EASIER_BELOW = _env_float("ADAPT_EASIER_BELOW", 5.0)

# Final recommendation thresholds on the average interview score (0-10).
STRONG_HIRE_MIN = _env_float("STRONG_HIRE_MIN", 8.0)
HIRE_MIN = _env_float("HIRE_MIN", 6.5)
CONSIDER_MIN = _env_float("CONSIDER_MIN", 5.0)


# --------------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------------- #
# Speech-to-text: "faster_whisper" (local, free, no key) | "groq" (free API, needs
# GROQ_API_KEY; best for cloud hosting) | "openai"
STT_PROVIDER = _env_str("STT_PROVIDER", "faster_whisper").lower()
# Optional second provider tried if the first fails ("" = none, "groq" or "openai").
STT_FALLBACK_PROVIDER = os.getenv("STT_FALLBACK_PROVIDER", "").strip().lower()
STT_LANGUAGE = _env_str("STT_LANGUAGE", "en")
# small.en is far more accurate than base.en for accented / technical English.
# Options: tiny.en, base.en, small.en, medium.en, distil-large-v3, large-v3
WHISPER_MODEL_SIZE = _env_str("WHISPER_MODEL_SIZE", "small.en")
# Optional: path to a model folder downloaded manually (for networks that block
# huggingface.co). Overrides WHISPER_MODEL_SIZE when set.
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "").strip()
WHISPER_DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT", "").strip() or None
WHISPER_DEVICE = _env_str("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _env_str("WHISPER_COMPUTE_TYPE", "int8")
# 1 = greedy decoding, roughly 2x faster than 5 with a very small accuracy cost on
# clear speech; Whisper still falls back to retries on hard audio. Use 5 for max accuracy.
WHISPER_BEAM_SIZE = _env_int("WHISPER_BEAM_SIZE", 1)
# CPU threads for speech recognition (0 = library default, which often uses only 4).
WHISPER_CPU_THREADS = _env_int("WHISPER_CPU_THREADS", min(8, os.cpu_count() or 4))

# Answer scoring: run in the background while the candidate answers the next question.
EVAL_IN_BACKGROUND = _env_bool("EVAL_IN_BACKGROUND", True)
# Answers scored at the same time at the end. Ollama only runs them truly in parallel
# when OLLAMA_NUM_PARALLEL > 1 is set for the Ollama server.
EVAL_MAX_WORKERS = _env_int("EVAL_MAX_WORKERS", 1)
# Give Whisper the question as context so technical terms are spelled correctly.
STT_USE_QUESTION_CONTEXT = _env_bool("STT_USE_QUESTION_CONTEXT", True)
# Below this confidence (0-1) the candidate is asked to check the transcript.
STT_LOW_CONFIDENCE = _env_float("STT_LOW_CONFIDENCE", 0.55)
OPENAI_STT_MODEL = _env_str("OPENAI_STT_MODEL", "whisper-1")
# Groq Whisper (free tier: 20 requests/min, 2,000/day, 2 hours of audio per hour)
GROQ_STT_MODEL = _env_str("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()

# Text-to-speech: "edge" (free, no key) | "gtts" (free, no key) | "openai"
TTS_PROVIDER = _env_str("TTS_PROVIDER", "edge").lower()
TTS_FALLBACK_TO_GTTS = _env_bool("TTS_FALLBACK_TO_GTTS", True)
EDGE_TTS_VOICE = _env_str("EDGE_TTS_VOICE", "en-US-AriaNeural")
EDGE_TTS_RATE = _env_str("EDGE_TTS_RATE", "+0%")
GTTS_LANG = _env_str("GTTS_LANG", "en")
GTTS_TLD = _env_str("GTTS_TLD", "com")
OPENAI_TTS_MODEL = _env_str("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = _env_str("OPENAI_TTS_VOICE", "alloy")
VOICE_REQUEST_TIMEOUT_SECONDS = _env_int("VOICE_REQUEST_TIMEOUT_SECONDS", 60)

# Recording validation
MIN_RECORDING_SECONDS = _env_float("MIN_RECORDING_SECONDS", 1.0)
MAX_RECORDING_SECONDS = _env_float("MAX_RECORDING_SECONDS", 300.0)
SILENCE_RMS_THRESHOLD = _env_float("SILENCE_RMS_THRESHOLD", 20.0)  # 16-bit PCM RMS

# --------------------------------------------------------------------------- #
# Proctoring (camera + fullscreen). Runs in the candidate's browser.
# --------------------------------------------------------------------------- #
PROCTORING_ENABLED = _env_bool("PROCTORING_ENABLED", True)
PROCTOR_MAX_VIOLATIONS = _env_int("PROCTOR_MAX_VIOLATIONS", 3)        # Nth camera warning ends it
PROCTOR_NO_FACE_SECONDS = _env_float("PROCTOR_NO_FACE_SECONDS", 3.0)
PROCTOR_MULTI_FACE_SECONDS = _env_float("PROCTOR_MULTI_FACE_SECONDS", 2.0)
PROCTOR_LOOK_AWAY_SECONDS = _env_float("PROCTOR_LOOK_AWAY_SECONDS", 4.0)
PROCTOR_FACE_COVER_SECONDS = _env_float("PROCTOR_FACE_COVER_SECONDS", 4.0)
PROCTOR_YAW_DEGREES = _env_float("PROCTOR_YAW_DEGREES", 35.0)          # head turned left/right
PROCTOR_PITCH_DEGREES = _env_float("PROCTOR_PITCH_DEGREES", 25.0)      # head tilted up/down
PROCTOR_COOLDOWN_SECONDS = _env_float("PROCTOR_COOLDOWN_SECONDS", 8.0)
PROCTOR_REPEAT_SECONDS = _env_float("PROCTOR_REPEAT_SECONDS", 15.0)
PROCTOR_FOCUS_GRACE_SECONDS = _env_float("PROCTOR_FOCUS_GRACE_SECONDS", 1.5)
# After a technical problem is fixed, time the candidate has to return to fullscreen
PROCTOR_RETURN_SECONDS = _env_float("PROCTOR_RETURN_SECONDS", 60.0)
_MP_VERSION = "0.10.14"
PROCTOR_MEDIAPIPE_BUNDLE = _env_str(
    "PROCTOR_MEDIAPIPE_BUNDLE",
    f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{_MP_VERSION}/vision_bundle.mjs")
PROCTOR_MEDIAPIPE_WASM = _env_str(
    "PROCTOR_MEDIAPIPE_WASM", f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{_MP_VERSION}/wasm")
PROCTOR_FACE_MODEL_URL = _env_str(
    "PROCTOR_FACE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
PROCTOR_HAND_MODEL_URL = _env_str(
    "PROCTOR_HAND_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Access: when set, the app asks for this password first (recommended for a public link,
# so strangers cannot use up the free API quota). Leave empty to disable.
APP_PASSWORD = (os.getenv("APP_PASSWORD") or "").strip()

# UI behaviour
AUTOPLAY_QUESTIONS = _env_bool("AUTOPLAY_QUESTIONS", True)
SHOW_SCORES_TO_CANDIDATE = _env_bool("SHOW_SCORES_TO_CANDIDATE", True)
ALLOW_TEXT_ANSWERS = _env_bool("ALLOW_TEXT_ANSWERS", True)

logger.info(
    "Config loaded: LLM=%s (%s), STT=%s, TTS=%s, max_questions=%d",
    LLM_PROVIDER, LLM_MODEL if HOSTED_LLM else OLLAMA_MODEL, STT_PROVIDER, TTS_PROVIDER,
    MAX_INTERVIEW_QUESTIONS,
)
