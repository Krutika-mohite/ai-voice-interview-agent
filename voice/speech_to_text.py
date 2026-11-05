"""
Speech-to-Text: candidate voice -> answer text.

Providers (STT_PROVIDER in .env):
  faster_whisper  (default) local Whisper, no API key, audio never leaves the machine
  groq            Groq's free Whisper API (needs GROQ_API_KEY) - fast, ideal for cloud hosting
  openai          OpenAI transcription API (needs OPENAI_API_KEY)
Optional STT_FALLBACK_PROVIDER is tried if the primary provider fails.

Accuracy / reliability measures:
  - audio decoded to 16 kHz mono and gain-normalised (quiet microphones)
  - the interview question is given to Whisper as context, so technical terms
    ("tuple", "FAISS", "LangGraph") are spelled correctly
  - voice-activity filter with generous padding; automatic retry WITHOUT it if it
    removed everything (it can cut quiet speech)
  - known Whisper "phantom" phrases on silence ("Thank you.") are rejected
  - a 0-1 confidence is computed so the UI can ask the candidate to check the text
  - model-loading problems (blocked download, missing Windows runtime) produce a
    specific message instead of a generic failure

Everything is processed in memory; no temporary files are created.
"""

import logging
import math
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

import config
from voice.audio_utils import AudioInfo, VoiceError, prepare_audio, samples_to_wav_bytes

logger = logging.getLogger("voice.stt")

_model_lock = threading.Lock()
_load_lock = threading.Lock()   # background warm-up and a first answer must not load twice


class STTError(VoiceError):
    """STT service/library failure (retryable)."""


class STTSetupError(STTError):
    """Speech recognition is not set up correctly on this machine (needs operator action)."""


class SpeechNotRecognizedError(VoiceError):
    """Audio was fine but no words were recognised."""


@dataclass
class TranscriptResult:
    text: str
    confidence: float       # 0-1 (from Whisper token log-probabilities)
    low_confidence: bool
    duration_s: float
    provider: str


# Phrases Whisper is known to invent on silence/noise.
_HALLUCINATIONS = {
    "thank you", "thank you very much", "thanks for watching", "thank you for watching",
    "you", "bye", "okay", "ok", "so", "uh", "um", "hmm", "please subscribe",
    "subtitles by the amara org community", "i m sorry",
}


def _norm_words(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _is_hallucination(text: str) -> bool:
    words = _norm_words(text)
    words = re.sub(r"\s+", " ", words)
    return not words or words in _HALLUCINATIONS or len(words.replace(" ", "")) < 2


def build_context_prompt(question: str = "", topic: str = "") -> Optional[str]:
    """Short prompt that primes Whisper with the vocabulary of the current question."""
    if not config.STT_USE_QUESTION_CONTEXT or not (question or topic):
        return None
    text = f"Technical job interview answer. Topic: {topic}. Question: {question}".strip()
    return text[:600]


# --------------------------------------------------------------------------- #
# faster-whisper (local)
# --------------------------------------------------------------------------- #
def _setup_error_message(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "dll" in text or "vcruntime" in text or "msvcp" in text:
        return ("Speech recognition could not start: a Windows system component is missing. "
                "Install the 'Microsoft Visual C++ Redistributable (x64)' from Microsoft, restart "
                "the app, and try again. (Details are in the terminal window.)")
    if any(k in text for k in ("connect", "resolve", "proxy", "ssl", "certificate", "timed out",
                               "huggingface", "hub", "offline", "network", "max retries")):
        return ("The speech-recognition model could not be downloaded (the network may be blocking "
                "huggingface.co). Connect to an open network for the first run, or download the model "
                "manually and set WHISPER_MODEL_PATH in .env. Run `python diagnose.py` for details.")
    if isinstance(exc, MemoryError) or "memory" in text:
        return ("Not enough memory to load the speech model. Set WHISPER_MODEL_SIZE=base.en in .env "
                "and restart the app.")
    return ("Speech recognition could not start on this computer. Run `python diagnose.py` in the "
            "project folder to see the exact cause. (Details are in the terminal window.)")


def _load_whisper():
    """Load once per process (thread-safe). Exceptions are not cached, so a later call retries."""
    with _load_lock:
        return _load_whisper_once()


@lru_cache(maxsize=1)
def _load_whisper_once():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        logger.exception("faster-whisper import failed")
        if "dll" in str(exc).lower():
            raise STTSetupError(_setup_error_message(exc)) from exc
        raise STTSetupError(
            "Speech recognition is not installed (faster-whisper). Run "
            "`python -m pip install -r requirements.txt`, then restart the app."
        ) from exc

    model_ref = config.WHISPER_MODEL_PATH or config.WHISPER_MODEL_SIZE
    logger.info("Loading Whisper model '%s' (%s/%s)...", model_ref,
                config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    try:
        # First run downloads the model from Hugging Face (cached afterwards).
        return WhisperModel(
            model_ref,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=config.WHISPER_DOWNLOAD_ROOT,
            cpu_threads=max(0, int(config.WHISPER_CPU_THREADS)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Whisper model load failed")
        raise STTSetupError(_setup_error_message(exc)) from exc


def preload_model() -> Tuple[bool, str]:
    """Warm-up before the interview starts. Returns (ok, user_message)."""
    if config.STT_PROVIDER != "faster_whisper":
        return True, ""
    try:
        _load_whisper()
        return True, ""
    except STTError as exc:
        return False, exc.user_message


def _run_whisper(model, samples: np.ndarray, prompt: Optional[str], use_vad: bool):
    kwargs = dict(
        language=config.STT_LANGUAGE or None,
        beam_size=config.WHISPER_BEAM_SIZE,
        condition_on_previous_text=False,   # prevents repetition loops
        initial_prompt=prompt,
        vad_filter=use_vad,
    )
    if use_vad:
        # Defaults cut quiet speech / pauses too aggressively for interview answers.
        kwargs["vad_parameters"] = {"threshold": 0.35, "min_silence_duration_ms": 1000,
                                    "speech_pad_ms": 500}
    segments, _info = model.transcribe(samples, **kwargs)
    texts, weights, logprobs = [], [], []
    for seg in segments:  # generator - transcription happens while iterating
        text = (seg.text or "").strip()
        if not text:
            continue
        # Drop segments Whisper itself thinks are non-speech and is unsure about.
        if getattr(seg, "no_speech_prob", 0) > 0.75 and getattr(seg, "avg_logprob", 0) < -0.8:
            continue
        texts.append(text)
        weights.append(max(0.1, float(seg.end - seg.start)))
        logprobs.append(float(seg.avg_logprob))
    text = " ".join(texts).strip()
    confidence = math.exp(float(np.average(logprobs, weights=weights))) if logprobs else 0.0
    return text, confidence


def _transcribe_local(samples: np.ndarray, prompt: Optional[str] = None) -> Tuple[str, float]:
    model = _load_whisper()
    attempts = [(prompt, True), (prompt, False)]   # 2nd: no VAD (it may cut quiet speech)
    if prompt:
        attempts.append((None, False))               # 3rd: no context prompt
    text, confidence = "", 0.0
    try:
        with _model_lock:  # one transcription at a time per process
            for i, (p, vad) in enumerate(attempts):
                try:
                    text, confidence = _run_whisper(model, samples, p, vad)
                except Exception as exc:  # noqa: BLE001 - e.g. VAD/onnxruntime issues
                    if i == len(attempts) - 1:
                        raise
                    logger.warning("Whisper pass %d failed (%s); retrying", i + 1, type(exc).__name__)
                    continue
                # Reject output that is just the context prompt echoed back.
                if p and text and _norm_words(text) in _norm_words(p):
                    logger.info("Transcript echoed the context prompt; retrying")
                    text = ""
                    continue
                if not _is_hallucination(text):
                    return text, confidence
                logger.info("Pass %d produced no usable speech; retrying", i + 1)
    except Exception as exc:  # decoding / runtime errors
        logger.exception("Local transcription failed")
        raise STTError("Your answer could not be transcribed. Please try recording again.") from exc
    return text, confidence


# --------------------------------------------------------------------------- #
# OpenAI (cloud)
# --------------------------------------------------------------------------- #
def _transcribe_api(samples: np.ndarray, prompt: Optional[str], *, model: str, api_key: Optional[str],
                    base_url: Optional[str], name: str) -> Tuple[str, float]:
    """Whisper over an OpenAI-compatible API (OpenAI itself, or Groq)."""
    try:
        import openai
        from openai import OpenAI
    except ImportError as exc:
        raise STTSetupError("The OpenAI package is not installed on the server.") from exc
    if api_key is not None and not api_key:
        raise STTSetupError(f"Speech recognition ({name}) has no API key configured. Please type your answer.")

    try:
        client_kwargs = {"timeout": config.VOICE_REQUEST_TIMEOUT_SECONDS, "max_retries": 2}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        kwargs = dict(model=model,
                      file=("answer.wav", samples_to_wav_bytes(samples), "audio/wav"),
                      language=config.STT_LANGUAGE or None)
        if prompt:
            kwargs["prompt"] = prompt
        result = client.audio.transcriptions.create(**kwargs)
        return (getattr(result, "text", "") or "").strip(), 0.9  # API gives no confidence
    except openai.AuthenticationError as exc:
        raise STTSetupError("Speech recognition is misconfigured (API key). Please type your answer.") from exc
    except openai.APIConnectionError as exc:
        raise STTError("Network problem while transcribing. Please check your connection and retry.") from exc
    except openai.RateLimitError as exc:
        raise STTError("Speech recognition is busy right now. Please retry in a moment.") from exc
    except openai.OpenAIError as exc:
        logger.warning("%s STT error: %s", name, type(exc).__name__)
        raise STTError("Speech recognition failed. Please retry or type your answer.") from exc


def _transcribe_openai(samples: np.ndarray, prompt: Optional[str] = None) -> Tuple[str, float]:
    return _transcribe_api(samples, prompt, model=config.OPENAI_STT_MODEL, api_key=None,
                           base_url=None, name="OpenAI")


def _transcribe_groq(samples: np.ndarray, prompt: Optional[str] = None) -> Tuple[str, float]:
    return _transcribe_api(samples, prompt, model=config.GROQ_STT_MODEL, api_key=config.GROQ_API_KEY,
                           base_url=config.GROQ_BASE_URL, name="Groq")


_PROVIDERS = {"faster_whisper": _transcribe_local, "openai": _transcribe_openai, "groq": _transcribe_groq}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def transcribe_detailed(audio_bytes: bytes, question: str = "", topic: str = "") -> TranscriptResult:
    """Decode, validate and transcribe a recording.

    Raises EmptyRecordingError, InvalidAudioError, SpeechNotRecognizedError, STTSetupError
    or STTError (all VoiceError subclasses with a candidate-safe `user_message`).
    """
    samples, info = prepare_audio(audio_bytes)
    prompt = build_context_prompt(question, topic)

    providers = [config.STT_PROVIDER if config.STT_PROVIDER in _PROVIDERS else "faster_whisper"]
    fb = config.STT_FALLBACK_PROVIDER
    if fb in _PROVIDERS and fb not in providers:
        providers.append(fb)

    text, confidence, used, last_error = "", 0.0, providers[0], None
    for name in providers:
        try:
            text, confidence = _PROVIDERS[name](samples, prompt)
            used = name
            if not _is_hallucination(text):
                break
        except STTError as exc:
            last_error = exc
            logger.warning("STT provider %s failed: %s", name, type(exc).__name__)

    if _is_hallucination(text):
        if last_error is not None and not text:
            raise last_error
        raise SpeechNotRecognizedError(
            "Sorry, we couldn't make out any words. Please speak clearly, a little closer to the "
            "microphone, and try again."
        )

    text = re.sub(r"\s+", " ", text).strip()
    result = TranscriptResult(
        text=text,
        confidence=round(confidence, 2),
        low_confidence=confidence < config.STT_LOW_CONFIDENCE,
        duration_s=round(info.duration_s, 1),
        provider=used,
    )
    # Log metadata only - never the candidate's words.
    logger.info("Transcribed %.1fs via %s: %d chars, confidence %.2f",
                info.duration_s, used, len(text), result.confidence)
    return result


def transcribe(audio_bytes: bytes, question: str = "", topic: str = "") -> str:
    """Return just the transcript text (backward compatible)."""
    return transcribe_detailed(audio_bytes, question, topic).text


def info_only(audio_bytes: bytes) -> AudioInfo:
    return prepare_audio(audio_bytes)[1]
