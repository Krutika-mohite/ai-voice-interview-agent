"""
Text-to-Speech: interview question text -> MP3 bytes for the browser to play.

Providers (TTS_PROVIDER in .env):
  edge    (default) Microsoft Edge neural voices via edge-tts - natural, free, no key
  gtts    Google Translate TTS via gTTS - free, no key (also used as fallback)
  openai  OpenAI speech API (needs OPENAI_API_KEY)

All audio is generated in memory; no files are written.
"""

import asyncio
import io
import logging
import threading
from typing import Tuple

import config
from voice.audio_utils import VoiceError

logger = logging.getLogger("voice.tts")

MAX_TTS_CHARS = 1000


class TTSError(VoiceError):
    """Speech could not be generated. The question text is still shown, so this is non-fatal."""


def _run_coro(coro):
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _edge_tts(text: str) -> bytes:
    import edge_tts

    async def _synth() -> bytes:
        communicate = edge_tts.Communicate(text, config.EDGE_TTS_VOICE, rate=config.EDGE_TTS_RATE)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    return _run_coro(asyncio.wait_for(_synth(), timeout=config.VOICE_REQUEST_TIMEOUT_SECONDS))


def _gtts(text: str) -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang=config.GTTS_LANG, tld=config.GTTS_TLD).write_to_fp(buf)
    return buf.getvalue()


def _openai_tts(text: str) -> bytes:
    from openai import OpenAI

    client = OpenAI(timeout=config.VOICE_REQUEST_TIMEOUT_SECONDS, max_retries=2)
    response = client.audio.speech.create(
        model=config.OPENAI_TTS_MODEL,
        voice=config.OPENAI_TTS_VOICE,
        input=text,
        response_format="mp3",
    )
    return response.content


_PROVIDERS = {"edge": _edge_tts, "gtts": _gtts, "openai": _openai_tts}


def synthesize(text: str) -> Tuple[bytes, str]:
    """Return (audio_bytes, mime_type). Tries the configured provider, then gTTS.

    Raises TTSError with a candidate-safe message if every provider fails.
    """
    text = (text or "").strip()
    if not text:
        raise TTSError("There is no question text to read aloud.")
    text = text[:MAX_TTS_CHARS]

    order = [config.TTS_PROVIDER if config.TTS_PROVIDER in _PROVIDERS else "edge"]
    if config.TTS_FALLBACK_TO_GTTS and "gtts" not in order:
        order.append("gtts")

    for name in order:
        try:
            audio = _PROVIDERS[name](text)
            if audio and len(audio) > 100:
                logger.info("TTS ok via %s (%d bytes)", name, len(audio))
                return audio, "audio/mp3"
            logger.warning("TTS via %s returned no audio", name)
        except ImportError:
            logger.warning("TTS provider '%s' is not installed", name)
        except Exception as exc:  # network errors, service changes, timeouts...
            logger.warning("TTS via %s failed: %s", name, type(exc).__name__)

    raise TTSError(
        "The question audio couldn't be generated right now (network or voice service issue). "
        "Please read the question on screen, or press 'Retry audio'."
    )
