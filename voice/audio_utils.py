"""
In-memory audio helpers. Nothing is written to disk.

Browser recordings arrive as bytes (WAV from st.audio_input, but any format PyAV can
read is accepted). They are decoded to 16 kHz mono float32 - the exact input Whisper
expects - validated (length, silence) and gain-normalised so quiet microphones are
still recognised.
"""

import hashlib
import io
import logging
import struct
import wave
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import config

logger = logging.getLogger("voice.audio")

TARGET_SR = 16000


class VoiceError(Exception):
    """Base class for voice errors. `user_message` is safe to show to the candidate."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class EmptyRecordingError(VoiceError):
    pass


class InvalidAudioError(VoiceError):
    pass


@dataclass
class AudioInfo:
    duration_s: float
    sample_rate: int
    channels: int
    rms: float          # in 16-bit units (0-32767) before gain normalisation
    peak: float         # 0-1 before gain normalisation
    gain: float = 1.0   # gain applied by normalize_gain()


def audio_fingerprint(audio_bytes: bytes) -> str:
    """Short hash used to avoid re-transcribing the same recording on Streamlit reruns."""
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples.astype(np.float32)
    n_out = int(round(samples.size * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _decode_wav(audio_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Tolerant WAV parser (no external libraries).

    Handles what browsers actually produce: 8/16/24/32-bit PCM, 32/64-bit IEEE float,
    WAVE_FORMAT_EXTENSIBLE, extra chunks, and streaming headers whose data size is
    0 / 0xFFFFFFFF / larger than the file. (Python's `wave` module rejects several of these.)
    """
    b = audio_bytes
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid = b[pos:pos + 4]
        size = struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body_start = pos + 8
        if cid == b"data":
            end = len(b) if size in (0, 0xFFFFFFFF) or body_start + size > len(b) else body_start + size
            data = b[body_start:end]
            break
        if cid == b"fmt ":
            fmt = b[body_start:body_start + size]
        pos = body_start + size + (size & 1)
    if fmt is None or len(fmt) < 16 or data is None:
        raise ValueError("WAV is missing fmt/data chunk")

    audio_format, channels, rate, _byte_rate, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if audio_format == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE
        audio_format = struct.unpack("<H", fmt[24:26])[0]
    if channels < 1 or rate < 1000:
        raise ValueError("invalid WAV header")
    width = max(1, bits // 8)
    data = data[: len(data) - len(data) % (width * channels)]

    if audio_format == 1:  # PCM
        if width == 1:
            x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
            ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
            x = ints.astype(np.float32) / 8388608.0
        elif width == 4:
            x = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported PCM width {width}")
    elif audio_format == 3:  # IEEE float
        if width == 4:
            x = np.frombuffer(data, dtype="<f4").astype(np.float32)
        elif width == 8:
            x = np.frombuffer(data, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"unsupported float width {width}")
        x = np.nan_to_num(np.clip(x, -1.0, 1.0))
    else:
        raise ValueError(f"unsupported WAV encoding {audio_format}")

    if channels > 1:
        x = x[: x.size - x.size % channels].reshape(-1, channels).mean(axis=1)
    return _resample_linear(x, rate), rate, channels


def _decode_with_pyav(audio_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Decode compressed audio (mp3/webm/ogg/m4a...) with PyAV directly.

    Written to work across PyAV versions (resample() returns a frame in old versions
    and a list in new ones; pts handling changed) instead of relying on
    faster_whisper.decode_audio, which raises TypeError with some PyAV releases.
    """
    import av  # installed with faster-whisper

    chunks = []
    src_rate, src_channels = TARGET_SR, 1
    with av.open(io.BytesIO(audio_bytes), mode="r") as container:
        stream = next(s for s in container.streams if s.type == "audio")
        src_rate = int(getattr(stream, "rate", 0) or getattr(stream.codec_context, "sample_rate", 0) or TARGET_SR)
        src_channels = int(getattr(stream, "channels", 0) or getattr(stream.codec_context, "channels", 0) or 1)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)

        def _collect(out):
            for fr in (out if isinstance(out, (list, tuple)) else [out]):
                if fr is not None:
                    chunks.append(np.asarray(fr.to_ndarray()).reshape(-1))

        for frame in container.decode(stream):
            try:
                frame.pts = None  # avoids timestamp errors in some files
            except (AttributeError, TypeError, ValueError):
                pass
            _collect(resampler.resample(frame))
        try:
            _collect(resampler.resample(None))  # flush
        except (TypeError, ValueError):
            pass
    if not chunks:
        raise ValueError("no audio frames decoded")
    samples = np.concatenate(chunks).astype(np.float32) / 32768.0
    return samples, src_rate, src_channels


def decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Decode any recording to 16 kHz mono float32 in [-1, 1].

    Returns (samples, original_sample_rate, original_channels).
    Raises EmptyRecordingError / InvalidAudioError.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        raise EmptyRecordingError("No audio was captured. Please record your answer again.")

    errors = []
    # 1) WAV (what st.audio_input produces) - own parser, no external libraries.
    if audio_bytes[:4] == b"RIFF":
        try:
            return _decode_wav(audio_bytes)
        except (ValueError, struct.error) as exc:
            errors.append(f"wav: {exc}")

    # 2) Anything else via PyAV directly.
    try:
        return _decode_with_pyav(audio_bytes)
    except ImportError:
        errors.append("pyav: not installed")
    except Exception as exc:  # noqa: BLE001 - PyAV raises many error types
        errors.append(f"pyav: {type(exc).__name__}: {exc}")

    # 3) Last resort: faster-whisper's own decoder.
    try:
        from faster_whisper.audio import decode_audio as fw_decode

        samples = fw_decode(io.BytesIO(audio_bytes), sampling_rate=TARGET_SR)
        return np.asarray(samples, dtype=np.float32), TARGET_SR, 1
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        errors.append(f"faster-whisper: {type(exc).__name__}: {exc}")

    logger.warning("Could not decode recording (%d bytes, header %r): %s",
                   len(audio_bytes), audio_bytes[:4], " | ".join(errors))
    raise InvalidAudioError(
        "The recording format could not be read. Please record again using the microphone button."
    )


# --------------------------------------------------------------------------- #
# Validation / normalisation
# --------------------------------------------------------------------------- #
def validate_samples(samples: np.ndarray, src_sr: int = TARGET_SR, channels: int = 1) -> AudioInfo:
    if samples.size == 0:
        raise EmptyRecordingError("The recording is empty. Please record your answer again.")

    duration = samples.size / float(TARGET_SR)
    if duration < config.MIN_RECORDING_SECONDS:
        raise EmptyRecordingError(
            f"The recording is too short ({duration:.1f}s). Please speak for at least "
            f"{config.MIN_RECORDING_SECONDS:.0f} second(s)."
        )
    if duration > config.MAX_RECORDING_SECONDS:
        raise InvalidAudioError(
            f"The recording is too long ({duration:.0f}s). Please keep answers under "
            f"{config.MAX_RECORDING_SECONDS:.0f} seconds."
        )

    centred = samples - float(np.mean(samples))
    rms = float(np.sqrt(np.mean(centred.astype(np.float64) ** 2))) * 32768.0
    peak = float(np.max(np.abs(centred))) if centred.size else 0.0
    if rms < config.SILENCE_RMS_THRESHOLD:
        raise EmptyRecordingError(
            "We couldn't hear anything in that recording. Check that the right microphone "
            "is selected in your browser/Windows sound settings and is not muted, then try again."
        )
    return AudioInfo(duration_s=duration, sample_rate=src_sr, channels=channels, rms=rms, peak=peak)


def normalize_gain(samples: np.ndarray, target_peak: float = 0.9, max_gain: float = 20.0) -> Tuple[np.ndarray, float]:
    """Remove DC offset and boost quiet recordings so Whisper hears them clearly."""
    centred = (samples - float(np.mean(samples))).astype(np.float32)
    peak = float(np.max(np.abs(centred))) if centred.size else 0.0
    if peak <= 0:
        return centred, 1.0
    gain = min(max_gain, target_peak / peak) if peak < 0.5 else 1.0
    if gain != 1.0:
        centred = np.clip(centred * gain, -1.0, 1.0)
    return centred, gain


def prepare_audio(audio_bytes: bytes) -> Tuple[np.ndarray, AudioInfo]:
    """Decode + validate + normalise. Returns (16 kHz mono float32 samples, info)."""
    samples, src_sr, channels = decode_audio(audio_bytes)
    info = validate_samples(samples, src_sr, channels)
    samples, info.gain = normalize_gain(samples)
    logger.info("Recording ok: %.1fs, src %d Hz, %d ch, rms %.0f, gain x%.1f",
                info.duration_s, src_sr, channels, info.rms, info.gain)
    return samples, info


def validate_wav(audio_bytes: bytes) -> AudioInfo:
    """Backward-compatible check used by tests/diagnostics."""
    return prepare_audio(audio_bytes)[1]


def samples_to_wav_bytes(samples: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode normalised samples as 16-bit WAV (used for cloud STT providers)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()
