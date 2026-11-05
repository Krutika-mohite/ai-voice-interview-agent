"""Voice I/O layer around the LangGraph interview agent (STT in, TTS out).

Nothing here makes interview decisions: it only converts audio <-> text.
"""

from voice.audio_utils import (
    EmptyRecordingError,
    InvalidAudioError,
    VoiceError,
    audio_fingerprint,
    prepare_audio,
    validate_wav,
)
from voice.speech_to_text import (
    STTError,
    STTSetupError,
    SpeechNotRecognizedError,
    TranscriptResult,
    preload_model,
    transcribe,
    transcribe_detailed,
)
from voice.text_to_speech import TTSError, synthesize

__all__ = [
    "VoiceError",
    "EmptyRecordingError",
    "InvalidAudioError",
    "STTError",
    "STTSetupError",
    "SpeechNotRecognizedError",
    "TTSError",
    "TranscriptResult",
    "audio_fingerprint",
    "prepare_audio",
    "validate_wav",
    "transcribe",
    "transcribe_detailed",
    "synthesize",
    "preload_model",
]
