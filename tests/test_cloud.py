"""
Tests for the cloud (free hosting) settings: Groq as the AI model and for speech
recognition. No network access is needed. Run from the project root:

    python -m pytest -q tests/test_cloud.py
"""

import json
import os
import subprocess
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from voice import speech_to_text  # noqa: E402

PROBE = (
    "import json, config; print(json.dumps({k: getattr(config, k) for k in ["
    "'LLM_PROVIDER','HOSTED_LLM','LLM_BASE_URL','LLM_MODEL','LLM_REASONING_EFFORT',"
    "'LLM_MAX_TOKENS_QUESTION','LLM_MAX_TOKENS_EVAL','STT_PROVIDER','APP_PASSWORD']}))"
)


def _config_in_subprocess(env_overrides):
    """Import config.py fresh with the given environment; return its settings or the error."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LLM_", "GROQ_", "OPENAI_", "STT_", "APP_PASSWORD"))}
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [ROOT, *sys.path] if p)
    env["LOG_LEVEL"] = "ERROR"
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return None, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), ""


def test_local_defaults_are_unchanged():
    cfg, err = _config_in_subprocess({})
    assert cfg is not None, err
    assert cfg["LLM_PROVIDER"] == "ollama" and cfg["HOSTED_LLM"] is False
    assert cfg["LLM_MAX_TOKENS_QUESTION"] == 400          # small caps for a laptop CPU
    assert cfg["APP_PASSWORD"] == ""


def test_groq_needs_only_the_key():
    cfg, err = _config_in_subprocess({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_test",
                                      "STT_PROVIDER": "groq", "APP_PASSWORD": "secret"})
    assert cfg is not None, err
    assert cfg["LLM_BASE_URL"] == "https://api.groq.com/openai/v1"
    assert cfg["LLM_MODEL"] == "openai/gpt-oss-20b"
    assert cfg["LLM_REASONING_EFFORT"] == "low"
    assert cfg["LLM_MAX_TOKENS_EVAL"] >= 1000              # room for the model's reasoning
    assert cfg["STT_PROVIDER"] == "groq" and cfg["APP_PASSWORD"] == "secret"


def test_groq_without_a_key_gives_a_clear_error():
    cfg, err = _config_in_subprocess({"LLM_PROVIDER": "groq"})
    assert cfg is None and "GROQ_API_KEY" in err


def test_model_can_be_overridden():
    cfg, err = _config_in_subprocess({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "k",
                                      "LLM_MODEL": "openai/gpt-oss-120b", "LLM_REASONING_EFFORT": "medium"})
    assert cfg["LLM_MODEL"] == "openai/gpt-oss-120b" and cfg["LLM_REASONING_EFFORT"] == "medium", err


def _fake_openai(calls):
    mod = types.ModuleType("openai")

    class OpenAIError(Exception):
        pass

    for name in ("AuthenticationError", "APIConnectionError", "RateLimitError"):
        setattr(mod, name, type(name, (OpenAIError,), {}))
    mod.OpenAIError = OpenAIError

    class OpenAI:
        def __init__(self, **kw):
            def create(**req):
                calls.append({"client": kw, "request": req})
                return types.SimpleNamespace(text="A tuple is immutable and a list is mutable.")
            self.audio = types.SimpleNamespace(transcriptions=types.SimpleNamespace(create=create))

    mod.OpenAI = OpenAI
    return mod


def test_groq_speech_recognition_uses_groq_whisper(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(calls))
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    samples = (0.2 * np.sin(np.linspace(0, 400, 32000))).astype(np.float32)
    text, conf = speech_to_text._transcribe_groq(samples, prompt="list vs tuple")
    assert text.startswith("A tuple") and conf > 0
    call = calls[0]
    assert call["client"]["base_url"] == "https://api.groq.com/openai/v1"
    assert call["client"]["api_key"] == "gsk_test"
    assert call["request"]["model"] == config.GROQ_STT_MODEL
    assert call["request"]["prompt"] == "list vs tuple"
    assert call["request"]["file"][1][:4] == b"RIFF"          # sent as WAV


def test_groq_speech_recognition_without_key_is_a_setup_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", _fake_openai([]))
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    samples = np.zeros(32000, dtype=np.float32)
    try:
        speech_to_text._transcribe_groq(samples)
    except speech_to_text.STTSetupError as exc:
        assert "API key" in exc.user_message
    else:
        raise AssertionError("expected STTSetupError")


def test_groq_is_a_registered_speech_provider():
    assert speech_to_text._PROVIDERS["groq"] is speech_to_text._transcribe_groq
