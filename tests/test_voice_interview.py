"""
End-to-end tests for the voice interview flow, using a FAKE LLM (no Ollama/OpenAI
needed) and synthetic audio. Run from the project root:

    python -m pytest -q tests/

For checks against the REAL speech model and LLM, run:  python diagnose.py
"""

import io
import json
import math
import os
import re
import struct
import sys
import uuid
import wave

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import llm_utils  # noqa: E402
from langgraph.types import Command  # noqa: E402

import evaluation_agent as ea  # noqa: E402
from graph import build_graph, pending_interrupt, thread_config  # noqa: E402
from interview_agent import clean_question, next_difficulty  # noqa: E402
from screening_agent import _parse_screening  # noqa: E402
import voice  # noqa: E402
from voice import speech_to_text, text_to_speech  # noqa: E402

KEY_POINTS = ["Lists are mutable", "Tuples are immutable", "Tuples are hashable"]


# --------------------------------------------------------------------------- #
# Fake LLM
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, content):
        self.content = content


def _eval_json(s, style="exact"):
    """Evaluation reply for score s. style='messy' uses wrong key names + nesting + prose."""
    status = "covered" if s >= 8 else "partial" if s >= 5 else "missing"
    cov = [{"point": p, "status": status} for p in KEY_POINTS]
    if style == "messy":
        return ("Sure! Here is the evaluation:\n```json\n" + json.dumps({
            "Key Points": [{"Point": c["point"], "Status": c["status"].upper()} for c in cov],
            "Errors": "None",
            "scores": {"Technical Accuracy": f"{s}/10", "Relevance": s, "Completeness": s,
                       "Depth of Understanding": s, "Clarity": s},
            "Feedback": f"Feedback for score {s}."}) + "\n```")
    return json.dumps({
        "key_point_coverage": cov, "factual_errors": [],
        "technical_correctness": s, "relevance": s, "completeness": s,
        "understanding": s, "explanation_quality": s,
        "feedback": f"Feedback for score {s}."})


class FakeLLM:
    """Answers may carry a tag like "[s9]": the grader then returns score 9 for it, which
    keeps tests deterministic even though answers are scored in background threads."""

    def __init__(self, decision="SELECTED", answer_scores=(9, 9, 3, 6, 7), fail_eval_once=False,
                 eval_style="exact", plain_questions=False, qualities=(), fail_eval_count=0,
                 fail_question_once=False):
        self.decision = decision
        self.answer_scores = list(answer_scores)
        self.fail_eval_count = 1 if fail_eval_once else fail_eval_count
        self.fail_question_once = fail_question_once
        self.eval_style = eval_style
        self.plain_questions = plain_questions
        self.qualities = list(qualities)
        self.calls = {"parse": 0, "screen": 0, "question": 0, "answer_eval": 0, "final": 0}
        self.prompts = {"question": [], "answer_eval": []}

    def invoke(self, prompt):
        if "Extract skills" in prompt:
            self.calls["parse"] += 1
            return _Msg("Candidate Name: John Doe\nSkills: Python, ML\nExperience: 3 years")
        if "compare resume with job description" in prompt:
            self.calls["screen"] += 1
            return _Msg(f"Good fit overall.\nMATCH SCORE: 82\nDECISION: {self.decision}")
        if "senior technical interviewer running" in prompt:
            if self.fail_question_once:
                self.fail_question_once = False
                raise ConnectionError("connection refused")
            self.calls["question"] += 1
            self.prompts["question"].append(prompt)
            n = self.calls["question"]
            if self.plain_questions:
                return _Msg(f'Here is your question:\n\n"**Question {n}:** Explain concept number {n} in Python?"')
            quality = self.qualities[n - 1] if n - 1 < len(self.qualities) else ("none" if n == 1 else "adequate")
            return _Msg(json.dumps({"previous_answer_quality": quality, "difficulty": "",
                                    "question": f"**Question {n}:** Explain concept number {n} in Python?",
                                    "topic": "Python", "key_points": KEY_POINTS}))
        if "grading ONE answer" in prompt:
            self.eval_attempts = getattr(self, "eval_attempts", 0) + 1
            if self.fail_eval_count > 0:
                self.fail_eval_count -= 1
                raise ConnectionError("connection refused")
            self.calls["answer_eval"] += 1
            self.prompts["answer_eval"].append(prompt)
            tag = re.search(r"\[s(\d+)\]", prompt)
            s = int(tag.group(1)) if tag else (self.answer_scores.pop(0) if self.answer_scores else 5)
            return _Msg(_eval_json(s, self.eval_style))
        if "Evaluate candidate based on" in prompt:
            self.calls["final"] += 1
            return _Msg(json.dumps({
                "strengths": ["Python fundamentals"], "weaknesses": ["Depth in ML"],
                "areas_for_improvement": ["Practice system design"],
                "justification": "Solid answers overall."}))
        raise AssertionError("unexpected prompt")


@pytest.fixture
def fake_llm(monkeypatch):
    def _make(**kw):
        llm = FakeLLM(**kw)
        monkeypatch.setattr(config, "llm", llm)
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
        monkeypatch.setattr(llm_utils.time, "sleep", lambda *_: None)
        return llm
    return _make


ANSWER = ("A list is mutable so items can be changed, while a tuple is immutable and "
          "therefore hashable, which lets it be used as a dictionary key.")


def _start(app):
    cfg = thread_config(uuid.uuid4().hex)
    app.invoke({"resume": "John Doe resume text", "jd": "Python developer with ML"}, cfg)
    return cfg


def _answer(app, cfg, text=ANSWER, skipped=False, **extra):
    app.invoke(Command(resume={"answer": text, "skipped": skipped, "source": "voice", **extra}), cfg)


def scored(score):
    """An answer the fake grader scores as `score`."""
    return f"{ANSWER} [s{score}]"


def _wait_background(llm, n, timeout=5.0):
    import time as _t
    end = _t.time() + timeout
    while llm.calls["answer_eval"] < n and _t.time() < end:
        _t.sleep(0.02)


# --------------------------------------------------------------------------- #
# Graph flow
# --------------------------------------------------------------------------- #
def test_full_adaptive_interview(fake_llm):
    llm = fake_llm(qualities=["none", "strong", "strong"])
    app = build_graph()
    cfg = _start(app)

    pending = pending_interrupt(app, cfg)
    assert pending["type"] == "start_interview"
    state = app.get_state(cfg).values
    assert state["candidate_name"] == "John Doe"
    assert state["screening_decision"] == "SELECTED" and state["match_score"] == 82

    app.invoke(Command(resume={"max_questions": 3}), cfg)
    pending = pending_interrupt(app, cfg)
    assert pending["type"] == "answer" and pending["number"] == 1 and pending["total"] == 3
    assert pending["question"] == "Explain concept number 1 in Python?"  # cleaned for TTS
    assert pending["topic"] == "Python"
    assert app.get_state(cfg).values["current_key_points"] == KEY_POINTS

    _answer(app, cfg, text=scored(9), transcript_confidence=0.91)
    assert pending_interrupt(app, cfg)["number"] == 2
    st = app.get_state(cfg).values
    assert st["current_difficulty"] == "hard"           # interviewer judged it strong
    assert st["qa_history"][0]["score"] is None         # not scored while the candidate waits
    _answer(app, cfg, text=scored(9), transcript_edited=True)
    _answer(app, cfg, text=scored(3))                   # weak answer, last question

    assert pending_interrupt(app, cfg) is None
    final = app.get_state(cfg).values
    hist = final["qa_history"]
    assert hist[0]["transcript_confidence"] == 0.91 and hist[1]["transcript_edited"] is True
    # The answer key reaches the grader:
    assert all("Tuples are hashable" in p for p in llm.prompts["answer_eval"])
    # All points covered -> high score; all missing -> low score (scores filled at the end)
    assert hist[0]["score"] >= 9 and hist[2]["score"] <= 3
    assert all(c["status"] == "covered" for c in hist[0]["key_point_coverage"])
    report = final["final_report"]
    assert report["num_questions"] == 3 and report["question_scores"][0]["coverage"] == "3/3 covered"
    assert report["recommendation"] in {"Hire", "Strong Hire"}
    # One LLM call per question, each answer scored exactly once, one report call:
    assert llm.calls == {"parse": 1, "screen": 1, "question": 3, "answer_eval": 3, "final": 1}
    # The next question is written with the previous answer in view (adaptive):
    assert "LAST ANSWER" in llm.prompts["question"][1] and "[s9]" in llm.prompts["question"][1]


def test_answers_are_scored_in_background_not_while_waiting(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 3}), cfg)
    _answer(app, cfg, text=scored(8))
    # Question 2 is already shown; answer 1 is being scored behind the scenes
    assert pending_interrupt(app, cfg)["number"] == 2
    _wait_background(llm, 1)
    assert llm.calls["answer_eval"] == 1
    assert app.get_state(cfg).values["qa_history"][0]["score"] is None   # hidden until the end
    _answer(app, cfg, text=scored(6))
    _answer(app, cfg, text=scored(7))
    final = app.get_state(cfg).values
    assert [q["score"] >= 0 for q in final["qa_history"]] == [True, True, True]
    assert llm.calls["answer_eval"] == 3          # background results reused, no double scoring


def test_weak_answer_makes_next_question_easier(fake_llm):
    fake_llm(qualities=["none", "weak"])
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 3}), cfg)
    _answer(app, cfg, text=scored(3))
    assert app.get_state(cfg).values["current_difficulty"] == "easy"


def test_messy_llm_json_is_still_scored(fake_llm):
    """Wrong key names / nesting / '8/10' strings must not turn into zero scores."""
    fake_llm(eval_style="messy")
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 1}), cfg)
    _answer(app, cfg, text=scored(8))
    entry = app.get_state(cfg).values["qa_history"][0]
    assert entry["criteria"]["technical_correctness"] == 8.0
    assert entry["criteria"]["explanation_quality"] == 8.0
    assert entry["score"] >= 8.0
    assert entry["factual_errors"] == []          # "None" is not an error
    assert entry["feedback"] == "Feedback for score 8."


def test_plain_text_question_fallback(fake_llm):
    fake_llm(plain_questions=True)
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 1}), cfg)
    pending = pending_interrupt(app, cfg)
    # One corrective retry, then the model's plain text is used as the question.
    assert pending["question"].startswith("Explain concept number")
    assert app.get_state(cfg).values["current_key_points"] == []


def test_non_answer_is_scored_without_llm(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 2}), cfg)
    _answer(app, cfg, text="Sorry, I don't know.")
    _answer(app, cfg, text="", skipped=True)
    entry = app.get_state(cfg).values["qa_history"][0]
    assert entry["score"] <= 1 and llm.calls["answer_eval"] == 0
    assert all(c["status"] == "missing" for c in entry["key_point_coverage"])


def test_non_answer_detection():
    assert ea.is_non_answer("I don't know")
    assert ea.is_non_answer("no idea really")
    assert ea.is_non_answer("um")
    assert not ea.is_non_answer("Not sure, but a tuple is immutable and a list is mutable")
    assert not ea.is_non_answer("Pass by reference means the function receives the object itself")


def test_deterministic_scoring_guardrails():
    ev = ea.AnswerEvaluation(
        key_point_coverage=[{"point": p, "status": "covered"} for p in KEY_POINTS],
        factual_errors=["claims tuples are mutable", "claims lists are faster"],
        technical_correctness=9, relevance=9, completeness=9, understanding=9,
        explanation_quality=9, feedback="x")
    crit, overall, cov = ea.compute_scores(ev, KEY_POINTS, ANSWER)
    assert cov == 10.0 and crit["technical_correctness"] == 5.0   # 2 errors cap at 5
    assert overall < 9

    off_topic = ea.AnswerEvaluation(technical_correctness=8, relevance=1, completeness=6,
                                    understanding=7, explanation_quality=9, feedback="x")
    _, overall, _ = ea.compute_scores(off_topic, [], ANSWER)
    assert overall <= 1.0

    short = ea.AnswerEvaluation(technical_correctness=9, relevance=9, completeness=9,
                                understanding=9, explanation_quality=9, feedback="x")
    crit, _, _ = ea.compute_scores(short, [], "Tuples are immutable.")
    assert crit["completeness"] <= 4 and crit["understanding"] <= 5

    with pytest.raises(Exception):  # missing criterion must NOT silently become 0
        ea.AnswerEvaluation(relevance=5, completeness=5, understanding=5,
                            explanation_quality=5, feedback="x")


def test_structured_output_path(fake_llm, monkeypatch):
    """Provider-native structured output is used when available."""
    llm = fake_llm()

    class Runnable:
        def __init__(self, cls):
            self.cls = cls

        def invoke(self, prompt):
            return self.cls(technical_correctness=7, relevance=7, completeness=7, understanding=7,
                            explanation_quality=7, feedback="structured",
                            key_point_coverage=[{"point": p, "status": "partial"} for p in KEY_POINTS])

    llm.with_structured_output = lambda cls, method=None: Runnable(cls)
    state = {"jd": "x", "current_question": "q", "current_key_points": KEY_POINTS, "qa_history": [],
             "pending_answer": {"answer": ANSWER, "skipped": False, "source": "text"}}
    entry = ea.answer_evaluation_agent(state)["qa_history"][0]
    assert entry["feedback"] == "structured" and llm.calls["answer_eval"] == 0
    assert entry["coverage_score"] == 5.0


def test_rejected_candidate_skips_interview(fake_llm):
    llm = fake_llm(decision="REJECTED")
    app = build_graph()
    cfg = _start(app)
    assert pending_interrupt(app, cfg) is None
    final = app.get_state(cfg).values
    assert final["final_report"]["recommendation"] == "No Hire"
    assert llm.calls["question"] == 0 and llm.calls["final"] == 0


def test_skip_answer(fake_llm):
    llm = fake_llm(qualities=["none", "strong"])   # a skip is easier, whatever the model says
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 2}), cfg)
    _answer(app, cfg, text="", skipped=True)
    state = app.get_state(cfg).values
    assert state["qa_history"][0]["skipped"] is True
    assert state["current_difficulty"] == "easy"        # skipped -> easier
    _answer(app, cfg, text="", skipped=True)
    final = app.get_state(cfg).values
    assert final["qa_history"][0]["score"] == 0.0
    assert llm.calls["answer_eval"] == 0                 # no LLM call for a skip


def test_background_failure_is_rescored_at_the_end(fake_llm):
    llm = fake_llm(fail_eval_once=True)
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 2}), cfg)
    _answer(app, cfg, text=scored(7))      # background scoring of this answer fails once
    _answer(app, cfg, text=scored(6))
    final = app.get_state(cfg).values
    assert [q["score"] for q in final["qa_history"]] != [None, None]
    assert 6 <= final["qa_history"][0]["score"] <= 7.5
    assert final["final_report"]["num_questions"] == 2


def test_llm_failure_then_retry(fake_llm):
    import time as _t
    llm = fake_llm(fail_eval_count=3)      # background attempt + both attempts at the end fail
    app = build_graph()
    cfg = _start(app)
    app.invoke(Command(resume={"max_questions": 2}), cfg)
    _answer(app, cfg, text=scored(7))
    for _ in range(100):                   # let the failing background attempt finish
        if getattr(llm, "eval_attempts", 0) >= 1:
            break
        _t.sleep(0.02)
    _t.sleep(0.1)
    with pytest.raises(llm_utils.LLMError):
        _answer(app, cfg, text=scored(6))
    snap = app.get_state(cfg)
    assert snap.next == ("evaluation",)    # stalled at the final step, answers kept
    app.invoke(None, cfg)                  # retry from checkpoint
    final = app.get_state(cfg).values
    assert 6 <= final["qa_history"][0]["score"] <= 7.5
    assert final["final_report"]["num_questions"] == 2


def test_restart_reuses_screening(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = _start(app)
    seed = {k: app.get_state(cfg).values[k] for k in
            ("resume", "jd", "parsed", "screening", "screening_decision", "match_score", "candidate_name")}
    cfg2 = thread_config(uuid.uuid4().hex)
    app.invoke(seed, cfg2)
    assert pending_interrupt(app, cfg2)["type"] == "start_interview"
    assert llm.calls["parse"] == 1 and llm.calls["screen"] == 1   # not reprocessed


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_helpers():
    assert next_difficulty("medium", 9) == "hard"
    assert next_difficulty("medium", 3) == "easy"
    assert next_difficulty("medium", 6) == "medium"
    assert next_difficulty("easy", None, skipped=True) == "easy"
    assert clean_question("Question 2: **What is a tuple?**") == "What is a tuple?"
    assert _parse_screening("Decision based on skills...\nMATCH SCORE: 40\nDECISION: NOT SELECTED") == ("REJECTED", 40)
    assert _parse_screening("Match score: 75") == ("SELECTED", 75)
    assert llm_utils.extract_json('text {"a": {"b": "}"}} more') == {"a": {"b": "}"}}
    assert llm_utils.normalize_key("Technical Correctness") == "technical_correctness"


# --------------------------------------------------------------------------- #
# Audio / voice layer
# --------------------------------------------------------------------------- #
def _wav(seconds=2.0, amplitude=8000, rate=16000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate))) * channels
            for i in range(int(seconds * rate))
        )
        wf.writeframes(frames)
    return buf.getvalue()


def test_audio_validation():
    with pytest.raises(voice.EmptyRecordingError):
        voice.validate_wav(b"")
    with pytest.raises(voice.EmptyRecordingError):
        voice.validate_wav(_wav(seconds=0.3))           # too short
    with pytest.raises(voice.EmptyRecordingError):
        voice.validate_wav(_wav(amplitude=0))           # silence
    with pytest.raises(voice.InvalidAudioError):
        voice.validate_wav(b"not a wav file" * 20)      # invalid format
    info = voice.validate_wav(_wav())
    assert 1.9 < info.duration_s < 2.1


def test_audio_resample_and_gain():
    # 48 kHz stereo browser recording -> 16 kHz mono
    samples, info = voice.prepare_audio(_wav(seconds=2.0, rate=48000, channels=2))
    assert abs(samples.size - 32000) <= 2 and info.sample_rate == 48000 and info.channels == 2
    # Quiet microphone is boosted, not rejected
    samples, info = voice.prepare_audio(_wav(amplitude=1500))
    assert info.gain > 5 and 0.85 <= float(abs(samples).max()) <= 0.91
    # Gain is capped (x20) so background hiss is not blown up into "speech"
    samples, info = voice.prepare_audio(_wav(amplitude=400))
    assert info.gain == 20.0


def _riff(fmt_tag, channels, rate, bits, payload, data_size=None, extensible=False):
    block = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * block, block, bits)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", fmt_tag) + b"\x00" * 14
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, channels, rate, rate * block, block, bits)
    size = len(payload) if data_size is None else data_size
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"LIST" + struct.pack("<I", 4) + b"INFO"           # extra chunk browsers add
            + b"data" + struct.pack("<I", size) + payload)
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + body


def test_browser_wav_variants():
    import numpy as np
    t = np.arange(32000) / 16000.0
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)
    variants = {
        "float32": _riff(3, 1, 16000, 32, sig.astype("<f4").tobytes()),
        "float32 extensible stereo 48k": _riff(
            3, 2, 48000, 32,
            np.repeat(np.interp(np.arange(96000) / 48000.0, t, sig), 2).astype("<f4").tobytes(),
            extensible=True),
        "pcm16 streaming size": _riff(1, 1, 16000, 16, (sig * 32767).astype("<i2").tobytes(),
                                      data_size=0xFFFFFFFF),
        "pcm24": _riff(1, 1, 16000, 24, b"".join(
            int(v * 8388607).to_bytes(3, "little", signed=True) for v in sig)),
    }
    for name, data in variants.items():
        samples, info = voice.prepare_audio(data)
        assert 1.9 < info.duration_s < 2.1, name
        assert 0.85 <= float(abs(samples).max()) <= 0.91, name   # decoded + normalised


def test_stt_failure_hallucination_and_echo(monkeypatch):
    monkeypatch.setattr(config, "STT_PROVIDER", "faster_whisper")
    monkeypatch.setattr(config, "STT_FALLBACK_PROVIDER", "")

    def boom(samples, prompt=None):
        raise speech_to_text.STTError("Speech recognition failed.")
    monkeypatch.setattr(speech_to_text, "_transcribe_local", boom)
    monkeypatch.setitem(speech_to_text._PROVIDERS, "faster_whisper", boom)
    with pytest.raises(voice.STTError):
        voice.transcribe(_wav())

    for phantom in (" . ", "Thank you.", "you"):
        monkeypatch.setitem(speech_to_text._PROVIDERS, "faster_whisper", lambda s, p=None, t=phantom: (t, 0.9))
        with pytest.raises(voice.SpeechNotRecognizedError):
            voice.transcribe(_wav())

    seen = {}

    def ok(samples, prompt=None):
        seen["prompt"] = prompt
        return "A tuple is immutable.", 0.4
    monkeypatch.setitem(speech_to_text._PROVIDERS, "faster_whisper", ok)
    res = voice.transcribe_detailed(_wav(), question="list vs tuple?", topic="Python")
    assert res.text == "A tuple is immutable." and res.low_confidence is True
    assert "list vs tuple?" in seen["prompt"]          # question given as context


def test_stt_fallback_provider(monkeypatch):
    monkeypatch.setattr(config, "STT_PROVIDER", "faster_whisper")
    monkeypatch.setattr(config, "STT_FALLBACK_PROVIDER", "openai")

    def setup_fail(samples, prompt=None):
        raise speech_to_text.STTSetupError("model download blocked")
    monkeypatch.setitem(speech_to_text._PROVIDERS, "faster_whisper", setup_fail)
    monkeypatch.setitem(speech_to_text._PROVIDERS, "openai", lambda s, p=None: ("Lists are mutable.", 0.9))
    res = voice.transcribe_detailed(_wav())
    assert res.text == "Lists are mutable." and res.provider == "openai"


def test_whisper_retries_without_vad(monkeypatch):
    """If the voice-activity filter removes everything, retry without it."""
    calls = []

    def fake_run(model, samples, prompt, use_vad):
        calls.append(use_vad)
        return ("", 0.0) if use_vad else ("Lists are mutable and tuples are not.", 0.8)
    monkeypatch.setattr(speech_to_text, "_load_whisper", lambda: object())
    monkeypatch.setattr(speech_to_text, "_run_whisper", fake_run)
    text, conf = speech_to_text._transcribe_local(voice.prepare_audio(_wav())[0], prompt="ctx")
    assert text.startswith("Lists") and calls == [True, False]


def test_setup_error_messages():
    assert "Visual C++" in speech_to_text._setup_error_message(ImportError("DLL load failed while importing"))
    assert "huggingface" in speech_to_text._setup_error_message(OSError("Failed to resolve huggingface.co"))


def test_tts_fallback_and_failure(monkeypatch):
    monkeypatch.setattr(config, "TTS_PROVIDER", "edge")
    monkeypatch.setattr(config, "TTS_FALLBACK_TO_GTTS", True)

    def fail(_):
        raise ConnectionError("offline")

    providers = dict(text_to_speech._PROVIDERS)
    providers["edge"] = fail
    providers["gtts"] = lambda _: b"\xff\xfb" * 200
    monkeypatch.setattr(text_to_speech, "_PROVIDERS", providers)
    audio, mime = voice.synthesize("What is a list?")
    assert mime == "audio/mp3" and len(audio) == 400   # fell back to gTTS

    providers["gtts"] = fail
    with pytest.raises(voice.TTSError):
        voice.synthesize("What is a list?")
