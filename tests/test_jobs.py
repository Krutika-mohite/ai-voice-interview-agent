"""
Graph steps run as background jobs so page re-runs (e.g. a proctoring camera warning
while the candidate waits) can never interrupt them. Run:

    python -m pytest -q tests/test_jobs.py
"""

import os
import sys
import threading
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import llm_utils  # noqa: E402
from langgraph.types import Command  # noqa: E402

from graph import build_graph, pending_interrupt, run_step, thread_config  # noqa: E402
from jobs import JobRunner  # noqa: E402
from tests.test_voice_interview import ANSWER, FakeLLM  # noqa: E402


@pytest.fixture
def fake_llm(monkeypatch):
    def _make(**kw):
        llm = FakeLLM(**kw)
        monkeypatch.setattr(config, "llm", llm)
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
        monkeypatch.setattr(llm_utils.time, "sleep", lambda *_: None)
        return llm
    return _make


def _msg(exc):
    return f"error: {exc}"


def test_job_runner_basics():
    runner = JobRunner(max_workers=2)
    gate = threading.Event()
    job = runner.start("t1", gate.wait, "Working")
    assert runner.running("t1")
    assert runner.start("t1", lambda: None, "Other") is job        # one job per interview
    assert runner.collect("t1", _msg) == (False, None)              # still running
    gate.set(); runner.wait("t1")
    assert runner.collect("t1", _msg) == (True, None) and runner.get("t1") is None
    runner.start("t2", lambda: 1 / 0, "Boom"); runner.wait("t2")
    finished, err = runner.collect("t2", _msg)
    assert finished and err.startswith("error:")


def test_question_arrives_even_if_the_page_stops_waiting(fake_llm):
    """The reported bug: a page re-run during question generation killed the step."""
    llm = fake_llm()
    slow = threading.Event()
    original = llm.invoke

    def slow_invoke(prompt):
        if "senior technical interviewer running" in prompt and llm.calls["question"] >= 1:
            slow.wait(2)                       # question 2 takes a while
        return original(prompt)
    llm.invoke = slow_invoke

    app = build_graph()
    cfg = thread_config(uuid.uuid4().hex)
    app.invoke({"resume": "John Doe", "jd": "Python developer"}, cfg)
    app.invoke(Command(resume={"max_questions": 3}), cfg)

    runner = JobRunner()
    tid = cfg["configurable"]["thread_id"]
    runner.start(tid, lambda: run_step(app, cfg, Command(resume={
        "answer": ANSWER, "skipped": False, "source": "voice"})), "Preparing the next question...")
    # ... the page is re-run here (camera warning): nobody is waiting for the job any more
    time.sleep(0.2)
    assert runner.running(tid)
    slow.set()
    runner.wait(tid, timeout=10)
    assert runner.collect(tid, _msg) == (True, None)
    assert pending_interrupt(app, cfg)["number"] == 2              # next question is ready


def test_a_failed_step_is_retried_automatically(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = thread_config(uuid.uuid4().hex)
    app.invoke({"resume": "John Doe", "jd": "Python developer"}, cfg)
    app.invoke(Command(resume={"max_questions": 3}), cfg)
    llm.fail_question_once = True                                  # first attempt at Q2 fails
    run_step(app, cfg, Command(resume={"answer": ANSWER, "skipped": False, "source": "voice"}))
    assert pending_interrupt(app, cfg)["number"] == 2              # no "did not finish" screen


def test_a_persistent_failure_is_reported(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = thread_config(uuid.uuid4().hex)
    app.invoke({"resume": "John Doe", "jd": "Python developer"}, cfg)
    app.invoke(Command(resume={"max_questions": 3}), cfg)
    original = llm.invoke

    def always_fail(prompt):
        if "senior technical interviewer running" in prompt:
            raise ConnectionError("connection refused")
        return original(prompt)
    llm.invoke = always_fail
    with pytest.raises(llm_utils.LLMError):
        run_step(app, cfg, Command(resume={"answer": ANSWER, "skipped": False, "source": "voice"}))
    assert app.get_state(cfg).next == ("interview",)               # answer kept; Retry works
