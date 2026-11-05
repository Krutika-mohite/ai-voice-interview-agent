"""
Tests for proctoring: server-side violation rules and ending an interview through
LangGraph. Fake LLM, no browser needed. Run from the project root:

    python -m pytest -q tests/test_proctoring.py

The browser rules engine has its own tests:  node tests/test_proctor_rules.js
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import llm_utils  # noqa: E402
from langgraph.types import Command  # noqa: E402

from graph import build_graph, pending_interrupt, terminate_interview, thread_config  # noqa: E402
from proctoring import ProctorSession  # noqa: E402
from tests.test_voice_interview import FakeLLM  # noqa: E402

ANSWER = ("A list is mutable so items can be changed, while a tuple is immutable and "
          "therefore hashable, which lets it be used as a dictionary key.")


@pytest.fixture
def fake_llm(monkeypatch):
    def _make(**kw):
        llm = FakeLLM(**kw)
        monkeypatch.setattr(config, "llm", llm)
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
        monkeypatch.setattr(llm_utils.time, "sleep", lambda *_: None)
        return llm
    return _make


def _ev(ev_type, fatal=False, label=None):
    return {"id": uuid.uuid4().hex[:8], "type": ev_type, "label": label or ev_type, "fatal": fatal,
            "at": "2026-11-03T10:00:00+00:00"}


# --------------------------------------------------------------------------- #
# ProctorSession (server-side decisions)
# --------------------------------------------------------------------------- #
def test_three_camera_warnings_end_the_interview():
    s = ProctorSession("t1", max_violations=3); s.arm()
    events = []
    events.append(_ev("no_face", label="Face not visible"))
    r = s.process({"nonce": "t1", "events": list(events)}, question=1)
    assert r == {"warnings": ["Face not visible"], "terminate": None} and s.violations == 1
    # the same cumulative list again: nothing new is counted
    assert s.process({"nonce": "t1", "events": list(events)}, 1)["warnings"] == []
    events.append(_ev("looking_away", label="Looking away from the screen"))
    assert s.process({"nonce": "t1", "events": list(events)}, 2)["terminate"] is None
    events.append(_ev("multiple_faces", label="More than one person in view"))
    r = s.process({"nonce": "t1", "events": list(events)}, 2)
    assert r["terminate"].startswith("3 proctoring warnings") and not s.armed
    assert [i["counted"] for i in s.incidents] == [True, True, True]
    # after the end, further events are ignored
    events.append(_ev("no_face"))
    assert s.process({"nonce": "t1", "events": events}, 3) == {"warnings": [], "terminate": None}


def test_switching_tab_or_leaving_fullscreen_ends_immediately():
    for kind in ("tab_switch", "left_fullscreen", "window_switch"):
        s = ProctorSession("t1"); s.arm()
        r = s.process({"nonce": "t1", "events": [_ev(kind, fatal=True, label="Left fullscreen mode")]}, 1)
        assert r["terminate"] == "Left fullscreen mode" and s.incidents[0]["fatal"] is True


def test_non_counted_events_stale_values_and_unarmed_sessions():
    s = ProctorSession("t1"); s.arm()
    r = s.process({"nonce": "t1", "events": [_ev("monitor_error", label="Face detection could not be loaded")]}, 1)
    assert r == {"warnings": [], "terminate": None} and s.violations == 0 and len(s.incidents) == 1
    # value left over from a previous interview (different thread) is ignored
    assert s.process({"nonce": "old", "events": [_ev("no_face")]}, 1)["warnings"] == []
    unarmed = ProctorSession("t2")
    assert unarmed.process({"nonce": "t2", "events": [_ev("tab_switch", fatal=True)]}, 1)["terminate"] is None


def test_nothing_counts_during_a_technical_pause():
    s = ProctorSession("t1"); s.arm()
    events = [_ev("window_switch", fatal=True, label="Switched to another window or app"),
              _ev("no_face", label="Face not visible")]
    r = s.process({"nonce": "t1", "events": events}, 2, paused=True)
    assert r == {"warnings": [], "terminate": None} and s.violations == 0 and s.armed
    assert all("ignored: technical problem" in i["label"] for i in s.incidents)
    # once resumed, the same events are not counted again; new ones count normally
    assert s.process({"nonce": "t1", "events": events}, 2)["terminate"] is None
    r = s.process({"nonce": "t1", "events": events + [_ev("no_face", label="Face not visible")]}, 2)
    assert r["warnings"] == ["Face not visible"] and s.violations == 1


def test_page_reload_is_fatal():
    s = ProctorSession("t1"); s.arm()
    r = s.page_reloaded(question=2)
    assert "reloaded" in r["terminate"] and s.incidents[-1]["type"] == "page_reload"


# --------------------------------------------------------------------------- #
# Ending the interview through LangGraph
# --------------------------------------------------------------------------- #
def _interview(app, n=3):
    cfg = thread_config(uuid.uuid4().hex)
    app.invoke({"resume": "John Doe resume text", "jd": "Python developer with ML"}, cfg)
    app.invoke(Command(resume={"max_questions": n}), cfg)
    return cfg


def test_terminate_while_waiting_for_an_answer(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = _interview(app)
    s = ProctorSession(cfg["configurable"]["thread_id"]); s.arm()
    s.process({"events": [_ev("no_face", label="Face not visible")]}, 1)
    app.invoke(Command(resume={"answer": ANSWER + " [s8]", "skipped": False, "source": "voice",
                               "incidents": list(s.incidents)}), cfg)          # question 1 answered
    r = s.process({"events": [_ev("tab_switch", fatal=True, label="Switched to another tab")]}, 2)
    terminate_interview(app, cfg, r["terminate"], s.incidents)

    assert pending_interrupt(app, cfg) is None
    st = app.get_state(cfg).values
    rep = st["final_report"]
    assert st["terminated"] and st["interview_status"] == "terminated"
    assert rep["terminated"] and rep["recommendation"] == "No Hire"
    assert rep["num_questions"] == 1 and rep["average_score"] >= 8
    assert "1 of 3 questions" in rep["justification"]
    assert [i["type"] for i in rep["incidents"]] == ["no_face", "tab_switch"]
    assert "INTERVIEW ENDED EARLY: Switched to another tab" in st["final"]
    assert "[ENDED INTERVIEW] Switched to another tab" in st["final"]
    assert llm.calls["final"] == 0          # no LLM needed to end an interview


def test_terminate_when_a_step_has_stalled(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = _interview(app)
    llm.fail_question_once = True                                  # next question fails
    with pytest.raises(llm_utils.LLMError):
        app.invoke(Command(resume={"answer": ANSWER, "skipped": False, "source": "voice"}), cfg)
    assert app.get_state(cfg).next == ("interview",)               # stalled
    terminate_interview(app, cfg, "Left fullscreen mode",
                        [{"type": "left_fullscreen", "label": "Left fullscreen mode", "fatal": True}])
    st = app.get_state(cfg).values
    assert st["final_report"]["terminated"] and st["final_report"]["recommendation"] == "No Hire"
    assert app.get_state(cfg).next == ()


def test_terminated_report_survives_a_model_outage(fake_llm):
    llm = fake_llm()
    app = build_graph()
    cfg = _interview(app)
    llm.fail_eval_count = 99                                       # grader completely down
    app.invoke(Command(resume={"answer": ANSWER, "skipped": False, "source": "voice"}), cfg)
    terminate_interview(app, cfg, "Left fullscreen mode", [])
    rep = app.get_state(cfg).values["final_report"]
    assert rep["terminated"] and rep["question_scores"][0]["score"] is None
    assert "not scored" in app.get_state(cfg).values["final"]


def test_warnings_reach_a_normal_report(fake_llm):
    fake_llm(answer_scores=(9,))
    app = build_graph()
    cfg = _interview(app, n=1)
    incidents = [{"type": "looking_away", "label": "Looking away from the screen", "counted": True,
                  "fatal": False, "question": 1, "at": "2026-11-03T10:00:00+00:00"}]
    app.invoke(Command(resume={"answer": ANSWER, "skipped": False, "source": "voice",
                               "incidents": incidents}), cfg)
    st = app.get_state(cfg).values
    assert not st.get("terminated")
    assert st["final_report"]["incidents"] == incidents
    assert "[warning] Looking away from the screen (question 1)" in st["final"]


def test_reports_without_proctoring_say_no_incidents(fake_llm):
    fake_llm(answer_scores=(7,))
    app = build_graph()
    cfg = _interview(app, n=1)
    app.invoke(Command(resume={"answer": ANSWER, "skipped": False, "source": "text"}), cfg)
    assert "No incidents" in app.get_state(cfg).values["final"]
