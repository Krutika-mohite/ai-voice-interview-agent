"""
AI Voice Interview - Streamlit UI.

    streamlit run app.py

This is a thin voice/UI layer around the existing LangGraph graph (graph.py):
  - The graph owns all interview logic and state (screening, adaptive questions,
    answer evaluation, final report). The UI never generates or scores anything itself.
  - The graph pauses with interrupt() when it needs the candidate; the UI resumes it
    with Command(resume=...).
  - Voice: question text -> TTS -> browser audio; microphone -> STT -> answer text.
"""

import logging
import re
import time
import uuid
from typing import Optional

import streamlit as st

st.set_page_config(page_title="Interview room", page_icon="🎙️", layout="centered")


def _load_secrets_into_env() -> None:
    """On Streamlit Community Cloud, settings live in the app's Secrets, not in .env.

    Copy simple top-level secrets into the environment (without overriding anything
    already set) so config.py works the same locally and in the cloud.
    """
    import os

    try:
        items = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file (normal when running locally)
        return
    for key, value in items.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value).lower() if isinstance(value, bool) else str(value)


_load_secrets_into_env()

try:
    from langgraph.types import Command

    import config
    import jobs
    import proctoring
    import voice
    from graph import build_graph, pending_interrupt, run_step, terminate_interview, thread_config
    from llm_utils import LLMError
    from resume_loader import load_resume
    import ui_theme as ui
except Exception as exc:  # configuration problems (e.g. missing API key) - show, don't crash
    st.error(f"The application could not start: {exc}")
    st.stop()

logger = logging.getLogger("app")
ss = st.session_state


def _outdated_files() -> list:
    """Project files left over from an older version (a partly-replaced folder)."""
    import inspect

    checks = {
        "config.py": lambda: hasattr(config, "LLM_BASE_URL") and hasattr(config, "APP_PASSWORD"),
        "llm_utils.py": lambda: hasattr(__import__("llm_utils"), "foreground"),
        "graph.py": lambda: hasattr(__import__("graph"), "run_step"),
        "jobs.py": lambda: hasattr(jobs, "JobRunner"),
        "interview_agent.py": lambda: hasattr(__import__("interview_agent"), "record_answer"),
        "evaluation_agent.py": lambda: hasattr(__import__("evaluation_agent"), "collect_scores"),
        "state.py": lambda: "proctoring_incidents" in inspect.getsource(__import__("state")),
        "proctoring.py": lambda: "camera_paused" in inspect.signature(proctoring.render_monitor).parameters,
        "ui_theme.py": lambda: hasattr(ui, "incident_rows"),
        "voice/speech_to_text.py": lambda: hasattr(__import__("voice.speech_to_text", fromlist=["x"]), "_transcribe_groq"),
        "proctor_component/index.html": lambda: "openOwnTab" in (proctoring.COMPONENT_DIR / "index.html").read_text(encoding="utf-8"),
    }
    old = []
    for name, ok in checks.items():
        try:
            if not ok():
                old.append(name)
        except Exception:  # noqa: BLE001
            old.append(name)
    return old


_OLD = _outdated_files()
if _OLD:
    st.error(
        "Some project files are from an older version, so the app cannot run safely: "
        + ", ".join(_OLD)
        + ". Extract the latest zip into a NEW empty folder, copy your .env file into it, "
        "and start the app from there."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Shared resources (one per server process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_app():
    # The compiled graph holds an in-memory checkpointer: interview state for every
    # session lives there, keyed by thread_id (survives browser refreshes).
    return build_graph()


@st.cache_resource(show_spinner=False)
def get_proctor_registry() -> dict:
    # thread_id -> {"owner": browser-session id, "session": ProctorSession}.
    # Server-wide, so a reload (new browser session) of a proctored interview is detected
    # and the full incident log survives it.
    return {}


@st.cache_resource(show_spinner=False)
def stt_warmup() -> dict:
    """Load the speech model in a background thread as soon as the app starts, so the
    interview never waits for it. The dict is updated when loading finishes."""
    import threading

    status = {"done": False, "ok": None, "message": ""}

    def _load():
        ok, msg = voice.preload_model()
        status.update(done=True, ok=ok, message=msg)

    threading.Thread(target=_load, name="stt-warmup", daemon=True).start()
    return status


@st.cache_resource(show_spinner=False)
def get_job_runner():
    # Graph steps run here, outside the page script, so page re-runs never interrupt them.
    return jobs.JobRunner()


APP = get_app()
REG = get_proctor_registry()
JOBS = get_job_runner()
STT_WARMUP = stt_warmup()

VOICE_TEST_SENTENCE = (
    "A tuple in Python is immutable, while a list is mutable and can be changed after creation."
)


def _word_accuracy(expected: str, heard: str) -> float:
    """Share of expected words recognised in order (simple, dependency-free WER proxy)."""
    import difflib

    a = re.findall(r"[a-z0-9]+", expected.lower())
    b = re.findall(r"[a-z0-9]+", heard.lower())
    if not a:
        return 0.0
    matched = sum(block.size for block in difflib.SequenceMatcher(None, a, b).get_matching_blocks())
    return matched / len(a)


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
_UUID_HEX = re.compile(r"^[0-9a-f]{32}$")


def _thread_exists(thread_id: str) -> bool:
    try:
        return bool(APP.get_state(thread_config(thread_id)).values)
    except Exception:  # noqa: BLE001
        return False


def _init_session() -> None:
    if "client_id" not in ss:
        ss.client_id = uuid.uuid4().hex   # identifies this browser session (reload detection)
    if "thread_id" not in ss:
        qp = st.query_params.get("session")
        ss.thread_id = qp if qp and _UUID_HEX.match(qp) and _thread_exists(qp) else None
    ss.setdefault("tts_cache", {})      # question key -> {"audio": bytes} | {"error": str}
    ss.setdefault("stt", {})            # question key -> {"fp", "text", "error"}
    ss.setdefault("rec_attempt", {})    # question key -> int (resets the recorder widget)
    ss.setdefault("autoplayed", set())  # question keys already auto-played
    ss.setdefault("proctor_setup", False)
    ss.setdefault("planned_questions", config.MAX_INTERVIEW_QUESTIONS)
    ss.setdefault("last_error", None)
    ss.setdefault("voice_test", None)   # result of the sidebar voice self-test


def _reset_voice_state() -> None:
    ss.tts_cache, ss.stt, ss.rec_attempt = {}, {}, {}
    ss.autoplayed = set()
    ss.proctor_setup = False
    ss.last_error = None
    for key in [k for k in ss.keys() if str(k).startswith("ans_")]:
        del ss[key]


def _delete_thread(thread_id) -> None:
    if not thread_id:
        return
    try:
        APP.checkpointer.delete_thread(thread_id)  # free candidate data held in memory
    except Exception:  # noqa: BLE001 - older langgraph versions may not support it
        pass


def _new_thread() -> str:
    thread_id = uuid.uuid4().hex
    ss.thread_id = thread_id
    st.query_params["session"] = thread_id  # lets a browser refresh resume the interview
    return thread_id


def _cfg() -> dict:
    return thread_config(ss.thread_id)


def _job_error_message(exc: BaseException) -> str:
    if isinstance(exc, LLMError):
        return str(exc)
    logger.error("Graph execution failed", exc_info=exc)
    return f"Something went wrong while processing ({type(exc).__name__}). Please retry."


def _graph_step(thread_id: str, payload) -> None:
    """One graph step, run in a worker thread. A failed step is retried once automatically."""
    run_step(APP, thread_config(thread_id), payload)


def wait_for_job(label: Optional[str] = None) -> bool:
    """Wait (with a spinner) for this interview's background job. True if it succeeded.

    If the page is re-run meanwhile, the job keeps going and the next run waits again.
    """
    job = JOBS.get(ss.thread_id)
    if job is None:
        return True
    if not job.future.done():
        with st.spinner(label or job.label):
            while not job.future.done():
                time.sleep(0.25)
    finished, error = JOBS.collect(ss.thread_id, _job_error_message)
    if finished:
        ss.last_error = error
    return error is None


def run_graph(payload, spinner: str) -> bool:
    """Run one graph step in the background and wait for it. Errors are stored for display."""
    tid = ss.thread_id
    JOBS.start(tid, lambda: _graph_step(tid, payload), spinner)
    return wait_for_job(spinner)


def current_phase():
    """Derive the UI phase from the LangGraph checkpoint (single source of truth)."""
    if not ss.thread_id:
        return "setup", None, {}
    snapshot = APP.get_state(_cfg())
    values = snapshot.values or {}
    if not values:
        return "setup", None, {}
    pending = pending_interrupt(APP, _cfg())
    if pending:
        return pending.get("type", "answer"), pending, values
    if snapshot.next:
        return "stalled", None, values   # a step failed (e.g. LLM offline) - can be retried
    if values.get("final"):
        return "report", None, values
    return "setup", None, values


# --------------------------------------------------------------------------- #
# Proctoring helpers
# --------------------------------------------------------------------------- #
def active_proctor():
    """ProctorSession for this interview if this browser session owns it, else None."""
    entry = REG.get(ss.thread_id) if ss.thread_id else None
    return entry["session"] if entry and entry["owner"] == ss.client_id else None


def end_proctored_interview(session, reason: str) -> None:
    """Stop the interview through LangGraph (so a report is produced) and rerun."""
    tid, incidents = ss.thread_id, list(session.incidents)
    REG.pop(tid, None)
    wait_for_job("Finishing the current step...")   # never touch the graph while a step runs
    JOBS.start(tid, lambda: terminate_interview(APP, thread_config(tid), reason, incidents),
               "Ending the interview...")
    if not wait_for_job("Ending the interview..."):
        ss.last_error = "The interview was ended, but the report could not be created. Please retry."
    st.rerun()


def check_page_reload(phase: str, values: dict) -> None:
    """A proctored interview opened in a new browser session (reload / new tab) ends it."""
    entry = REG.get(ss.thread_id) if ss.thread_id else None
    if entry and entry["owner"] != ss.client_id and phase in {"answer", "stalled"}:
        session = entry["session"]
        session.page_reloaded(values.get("question_count"))
        entry["owner"] = ss.client_id
        end_proctored_interview(session, session.ended_reason)


def proctor_guard(values: dict, paused: bool = False, camera_paused: bool = False) -> None:
    """Render the live proctoring strip and apply any new violations.

    paused=True while the app itself has a problem: nothing counts against the candidate.
    camera_paused=True while the candidate waits for the AI: camera warnings are off,
    fullscreen / tab / window rules stay on.
    """
    session = active_proctor()
    if session is None:
        return
    value = proctoring.render_monitor(session, paused=paused, camera_paused=camera_paused)
    result = session.process(value, values.get("question_count"), paused=paused)
    if result["terminate"]:
        end_proctored_interview(session, result["terminate"])
    for label in result["warnings"]:
        st.toast(f"Warning {session.violations} of {session.max_violations}: {label}", icon="⚠️")
    if session.last_warning:
        left = session.max_violations - session.violations
        ui.note(f"Warning {session.violations} of {session.max_violations}: {session.last_warning}. "
                f"{left} more {'warning ends' if left == 1 else 'warnings end'} the interview.", "warn")


def begin_interview(n: int) -> None:
    """Start question 1 (after the proctoring check when it is enabled).

    Speech recognition is already loading in the background (stt_warmup), so this
    only waits for the first question.
    """
    if config.PROCTORING_ENABLED:
        session = proctoring.ProctorSession(ss.thread_id)
        session.arm()
        REG[ss.thread_id] = {"owner": ss.client_id, "session": session}
    ss.proctor_setup = False
    tid = ss.thread_id
    JOBS.start(tid, lambda: _graph_step(tid, Command(resume={"max_questions": n})),
               "Preparing the first question...")
    st.rerun()   # the "working" screen waits for it, with camera warnings paused


def render_working(values: dict) -> None:
    """A graph step is running in the background: keep proctoring live and wait for it."""
    job = JOBS.get(ss.thread_id)
    proctor_guard(values, camera_paused=True)
    if job is not None and not job.future.done():
        ui.note("Please wait. Stay in fullscreen; camera warnings are paused while the AI works.")
    wait_for_job()
    st.rerun()


def exit_fullscreen() -> None:
    """Leave browser fullscreen once the interview is over."""
    if config.PROCTORING_ENABLED and ss.thread_id:
        try:
            proctoring.render_exit(ss.thread_id)
        except Exception:  # noqa: BLE001 - never block the report over this
            logger.warning("Could not leave fullscreen automatically")


# --------------------------------------------------------------------------- #
# UI pieces
# --------------------------------------------------------------------------- #
def header() -> None:
    ui.inject_theme()
    name = ""
    if ss.get("thread_id"):
        try:
            name = (APP.get_state(_cfg()).values or {}).get("candidate_name", "")
        except Exception:  # noqa: BLE001
            name = ""
    ui.header(f"Interviewing {name}" if name and name != "Candidate" else "AI technical interview")


def sidebar(phase: str, values: dict) -> None:
    with st.sidebar:
        st.subheader("Session")
        st.caption(
            f"Language model: {config.LLM_PROVIDER}. Speech recognition: {config.STT_PROVIDER}. "
            f"Voice: {config.TTS_PROVIDER}."
        )
        if phase == "working":
            st.info("The AI is working on the next step. Session controls are available again in a moment.")
            return
        if active_proctor() is not None and phase in {"answer", "stalled"}:
            st.info("Proctored interview in progress. Session controls are locked until it ends.")
            return
        can_restart = phase in {"answer", "report", "stalled"} and values.get("parsed") \
            and values.get("screening_decision") == "SELECTED" and not values.get("terminated")
        if can_restart and st.button("Restart interview", width="stretch",
                                     help="Same candidate and screening, fresh questions"):
            seed = {k: values.get(k) for k in (
                "resume", "jd", "parsed", "screening", "screening_decision",
                "match_score", "candidate_name")}
            old = ss.thread_id
            _reset_voice_state()
            _new_thread()
            _delete_thread(old)
            REG.pop(old, None)
            # route_start() skips parsing/screening because they are already present.
            run_graph(seed, "Restarting the interview...")
            st.rerun()
        if phase != "setup" and st.button("New candidate", width="stretch"):
            REG.pop(ss.thread_id, None)
            _delete_thread(ss.thread_id)
            _reset_voice_state()
            ss.thread_id = None
            st.query_params.clear()
            st.rerun()
        st.divider()
        st.subheader("Voice check")
        st.caption("Plays a known sentence through speech recognition and measures how "
                   "accurately it is recognised. No microphone needed.")
        if st.button("Test speech recognition", width="stretch"):
            with st.spinner("Testing the voice pipeline (the first run loads the speech model)..."):
                ss.voice_test = _run_voice_test()
        if ss.voice_test:
            kind, msg = ss.voice_test
            getattr(st, kind)(msg)
        st.divider()
        st.caption(
            "Recordings are processed in memory and never saved to disk. "
            "Interview data is kept only in this server's memory for the session."
        )


def _run_voice_test():
    """TTS -> STT round trip. Returns (streamlit_method, message)."""
    try:
        audio, _ = voice.synthesize(VOICE_TEST_SENTENCE)
    except voice.TTSError as exc:
        return "warning", f"Text-to-speech failed, so the test could not run: {exc.user_message}"
    try:
        result = voice.transcribe_detailed(audio, question="list vs tuple in Python")
    except voice.InvalidAudioError:
        return "warning", ("The test sound (MP3) could not be decoded on this computer - see the terminal "
                           "for details. This does NOT affect microphone answers, which are recorded as WAV. "
                           "Please test by recording an answer.")
    except voice.VoiceError as exc:
        return "error", f"Speech recognition failed: {exc.user_message}"
    except Exception:  # noqa: BLE001
        logger.exception("Voice self-test failed")
        return "error", "Speech recognition failed with an unexpected error. See the terminal window."
    acc = _word_accuracy(VOICE_TEST_SENTENCE, result.text)
    msg = (f"Heard: \"{result.text}\"\n\nWord accuracy {acc:.0%}, confidence "
           f"{result.confidence:.0%}, model {config.WHISPER_MODEL_PATH or config.WHISPER_MODEL_SIZE}.")
    return ("success" if acc >= 0.9 else "warning"), msg


def show_error_banner() -> None:
    if ss.last_error:
        st.error(ss.last_error)


def render_setup() -> None:
    ui.intro()
    resume_file = st.file_uploader("Candidate resume (PDF or TXT)", type=["pdf", "txt"])
    jd = st.text_area("Job description", height=170,
                      placeholder="Paste the role, required skills and experience. For example: "
                                  "Python developer with 3+ years in ML, LangChain and REST APIs.")
    show_error_banner()
    if st.button("Screen resume", type="primary", width="stretch"):
        if not resume_file:
            st.warning("Upload the candidate's resume to continue.")
            return
        if not jd.strip():
            st.warning("Paste the job description to continue.")
            return
        resume_text = load_resume(resume_file)
        if not resume_text.strip():
            st.error("No text could be read from that resume. If it is a scanned PDF, upload a "
                     "text-based PDF or a .txt file instead.")
            return
        _reset_voice_state()
        _new_thread()
        run_graph({"resume": resume_text, "jd": jd.strip()}, "Reading the resume and comparing it with the role...")
        st.rerun()


def render_proctor_setup() -> None:
    """Camera check, then the component puts the page in fullscreen and reports ready."""
    ui.section_title("Before you start")
    value = proctoring.render_setup(ss.thread_id)
    if isinstance(value, dict) and value.get("ready") and value.get("nonce") == ss.thread_id:
        begin_interview(int(ss.planned_questions))
    show_error_banner()
    if st.button("Back", width="content"):
        ss.proctor_setup = False
        st.rerun()


def render_start(values: dict) -> None:
    if ss.proctor_setup:
        render_proctor_setup()
        return
    name = values.get("candidate_name", "Candidate")
    score = values.get("match_score")
    ui.verdict("Shortlisted", f"{name} matches the role well enough for a technical interview.")
    ui.stats([("Resume match", f"{score}/100" if score is not None else "n/a"),
              ("Screening decision", values.get("screening_decision", "-").title())])
    with st.expander("Why this candidate was shortlisted"):
        st.write(values.get("screening", ""))

    ui.section_title("Set up the interview")
    n = st.slider(
        "Number of questions",
        min_value=config.INTERVIEW_QUESTIONS_MIN,
        max_value=config.INTERVIEW_QUESTIONS_MAX,
        value=min(max(config.MAX_INTERVIEW_QUESTIONS, config.INTERVIEW_QUESTIONS_MIN),
                  config.INTERVIEW_QUESTIONS_MAX),
    )
    ui.note("The interviewer speaks each question aloud and you answer with your microphone. "
            "Your browser asks for microphone access the first time. Headphones work best, and the "
            "page must be opened on localhost or an https address.")
    if config.PROCTORING_ENABLED:
        ui.note("This interview is proctored. Your camera checks that you stay in view, and the "
                "interview runs in fullscreen. Leaving fullscreen or switching tab or window ends it "
                f"immediately; warning {config.PROCTOR_MAX_VIOLATIONS} from the camera also ends it.", "warn")
    show_error_banner()
    label = "Continue to camera check" if config.PROCTORING_ENABLED else "Start interview"
    if st.button(label, type="primary", width="stretch"):
        ss.planned_questions = n
        if config.PROCTORING_ENABLED:
            ss.proctor_setup = True
            st.rerun()
        begin_interview(n)


def _question_audio(qkey: str, question: str) -> None:
    # Generate once per question; keep only the current question's audio in memory.
    if qkey not in ss.tts_cache:
        ss.tts_cache = {}
        try:
            with st.spinner("Preparing the interviewer's voice..."):
                audio, _mime = voice.synthesize(question)
            ss.tts_cache[qkey] = {"audio": audio}
        except voice.TTSError as exc:
            ss.tts_cache[qkey] = {"error": exc.user_message}

    item = ss.tts_cache[qkey]
    if "audio" in item:
        autoplay = config.AUTOPLAY_QUESTIONS and qkey not in ss.autoplayed
        ss.autoplayed.add(qkey)
        st.audio(item["audio"], format="audio/mp3", autoplay=autoplay)
    else:
        st.warning(item["error"])
        if st.button("Retry audio"):
            ss.tts_cache.pop(qkey, None)
            st.rerun()


def _handle_recording(qkey: str, audio_file, question: str, topic: str) -> None:
    """Transcribe a new recording once (Streamlit reruns must not re-transcribe)."""
    audio_bytes = audio_file.getvalue()
    fp = voice.audio_fingerprint(audio_bytes)
    current = ss.stt.get(qkey, {})
    if current.get("fp") == fp:
        return
    try:
        with st.spinner("Transcribing your answer..."):
            result = voice.transcribe_detailed(audio_bytes, question=question, topic=topic)
        ss.stt[qkey] = {"fp": fp, "text": result.text, "error": None,
                        "confidence": result.confidence, "low": result.low_confidence}
        ss[f"ans_{qkey}"] = result.text  # pre-fill the answer box
    except voice.VoiceError as exc:
        ss.stt[qkey] = {"fp": fp, "text": "", "error": exc.user_message,
                        "retryable": isinstance(exc, voice.STTError)}
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected STT failure")
        ss.stt[qkey] = {"fp": fp, "text": "", "error": "Transcription failed. Please try again.",
                        "retryable": True}


def render_question(pending: dict, values: dict) -> None:
    number = int(pending.get("number", 1))
    total = int(pending.get("total", config.MAX_INTERVIEW_QUESTIONS))
    question = pending.get("question", "")
    qkey = f"{ss.thread_id}:{number}"

    ui.topline(values.get("candidate_name", "Candidate"), total, number, values.get("qa_history") or [])
    ui.interviewer_panel(question, pending.get("topic", ""), values.get("current_difficulty", ""))
    _question_audio(qkey, question)

    ui.section_title("Your answer")
    attempt = ss.rec_attempt.get(qkey, 0)
    audio_file = st.audio_input(
        "Click the microphone to start recording, and click it again when you finish.",
        key=f"rec_{qkey}_{attempt}",
    )
    if audio_file is not None:
        _handle_recording(qkey, audio_file, question, pending.get("topic", ""))

    if STT_WARMUP["done"] and not STT_WARMUP["ok"]:
        st.error(STT_WARMUP["message"])

    stt_state = ss.stt.get(qkey, {})
    if stt_state.get("text"):
        if stt_state.get("low"):
            ui.note(f"Speech recognition was unsure (confidence {stt_state['confidence']:.0%}). "
                    "Read the transcript below and correct any misheard words, or record again "
                    "closer to the microphone.", "warn")
        else:
            ui.note(f"Answer transcribed with {stt_state['confidence']:.0%} confidence. "
                    "Check it below before you submit.")
    if stt_state.get("error"):
        ui.note(stt_state["error"], "bad")
        c1, c2 = st.columns(2)
        if stt_state.get("retryable") and c1.button("Retry transcription"):
            ss.stt.pop(qkey, None)
            st.rerun()
        if c2.button("Record again"):
            ss.stt.pop(qkey, None)
            ss.rec_attempt[qkey] = attempt + 1
            st.rerun()

    ans_key = f"ans_{qkey}"
    if config.ALLOW_TEXT_ANSWERS:
        st.text_area("Transcript (you can edit it, or type your answer instead)",
                     key=ans_key, height=150)
    elif stt_state.get("text"):
        ui.feedback(stt_state["text"])

    with st.expander("Microphone not working?"):
        st.markdown(
            "- Allow microphone access when the browser asks. If you blocked it, click the icon at the "
            "left of the address bar, allow the microphone, then reload. Your interview continues.\n"
            "- Check that a microphone is connected, not muted, and not in use by another app.\n"
            "- Open the app on **localhost** or an **https** address.\n"
            + ("- You can also type your answer in the box above." if config.ALLOW_TEXT_ANSWERS else "")
        )

    show_error_banner()
    c1, c2 = st.columns([3, 1])
    submit = c1.button("Submit answer", type="primary", width="stretch")
    skip = c2.button("Skip question", width="stretch")

    if submit or skip:
        # Guard against double clicks / stale pages: only answer the question still pending.
        latest = pending_interrupt(APP, _cfg())
        if not latest or latest.get("type") != "answer" or int(latest.get("number", -1)) != number:
            st.rerun()
        if submit:
            answer = (ss.get(ans_key) or "").strip()
            if not answer:
                st.warning("Record or type an answer first, or skip the question.")
                return
            transcript = stt_state.get("text") or ""
            source = "voice" if transcript else "text"
            norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
            resume = {
                "answer": answer, "skipped": False, "source": source,
                "transcript_confidence": stt_state.get("confidence") if transcript else None,
                "transcript_edited": bool(transcript) and norm(answer) != norm(transcript),
            }
        else:
            resume = {"answer": "", "skipped": True, "source": "skipped"}
        session = active_proctor()
        if session is not None:
            resume["incidents"] = list(session.incidents)   # proctoring log travels with the answer
            session.last_warning = None

        before = len(values.get("qa_history") or [])
        last_question = number >= total
        ok = run_graph(Command(resume=resume),
                       "Scoring your interview and writing the report..." if last_question
                       else "Saving your answer and preparing the next question...")
        after = len((APP.get_state(_cfg()).values or {}).get("qa_history") or [])
        if ok or after > before:
            ss.stt.pop(qkey, None)
            ss.pop(ans_key, None)
        st.rerun()


def render_answer_phase(pending: dict, values: dict) -> None:
    """Questions follow each other directly; all scoring is shown in the final report."""
    proctor_guard(values)
    render_question(pending, values)


def render_stalled(values: dict) -> None:
    # A technical problem is never the candidate's fault: proctoring pauses until it is fixed.
    proctor_guard(values, paused=True)
    ui.note(ss.last_error or "The previous step did not finish. Retry to continue from where it stopped.", "bad")
    if active_proctor() is not None:
        ui.note("This is a technical problem, not something you did. Proctoring is paused until it is "
                "fixed; after that you will be asked to return to fullscreen.")
    c1, c2 = st.columns(2)
    if c1.button("Retry", type="primary", width="stretch"):
        run_graph(None, "Retrying...")  # resumes from the last LangGraph checkpoint
        st.rerun()
    if active_proctor() is None and c2.button("Start over", width="stretch"):
        REG.pop(ss.thread_id, None)
        _delete_thread(ss.thread_id)
        _reset_voice_state()
        ss.thread_id = None
        st.query_params.clear()
        st.rerun()


def render_report(values: dict) -> None:
    REG.pop(ss.thread_id, None)   # interview over: proctoring stops
    exit_fullscreen()
    history = values.get("qa_history") or []
    terminated = bool(values.get("terminated"))

    report = values.get("final_report") or {}
    name = report.get("candidate_name", values.get("candidate_name", "Candidate"))
    ui.section_title(f"Interview report for {name}")
    if terminated:
        ui.verdict("Interview ended early", report.get("justification", ""))
    else:
        ui.verdict(report.get("recommendation", "-"), report.get("justification", ""))

    match = report.get("match_score")
    if report.get("screening_decision") == "REJECTED":
        ui.stats([("Resume match", f"{match}/100" if match is not None else "n/a"),
                  ("Screening decision", "Not shortlisted")])
    else:
        ui.stats([("Average score", f"{report.get('average_score', 0)}/10"),
                  ("Technical score", f"{report.get('technical_score', 0)}/10"),
                  ("Resume match", f"{match}/100" if match is not None else "n/a"),
                  ("Questions", report.get("num_questions", 0))])

    if report.get("question_scores"):
        ui.section_title("Question by question")
        ui.question_rows(report["question_scores"])

    ui.lists_block([("Strengths", report.get("strengths") or []),
                    ("Weaknesses", report.get("weaknesses") or []),
                    ("Areas to improve", report.get("improvements") or [])])

    incidents = report.get("incidents") or []
    if incidents or terminated or config.PROCTORING_ENABLED:
        ui.section_title("Proctoring log")
        ui.incident_rows(incidents)

    with st.expander("Resume screening details"):
        st.write(values.get("screening", ""))
    if history:
        ui.section_title("Answers and feedback")
        for i, qa in enumerate(history, 1):
            score = qa.get("score")
            label = "skipped" if qa.get("skipped") else ("not scored" if score is None else f"{score}/10")
            with st.expander(f"Question {i}: {label}"):
                st.markdown(f"**{qa.get('question')}**")
                st.markdown(f"*Answer:* {'(skipped)' if qa.get('skipped') else qa.get('candidate_answer')}")
                notes = []
                if qa.get("transcript_confidence") is not None:
                    notes.append(f"speech confidence {qa['transcript_confidence']:.0%}")
                if qa.get("transcript_edited"):
                    notes.append("transcript edited by candidate")
                if notes:
                    st.caption(", ".join(notes).capitalize())
                if score is not None and not qa.get("skipped"):
                    ui.evaluation_block(qa)
                if qa.get("key_point_coverage"):
                    ui.key_points(qa["key_point_coverage"])
                if qa.get("factual_errors"):
                    st.markdown("*Incorrect statements:* " + "; ".join(qa["factual_errors"]))
                if qa.get("feedback"):
                    ui.feedback(qa["feedback"])

    st.download_button(
        "Download report",
        data=values.get("final", ""),
        file_name=f"interview_report_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.txt",
        mime="text/plain",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def require_password() -> None:
    """If APP_PASSWORD is set, nothing is shown until the right password is entered."""
    import hmac

    if not config.APP_PASSWORD or ss.get("authenticated"):
        return
    ui.inject_theme()
    ui.header("AI technical interview")
    st.write("This interview room is private. Enter the access password to continue.")
    with st.form("login"):
        pwd = st.text_input("Password", type="password")
        ok = st.form_submit_button("Continue", type="primary")
    if ok:
        if hmac.compare_digest(pwd.encode("utf-8"), config.APP_PASSWORD.encode("utf-8")):
            ss.authenticated = True
            st.rerun()
        st.error("That password is not correct.")
    st.stop()


require_password()
_init_session()
header()
if ss.thread_id and JOBS.get(ss.thread_id) is not None:
    if JOBS.running(ss.thread_id):
        _values = APP.get_state(_cfg()).values or {}
        sidebar("working", _values)
        render_working(_values)
        st.stop()
    wait_for_job()   # finished while the page was re-running: pick up its result
phase, pending, values = current_phase()
check_page_reload(phase, values)   # a reloaded proctored interview ends here
sidebar(phase, values)

if phase == "setup":
    render_setup()
elif phase == "start_interview":
    render_start(values)
elif phase == "answer":
    render_answer_phase(pending, values)
elif phase == "stalled":
    render_stalled(values)
elif phase == "report":
    render_report(values)
