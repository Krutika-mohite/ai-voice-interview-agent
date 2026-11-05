import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from state import AgentState

logger = logging.getLogger("graph")

from parser_agent import parser_agent
from screening_agent import screening_agent
from interview_agent import interview_agent, await_interview_start, await_candidate_answer, record_answer
from evaluation_agent import evaluation_agent


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_start(state: AgentState) -> str:
    """Skip parsing/screening when they are already in the state
    (e.g. "restart interview" for the same candidate) - avoids reprocessing the resume."""
    if state.get("terminated"):
        return "evaluation"
    if state.get("parsed") and state.get("screening") and state.get("screening_decision"):
        return route_after_screening(state)
    return "parser"


def route_after_screening(state: AgentState) -> str:
    if state.get("screening_decision", "SELECTED") == "SELECTED":
        return "await_start"
    return "evaluation"   # not shortlisted -> straight to the final report


def route_after_await_answer(state: AgentState) -> str:
    """A proctoring termination skips scoring and goes straight to the final report."""
    return "evaluation" if state.get("terminated") else "record_answer"


def route_after_answer(state: AgentState) -> str:
    asked = len(state.get("qa_history") or [])
    if asked < int(state.get("max_questions") or 0):
        return "interview"     # adaptive: generate the next question
    return "evaluation"        # all questions answered -> final report


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    """
    START -> parser -> screening -> [selected?] -> await_start (interrupt)
          -> interview (1 question) -> await_answer (interrupt: voice/text answer)
          -> record_answer -> [more questions?] -> interview ... -> evaluation -> END
    Answers are scored in the background while the next question is answered, and
    evaluation collects every score at the end (no scoring wait between questions).
    A proctoring termination resumes await_answer with terminated=True and goes
    straight to evaluation (report of the questions answered so far).

    A checkpointer is required for interrupt(); an in-memory one is used by default.
    Every invoke must pass config={"configurable": {"thread_id": ...}}.
    """
    workflow = StateGraph(AgentState)

    # Add nodes (original four + three new ones)
    workflow.add_node("parser", parser_agent)
    workflow.add_node("screening", screening_agent)
    workflow.add_node("await_start", await_interview_start)
    workflow.add_node("interview", interview_agent)
    workflow.add_node("await_answer", await_candidate_answer)
    workflow.add_node("record_answer", record_answer)   # stores the answer; scored later
    workflow.add_node("evaluation", evaluation_agent)

    # Flow
    workflow.add_conditional_edges(
        START, route_start,  # may return "evaluation" for terminated/rejected candidates
        {"parser": "parser", "await_start": "await_start", "evaluation": "evaluation"},
    )
    workflow.add_edge("parser", "screening")
    workflow.add_conditional_edges(
        "screening", route_after_screening,
        {"await_start": "await_start", "evaluation": "evaluation"},
    )
    workflow.add_edge("await_start", "interview")
    workflow.add_edge("interview", "await_answer")
    workflow.add_conditional_edges(
        "await_answer", route_after_await_answer,
        {"record_answer": "record_answer", "evaluation": "evaluation"},
    )
    workflow.add_conditional_edges(
        "record_answer", route_after_answer,
        {"interview": "interview", "evaluation": "evaluation"},
    )
    workflow.add_edge("evaluation", END)

    return workflow.compile(checkpointer=checkpointer or InMemorySaver())


# --------------------------------------------------------------------------- #
# Helpers shared by the Streamlit UI and the CLI
# --------------------------------------------------------------------------- #
def run_step(app, config: dict, payload) -> None:
    """Run one graph step; if it fails on an LLM problem, retry once from the checkpoint."""
    from llm_utils import LLMError

    try:
        app.invoke(payload, config)
    except LLMError as exc:
        if app.get_state(config).next and pending_interrupt(app, config) is None:
            logger.warning("Graph step failed (%s); retrying once automatically", exc)
            app.invoke(None, config)
        else:
            raise


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def terminate_interview(app, config: dict, reason: str, incidents: list) -> None:
    """End an interview early (proctoring) and produce the final report.

    - Paused waiting for an answer: resume that interrupt with terminated=True.
    - Stalled on a failed step (e.g. LLM offline): record the termination as if the
      answer node had produced it, then continue - the report needs no LLM call.
    """
    payload = {"terminated": True, "reason": reason, "incidents": list(incidents or [])}
    pending = pending_interrupt(app, config)
    if pending and pending.get("type") == "answer":
        from langgraph.types import Command
        app.invoke(Command(resume=payload), config)
        return
    app.update_state(
        config,
        {"terminated": True, "termination_reason": reason, "pending_answer": {},
         "proctoring_incidents": payload["incidents"], "interview_status": "terminated"},
        as_node="await_answer",
    )
    app.invoke(None, config)


def pending_interrupt(app, config: dict):
    """Return the value passed to interrupt() if the run is paused, else None."""
    snapshot = app.get_state(config)
    for task in snapshot.tasks or ():
        for intr in getattr(task, "interrupts", ()) or ():
            return intr.value
    return None
