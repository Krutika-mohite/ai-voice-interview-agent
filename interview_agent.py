"""
Interview agent.

Originally: one LLM call that produced 3 questions at once.
Now: ADAPTIVE - each call generates exactly ONE next question using the resume,
the JD and the previous questions and answers. In the SAME call the interviewer
judges the previous answer (strong / adequate / weak) and moves the difficulty up,
keeps it, or moves it down - so adapting costs no extra LLM call.

Answers are not scored during the interview: record_answer stores each answer, and
it is scored in the background (evaluation_agent.prefetch_score) while the candidate
answers the next question. The final report collects all scores at the end.

`await_candidate_answer` is the human-in-the-loop node: it pauses the LangGraph run
with interrupt() until the UI (voice or text) or CLI resumes it with the answer.
"""

import logging
import re
from typing import List, Optional

from langgraph.types import interrupt
from pydantic import BaseModel, Field

import config
from evaluation_agent import build_answer_entry, prefetch_score
from llm_utils import LLMBadOutput, foreground, invoke_structured

QUALITY_TO_SCORE = {"strong": 9.0, "adequate": 6.5, "weak": 3.0}


class QuestionPlan(BaseModel):
    """One interview question plus the answer key used later to grade it."""

    previous_answer_quality: str = Field(
        default="none", description='"strong", "adequate", "weak", or "none" for the first question')
    difficulty: str = Field(default="", description='"easy", "medium" or "hard"')
    question: str = Field(description="the single interview question to ask, plain spoken English")
    topic: str = Field(default="", description="the skill/topic this question tests, 1-4 words")
    key_points: List[str] = Field(
        default_factory=list,
        description="3 to 5 short, technically accurate points a strong answer must contain",
    )

logger = logging.getLogger("interview_agent")

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def next_difficulty(current: str, last_score: Optional[float], skipped: bool = False) -> str:
    """Pure function: adapt difficulty from the previous answer's score (0-10)."""
    idx = DIFFICULTY_LEVELS.index(current) if current in DIFFICULTY_LEVELS else 1
    if skipped or (last_score is not None and last_score < config.ADAPT_EASIER_BELOW):
        idx = max(0, idx - 1)
    elif last_score is not None and last_score >= config.ADAPT_HARDER_AT:
        idx = min(len(DIFFICULTY_LEVELS) - 1, idx + 1)
    return DIFFICULTY_LEVELS[idx]


def _adaptive_rules(last: Optional[dict], previous: str) -> str:
    if not last:
        return (
            f'This is the opening question: set "previous_answer_quality" to "none" and "difficulty" '
            f'to "{previous}". Start with a core skill that appears in both the job description and the resume.'
        )
    if last.get("skipped"):
        return (
            f'The candidate skipped the previous question: set "previous_answer_quality" to "weak" and '
            f'"difficulty" to "{next_difficulty(previous, None, True)}". Do not repeat it; move to a '
            "different skill from the job description with a more approachable question."
        )
    harder = next_difficulty(previous, 10)
    easier = next_difficulty(previous, 0)
    return (
        "First judge the candidate's LAST answer against the question it answered:\n"
        f'- "strong" (correct, complete, shows depth): set "difficulty" to "{harder}" and go deeper on '
        "the same topic or move to a harder area of the job description.\n"
        f'- "adequate" (mostly correct, some gaps): keep "difficulty" "{previous}" and probe the gap or '
        "move to another skill from the job description.\n"
        f'- "weak" (wrong, vague or very short): set "difficulty" to "{easier}" and ask a simpler question '
        "or a short clarifying follow-up that lets them show what they do know."
    )


def _history_text(history: List[dict]) -> str:
    """Compact history: earlier answers as short excerpts, the last answer in full."""
    if not history:
        return "(no questions asked yet)"
    lines = []
    for i, qa in enumerate(history, 1):
        answer = "[skipped]" if qa.get("skipped") else qa.get("candidate_answer", "")
        limit = 1200 if i == len(history) else 200
        excerpt = answer if len(answer) <= limit else answer[:limit] + "..."
        lines.append(
            f"Q{i} ({qa.get('difficulty', '?')}, topic: {qa.get('topic', '') or '-'}): {qa.get('question', '')}\n"
            f"A{i}{' (LAST ANSWER)' if i == len(history) else ''}: {excerpt}"
        )
    return "\n\n".join(lines)


_PREFIX_RE = re.compile(
    r"^(?:here(?:'s| is)[^:\n]*:|(?:next\s+)?question\s*\d*\s*[:.\-]|q\d+\s*[:.\-])\s*",
    re.IGNORECASE,
)


def clean_question(raw: str) -> str:
    """Strip LLM chatter/markdown so the question reads well and sounds right in TTS."""
    text = (raw or "").strip()
    # Prefer the first non-empty paragraph if the model added commentary afterwards.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if parts:
        text = parts[0]
        # If the first paragraph is only a lead-in ("Here is your question:"), take the next.
        if len(parts) > 1 and text.endswith(":") and len(text) < 80:
            text = parts[1]
    text = re.sub(r"[*_`#>]+", "", text)        # markdown symbols
    text = re.sub(r"\s+", " ", text).strip().strip("\"'“”‘’ ")
    text = _PREFIX_RE.sub("", text)             # "Question 2:", "Here is ...:" lead-ins
    text = text.strip("\"'“”‘’ ")
    return text


def interview_agent(state):
    """Generate the NEXT interview question (one LLM call, which also sets the difficulty)."""
    history: List[dict] = list(state.get("qa_history") or [])
    asked: List[str] = list(state.get("questions") or [])
    max_q = int(state.get("max_questions") or config.MAX_INTERVIEW_QUESTIONS)
    number = len(history) + 1

    last = history[-1] if history else None
    previous = (last.get("difficulty") if last else None) or state.get("current_difficulty") \
        or config.INTERVIEW_START_DIFFICULTY
    if previous not in DIFFICULTY_LEVELS:
        previous = "medium"

    # Static context first, changing parts last: Ollama reuses the cached prompt prefix.
    prompt = f"""
You are a senior technical interviewer running a live, spoken technical interview.

Job description:
{str(state.get('jd', ''))[:1500]}

Candidate profile (parsed resume):
{str(state.get('parsed', ''))[:1200]}

Interview so far:
{_history_text(history)}

Prepare question {number} of {max_q}. Previous difficulty: {previous}.
{_adaptive_rules(last, previous)}

Rules for "question":
- Exactly ONE question that has a clear, checkable technical answer.
- It will be read aloud: plain conversational English, no code, no markdown,
  no bullet points, at most 45 words.
- Do not repeat or rephrase a question that was already asked.
- Across the interview, cover different skills required by the job description.

Rules for "key_points" (the answer key, never shown to the candidate):
- 3 to 5 short points that a strong answer at this difficulty must contain.
- Each point must be technically correct and specific (no vague points like "good explanation").

Also give "topic": the skill being tested in 1-4 words.
"""
    key_points: List[str] = []
    topic = ""
    quality = "none"
    chosen = ""
    try:
        with foreground():   # background scoring waits; the candidate is waiting for this
            plan = invoke_structured(prompt, QuestionPlan, max_tokens=config.LLM_MAX_TOKENS_QUESTION)
        question = clean_question(plan.question)
        topic = (plan.topic or "").strip()[:60]
        key_points = [re.sub(r"\s+", " ", p).strip() for p in plan.key_points if str(p).strip()][:6]
        quality = str(plan.previous_answer_quality or "none").strip().lower()
        chosen = str(plan.difficulty or "").strip().lower()
    except LLMBadOutput as exc:
        # The model answered but not in JSON: use its text as the question (no answer key).
        logger.warning("Question plan was not structured; using plain text")
        question = clean_question(exc.raw)
    if not question:
        question = "Could you walk me through a recent technical project you worked on and your role in it?"

    # Difficulty: a skip always means easier; otherwise follow the interviewer's judgement
    # of the last answer (or its chosen level), falling back to the previous level.
    if last is None:
        difficulty = chosen if chosen in DIFFICULTY_LEVELS else previous
    elif last.get("skipped"):
        difficulty = next_difficulty(previous, None, True)
    elif quality in QUALITY_TO_SCORE:
        difficulty = next_difficulty(previous, QUALITY_TO_SCORE[quality])
    else:
        difficulty = chosen if chosen in DIFFICULTY_LEVELS else previous

    # Score the previous answer in the background now that the next question is ready,
    # so scoring does not compete with question generation for the model.
    if last is not None:
        prefetch_score(state.get("jd", ""), last)

    logger.info("Generated question %d/%d (difficulty=%s, previous answer=%s, %d key points)",
                number, max_q, difficulty, quality, len(key_points))
    return {
        "questions": asked + [question],
        "current_question": question,
        "current_topic": topic,
        "current_key_points": key_points,
        "current_difficulty": difficulty,
        "question_count": number,
        "interview_status": "in_progress",
    }


def record_answer(state):
    """Store the candidate's answer (unscored). No LLM call: scoring happens later."""
    history: List[dict] = list(state.get("qa_history") or [])
    entry = build_answer_entry(state, state.get("pending_answer") or {})
    logger.info("Recorded answer %d (%s)", len(history) + 1, entry["answer_source"])
    return {"qa_history": history + [entry], "pending_answer": {}}


def await_interview_start(state):
    """Pause after a successful screening until the recruiter/candidate starts the interview.

    Resume value (dict): {"max_questions": int}
    """
    decision = interrupt({
        "type": "start_interview",
        "candidate_name": state.get("candidate_name", "Candidate"),
    })
    max_q = config.MAX_INTERVIEW_QUESTIONS
    if isinstance(decision, dict) and decision.get("max_questions"):
        try:
            max_q = int(decision["max_questions"])
        except (TypeError, ValueError):
            pass
    max_q = max(1, min(max_q, 20))
    return {
        "max_questions": max_q,
        "qa_history": [],
        "questions": [],
        "question_count": 0,
        "current_difficulty": config.INTERVIEW_START_DIFFICULTY,
        "interview_status": "in_progress",
    }


def await_candidate_answer(state):
    """Human-in-the-loop: wait for the candidate's answer (voice transcript or text).

    Contains no LLM calls on purpose: LangGraph re-runs an interrupted node from the
    top when it resumes, so anything expensive here would be repeated.

    Resume value (dict): {"answer": str, "skipped": bool, "source": "voice"|"text",
                          "incidents": [...]}                  # optional proctoring log
                     or  {"terminated": True, "reason": str, "incidents": [...]}
    """
    reply = interrupt({
        "type": "answer",
        "question": state.get("current_question", ""),
        "topic": state.get("current_topic", ""),
        "number": state.get("question_count", 1),
        "total": state.get("max_questions", config.MAX_INTERVIEW_QUESTIONS),
    })
    if isinstance(reply, str):
        reply = {"answer": reply}
    reply = reply or {}

    update = {}
    if reply.get("incidents") is not None:           # proctoring log so far (warnings)
        update["proctoring_incidents"] = list(reply["incidents"])
    if reply.get("terminated"):                      # proctoring rule ended the interview
        update.update({
            "terminated": True,
            "termination_reason": str(reply.get("reason") or "Proctoring violation"),
            "pending_answer": {},
            "interview_status": "terminated",
        })
        return update

    answer = str(reply.get("answer") or "").strip()
    skipped = bool(reply.get("skipped")) or not answer
    confidence = reply.get("transcript_confidence")
    return {
        **update,
        "pending_answer": {
            "answer": "" if skipped else answer,
            "skipped": skipped,
            "source": "skipped" if skipped else reply.get("source", "text"),
            "transcript_confidence": float(confidence) if confidence is not None else None,
            "transcript_edited": bool(reply.get("transcript_edited", False)),
        }
    }
