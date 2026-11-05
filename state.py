from typing import Any, Dict, List, Optional, TypedDict


class QAEntry(TypedDict, total=False):
    """One interview turn: the question, the candidate's answer and its evaluation."""

    question: str
    candidate_answer: str
    score: float                 # overall 0-10 (mean of the criteria below)
    feedback: str
    criteria: Dict[str, float]   # technical_correctness, relevance, completeness,
                                 # understanding, explanation_quality (each 0-10)
    difficulty: str              # easy | medium | hard
    skipped: bool
    answer_source: str           # "voice" | "text" | "skipped"
    topic: str
    key_points: List[str]                    # answer key generated with the question
    key_point_coverage: List[Dict[str, str]] # [{"point", "status": covered|partial|missing}]
    coverage_score: Optional[float]          # 0-10 measured key-point coverage
    factual_errors: List[str]
    transcript_confidence: Optional[float]   # 0-1 speech-recognition confidence
    transcript_edited: bool                  # candidate changed the voice transcript


class AgentState(TypedDict, total=False):
    # ---- Original fields (kept) -------------------------------------------
    resume: str
    jd: str
    parsed: str
    screening: str
    questions: List[str]         # was a single string of 3 questions; now the
                                 # list of questions actually asked, in order
    final: str                   # final report as readable text (as before)

    # ---- Screening (new) ----------------------------------------------------
    candidate_name: str
    screening_decision: str      # "SELECTED" | "REJECTED"
    match_score: Optional[int]   # 0-100 if the screening LLM provided one

    # ---- Adaptive interview (new) -------------------------------------------
    max_questions: int
    current_question: str
    current_topic: str
    current_key_points: List[str]   # answer key for the current question (not shown to candidate)
    current_difficulty: str
    question_count: int          # number of questions asked so far
    pending_answer: Dict[str, Any]   # answer received from the candidate (voice/text)
    qa_history: List[QAEntry]
    interview_status: str        # awaiting_start | in_progress | completed | not_shortlisted | terminated

    # ---- Proctoring (new) ---------------------------------------------------
    proctoring_incidents: List[Dict[str, Any]]  # warnings/violations: type, label, at, question, counted
    terminated: bool                            # interview ended early by a proctoring rule
    termination_reason: str

    # ---- Final report (new) -------------------------------------------------
    final_report: Dict[str, Any]
