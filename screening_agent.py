import logging
import re

import config
from llm_utils import safe_invoke

logger = logging.getLogger("screening_agent")


def _parse_screening(text: str):
    """Return (decision, match_score) from the screening text.

    decision is "SELECTED" or "REJECTED". Explicit DECISION line wins; otherwise the
    match score is compared with SCREENING_PASS_SCORE; otherwise default to SELECTED
    (so an unparseable reply never silently blocks a candidate - the full screening
    text is still shown to the recruiter).
    """
    text = text or ""
    score = None
    scores = re.findall(r"match\s*score\s*[:\-]?\s*\**\s*(\d{1,3})", text, re.IGNORECASE)
    if scores:  # the required closing line is the last occurrence
        score = max(0, min(100, int(scores[-1])))

    for word in reversed(re.findall(r"decision\s*[:\-]\s*\**\s*([A-Za-z ]+)", text, re.IGNORECASE)):
        word = word.strip().lower()
        if word.startswith(("not select", "not shortlist", "reject", "no hire", "no")) and not word.startswith("now"):
            return "REJECTED", score
        if word.startswith(("select", "shortlist", "yes", "proceed", "hire")):
            return "SELECTED", score

    if score is not None:
        return ("SELECTED" if score >= config.SCREENING_PASS_SCORE else "REJECTED"), score

    logger.warning("Could not parse screening decision; defaulting to SELECTED")
    return "SELECTED", score


def screening_agent(state):
    propmt = f"""
    compare resume with job description.
    Give match score and decision.

    Write 2-4 short sentences: which required skills match, and which are missing.
    Then end your answer with exactly these two lines:
    MATCH SCORE: <number from 0 to 100>
    DECISION: SELECTED or REJECTED

    Resume: {state['parsed']}
    JD: {str(state['jd'])[:4000]}
"""
    screening = safe_invoke(propmt, max_tokens=config.LLM_MAX_TOKENS_SCREEN)
    decision, score = _parse_screening(screening)
    return {
        "screening": screening,
        "screening_decision": decision,
        "match_score": score,
    }
