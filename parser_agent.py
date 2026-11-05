import re

import config
from llm_utils import safe_invoke


def _extract_candidate_name(parsed: str) -> str:
    """Read the 'Candidate Name: ...' line the prompt asks for (best effort)."""
    m = re.search(r"candidate\s*name\s*[:\-]\s*\**\s*([^\n*]+)", parsed or "", re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip(".,\"'")
        if name and name.lower() not in {"unknown", "n/a", "not provided", "not mentioned"}:
            return name[:80]
    return "Candidate"


def parser_agent(state):
    propmt = f"""
Extract skills, experience and education from this resume.
Answer in at most 120 words, in exactly this format:
Candidate Name: <full name or Unknown>
Skills: <comma-separated technical skills>
Experience: <total years; recent roles and key projects, 1-3 short lines>
Education: <highest degree, 1 line>

Resume:
{str(state['resume'])[:12000]}
    """
    parsed = safe_invoke(propmt, max_tokens=config.LLM_MAX_TOKENS_PARSE)
    return {"parsed": parsed, "candidate_name": _extract_candidate_name(parsed)}
