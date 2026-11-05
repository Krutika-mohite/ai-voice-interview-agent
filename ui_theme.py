"""
Visual theme and HTML components for the Streamlit interview UI.

All dynamic text is HTML-escaped before rendering. HTML strings are built without
indentation or blank lines so Streamlit's markdown renderer never turns them into
code blocks.
"""

import html
from typing import Dict, Iterable, List, Optional

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --ink: #12303B;
  --ink-soft: #24495A;
  --mist: #F2F5F4;
  --paper: #FFFFFF;
  --line: #D5DEDC;
  --muted: #5C7179;
  --signal: #E9A23B;
  --go: #2F7F73;
  --go-dark: #25665C;
  --mid: #B7791F;
  --low: #B4432F;
}

/* Base */
.stApp { background: var(--mist); color: var(--ink); }
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, label, input, textarea,
.stButton button, .stDownloadButton button, [data-testid="stCaptionContainer"] {
  font-family: 'Source Sans 3', system-ui, sans-serif;
}
.stMarkdown p, .stMarkdown li { font-size: 1.02rem; line-height: 1.6; }
h1, h2, h3, h4, .iv-display {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif !important;
  color: var(--ink); letter-spacing: -0.015em;
}
.block-container { max-width: 840px; padding-top: 1.6rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 999px; font-weight: 600; padding: 0.55rem 1.35rem;
  border: 1px solid var(--line); background: var(--paper); color: var(--ink);
  transition: background-color .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--go); color: var(--go); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--go); border-color: var(--go); color: #fff;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: var(--go-dark); border-color: var(--go-dark); color: #fff;
}
button:focus-visible, textarea:focus-visible, input:focus-visible {
  outline: 3px solid var(--signal) !important; outline-offset: 2px;
}

/* Inputs */
[data-testid="stFileUploader"] section {
  border: 1.5px dashed #B9C8C5; background: var(--paper); border-radius: 14px;
}
[data-testid="stTextArea"] textarea { border-radius: 12px; background: var(--paper); font-size: 1.02rem; }
[data-testid="stAudioInput"] {
  background: var(--paper); border: 1px solid var(--line); border-radius: 16px; padding: 0.35rem 0.5rem;
}
[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--line); background: var(--paper); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-size: 1.05rem; }

/* ---------- Components ---------- */
.iv-header { display: flex; align-items: baseline; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap; padding-bottom: 0.9rem; margin-bottom: 1.4rem;
  border-bottom: 1px solid var(--line); }
.iv-brand { display: flex; align-items: center; gap: 0.6rem; }
.iv-brand-mark { width: 30px; height: 30px; border-radius: 9px; background: var(--ink);
  display: inline-flex; align-items: center; justify-content: center; gap: 2px; }
.iv-brand-mark i { display: block; width: 3px; border-radius: 2px; background: var(--signal); }
.iv-brand-mark i:nth-child(1) { height: 8px; } .iv-brand-mark i:nth-child(2) { height: 16px; }
.iv-brand-mark i:nth-child(3) { height: 11px; } .iv-brand-mark i:nth-child(4) { height: 6px; }
.iv-brand-name { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: 1.35rem; color: var(--ink); }
.iv-header-note { color: var(--muted); font-size: 0.95rem; }

.iv-intro h1 { font-size: 2.3rem; line-height: 1.12; margin: 0 0 0.6rem; max-width: 26ch; text-wrap: balance; }
.iv-intro p { color: var(--ink-soft); font-size: 1.08rem; line-height: 1.6; max-width: 60ch; margin: 0 0 1.4rem; }
.iv-flow { display: flex; flex-wrap: wrap; gap: 0.5rem 1.4rem; margin: 0 0 1.8rem; padding: 0; list-style: none; }
.iv-flow li { color: var(--muted); font-size: 0.95rem; display: flex; align-items: center; gap: 0.45rem; }
.iv-flow b { display: inline-flex; width: 1.5rem; height: 1.5rem; border-radius: 50%;
  align-items: center; justify-content: center; font-size: 0.8rem; background: var(--paper);
  border: 1px solid var(--line); color: var(--ink); }

.iv-section-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: 1.15rem; color: var(--ink); margin: 1.6rem 0 0.5rem; }

/* Candidate line + stepper */
.iv-topline { display: flex; justify-content: space-between; align-items: center; gap: 1rem;
  flex-wrap: wrap; margin-bottom: 0.9rem; }
.iv-candidate { font-weight: 600; color: var(--ink); font-size: 1.05rem; }
.iv-candidate span { color: var(--muted); font-weight: 400; }
.iv-steps { display: flex; gap: 0.4rem; list-style: none; margin: 0; padding: 0; }
.iv-steps li { width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 0.82rem; font-weight: 700; color: var(--muted);
  border: 1.5px solid var(--line); background: var(--paper); }
.iv-steps li.done { color: #fff; border-color: transparent; }
.iv-steps li.current { border: 2px solid var(--signal); color: var(--ink); box-shadow: 0 0 0 3px rgba(233,162,59,.18); }

/* The interviewer panel - the one bold element */
.iv-panel { background: var(--ink); color: #fff; border-radius: 22px; padding: 1.6rem 1.8rem 1.8rem;
  margin: 0 0 0.8rem; position: relative; overflow: hidden; }
.iv-panel::after { content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  border-radius: 50%; border: 1px solid rgba(233,162,59,.18); }
.iv-who { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1rem; color: #C9D8DC; font-size: 0.95rem; }
.iv-bars { display: inline-flex; align-items: center; gap: 3px; height: 22px; }
.iv-bars i { display: block; width: 4px; height: 100%; border-radius: 3px; background: var(--signal);
  transform-origin: center; animation: iv-talk 1.1s ease-in-out infinite; }
.iv-bars i:nth-child(2) { animation-delay: .15s; } .iv-bars i:nth-child(3) { animation-delay: .3s; }
.iv-bars i:nth-child(4) { animation-delay: .45s; } .iv-bars i:nth-child(5) { animation-delay: .6s; }
@keyframes iv-talk { 0%, 100% { transform: scaleY(.3); } 50% { transform: scaleY(1); } }
.iv-chip { margin-left: auto; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,.22); color: #E4ECEE; white-space: nowrap; }
.iv-q { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 500; font-size: 1.62rem;
  line-height: 1.3; margin: 0; max-width: 40ch; text-wrap: pretty; color: #fff; position: relative; z-index: 1; }

/* Status notes */
.iv-note { border-left: 4px solid var(--go); background: var(--paper); padding: 0.7rem 1rem;
  border-radius: 0 10px 10px 0; color: var(--ink-soft); margin: 0.6rem 0 0.4rem; font-size: 0.98rem; }
.iv-note.warn { border-left-color: var(--mid); }
.iv-note.bad { border-left-color: var(--low); }

/* Evaluation */
.iv-eval { display: grid; grid-template-columns: 150px 1fr; gap: 1.6rem; align-items: center;
  background: var(--paper); border: 1px solid var(--line); border-radius: 18px; padding: 1.4rem 1.6rem; margin: 0.4rem 0 1rem; }
.iv-ring { width: 136px; height: 136px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.iv-ring > div { width: 104px; height: 104px; border-radius: 50%; background: var(--paper);
  display: flex; flex-direction: column; align-items: center; justify-content: center; }
.iv-ring b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.1rem; line-height: 1; color: var(--ink); }
.iv-ring span { color: var(--muted); font-size: 0.85rem; margin-top: 0.15rem; }
.iv-crit { display: grid; gap: 0.55rem; }
.iv-crit-row { display: grid; grid-template-columns: 11rem 1fr 2.4rem; gap: 0.7rem; align-items: center; font-size: 0.95rem; }
.iv-crit-row span:first-child { color: var(--ink-soft); }
.iv-track { height: 8px; border-radius: 99px; background: #E6ECEB; overflow: hidden; }
.iv-track > i { display: block; height: 100%; border-radius: 99px; }
.iv-crit-row b { text-align: right; color: var(--ink); }

.iv-points { list-style: none; padding: 0; margin: 0.3rem 0 0.8rem; display: grid; gap: 0.45rem; }
.iv-points li { display: flex; gap: 0.65rem; align-items: flex-start; line-height: 1.45; }
.iv-points .mk { flex: none; width: 1.4rem; height: 1.4rem; border-radius: 50%; display: inline-flex;
  align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; color: #fff; margin-top: 0.1rem; }
.iv-points .covered .mk { background: var(--go); }
.iv-points .partial .mk { background: var(--mid); }
.iv-points .missing .mk { background: #fff; color: var(--low); border: 1.5px solid var(--low); }
.iv-points .missing { color: var(--muted); }

.iv-feedback { font-size: 1.05rem; line-height: 1.65; color: var(--ink); margin: 0.2rem 0 1rem; max-width: 68ch; }

/* Report */
.iv-verdict { border-radius: 22px; padding: 1.6rem 1.8rem; margin: 0 0 1.2rem; color: #fff; }
.iv-verdict small { display: block; opacity: .85; font-size: 0.95rem; margin-bottom: 0.2rem; }
.iv-verdict .word { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.6rem; line-height: 1.05; }
.iv-verdict p { margin: 0.7rem 0 0; max-width: 62ch; line-height: 1.55; opacity: .95; }
.iv-verdict.strong { background: #1F6B5F; } .iv-verdict.hire { background: var(--go); }
.iv-verdict.consider { background: #8A5A12; } .iv-verdict.nohire { background: #8E3324; }
.iv-stats { display: flex; flex-wrap: wrap; gap: 0.8rem 2.4rem; margin: 0 0 1.4rem; padding: 0; list-style: none; }
.iv-stats li { display: flex; flex-direction: column; }
.iv-stats b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; color: var(--ink); line-height: 1.1; }
.iv-stats span { color: var(--muted); font-size: 0.9rem; }
.iv-qrow { display: grid; grid-template-columns: 2rem 1fr 10.5rem; gap: 0.8rem; align-items: center;
  padding: 0.65rem 0; border-bottom: 1px solid var(--line); }
.iv-qrow .n { font-weight: 700; color: var(--muted); }
.iv-qrow.iv-inc { grid-template-columns: 4.6rem 1fr 10.5rem; }
.iv-qrow .t { color: var(--ink); line-height: 1.4; }
.iv-qrow .t small { display: block; color: var(--muted); font-size: 0.85rem; margin-top: 0.15rem; }
.iv-qrow .s { display: grid; grid-template-columns: 1fr 4.2rem; gap: 0.5rem; align-items: center; }
.iv-qrow .s b { text-align: right; }
.iv-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.4rem; margin: 1.2rem 0; }
.iv-cols h4 { margin: 0 0 0.4rem; font-size: 1.05rem; }
.iv-cols ul { margin: 0; padding-left: 1.1rem; }
.iv-cols li { margin-bottom: 0.35rem; line-height: 1.5; }

@media (max-width: 640px) {
  .iv-intro h1 { font-size: 1.8rem; }
  .iv-q { font-size: 1.3rem; }
  .iv-panel { padding: 1.3rem 1.2rem 1.4rem; border-radius: 18px; }
  .iv-eval { grid-template-columns: 1fr; justify-items: center; }
  .iv-crit { width: 100%; }
  .iv-crit-row { grid-template-columns: 8.5rem 1fr 2.2rem; }
  .iv-qrow { grid-template-columns: 1.6rem 1fr; }
  .iv-qrow .s { grid-column: 2; }
  .iv-qrow.iv-inc { grid-template-columns: 4.2rem 1fr; }
  .iv-qrow.iv-inc .s { grid-column: 2; text-align: left !important; }
}
@media (prefers-reduced-motion: reduce) {
  .iv-bars i { animation: none; transform: scaleY(.7); }
  .stButton > button { transition: none; }
}
</style>
"""


def _e(text) -> str:
    return html.escape(str(text if text is not None else ""))


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_theme() -> None:
    render(CSS)


def score_color(score: Optional[float]) -> str:
    if score is None:
        return "#B9C8C5"
    if score >= 7:
        return "#2F7F73"
    if score >= 5:
        return "#B7791F"
    return "#B4432F"


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def header(note: str = "") -> None:
    render(
        '<div class="iv-header"><div class="iv-brand">'
        '<span class="iv-brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>'
        '<span class="iv-brand-name">Interview room</span></div>'
        f'<span class="iv-header-note">{_e(note)}</span></div>'
    )


def intro() -> None:
    render(
        '<div class="iv-intro"><h1>Screen a resume, then interview the candidate by voice</h1>'
        '<p>Upload a resume and paste the job description. If the candidate is a match, '
        'the AI interviewer asks spoken questions, adapts to each answer and writes a hiring report.</p>'
        '<ol class="iv-flow"><li><b>1</b>Resume screening</li><li><b>2</b>Voice interview</li>'
        '<li><b>3</b>Hiring report</li></ol></div>'
    )


def section_title(text: str) -> None:
    render(f'<div class="iv-section-title">{_e(text)}</div>')


def note(text: str, kind: str = "") -> None:
    render(f'<div class="iv-note {kind}">{_e(text)}</div>')


def topline(candidate: str, total: int, current: int, history: List[dict], status: str = "") -> None:
    """Candidate name + question tracker (done questions coloured by score)."""
    items = []
    for i in range(1, max(total, 1) + 1):
        if i <= len(history):
            qa = history[i - 1]
            if qa.get("skipped"):
                label, color = "skipped", score_color(None)
            elif qa.get("score") is None:          # scores are only shown in the final report
                label, color = "answered", "#24495A"
            else:
                label, color = f"{qa['score']}/10", score_color(qa["score"])
            items.append(f'<li class="done" style="background:{color}" '
                         f'title="Question {i}: {label}" aria-label="Question {i}, {label}">{i}</li>')
        elif i == current:
            items.append(f'<li class="current" aria-current="step" aria-label="Question {i}, current">{i}</li>')
        else:
            items.append(f'<li aria-label="Question {i}, not started">{i}</li>')
    render(
        f'<div class="iv-topline"><div class="iv-candidate">{_e(candidate)} '
        f'<span>{_e(status or f"is answering question {min(current, total)} of {total}")}</span></div>'
        f'<ol class="iv-steps" aria-label="Interview progress">{"".join(items)}</ol></div>'
    )


def interviewer_panel(question: str, topic: str = "", difficulty: str = "") -> None:
    chip = ""
    bits = [b for b in (topic, f"{difficulty} difficulty" if difficulty else "") if b]
    if bits:
        chip = f'<span class="iv-chip">{_e(", ".join(bits))}</span>'
    render(
        '<section class="iv-panel" aria-label="Interviewer question">'
        '<div class="iv-who"><span class="iv-bars" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>'
        f'<span>The interviewer asks</span>{chip}</div>'
        f'<p class="iv-q">{_e(question)}</p></section>'
    )


def evaluation_block(entry: dict) -> None:
    score = float(entry.get("score") or 0)
    pct = max(0.0, min(100.0, score * 10))
    color = score_color(score)
    rows = []
    labels = {
        "technical_correctness": "Technical correctness",
        "completeness": "Completeness",
        "understanding": "Understanding",
        "relevance": "Relevance",
        "explanation_quality": "Explanation",
    }
    crit = entry.get("criteria") or {}
    for key, label in labels.items():
        if key in crit:
            v = float(crit[key])
            rows.append(
                f'<div class="iv-crit-row"><span>{label}</span><div class="iv-track">'
                f'<i style="width:{v * 10:.0f}%;background:{score_color(v)}"></i></div><b>{v:g}</b></div>'
            )
    render(
        '<div class="iv-eval">'
        f'<div class="iv-ring" style="background:conic-gradient({color} {pct:.0f}%, #E6ECEB 0)" '
        f'role="img" aria-label="Score {score:g} out of 10"><div><b>{score:g}</b><span>out of 10</span></div></div>'
        f'<div class="iv-crit">{"".join(rows)}</div></div>'
    )


def key_points(coverage: Iterable[Dict[str, str]]) -> None:
    marks = {"covered": "✓", "partial": "½", "missing": "✕"}
    words = {"covered": "Covered", "partial": "Partly covered", "missing": "Missed"}
    items = []
    for c in coverage:
        status = c.get("status", "missing")
        items.append(
            f'<li class="{status}"><span class="mk" aria-label="{words.get(status, "Missed")}">'
            f'{marks.get(status, "✕")}</span><span>{_e(c.get("point", ""))}</span></li>'
        )
    if items:
        render(f'<ul class="iv-points">{"".join(items)}</ul>')


def feedback(text: str) -> None:
    render(f'<p class="iv-feedback">{_e(text)}</p>')


def verdict(recommendation: str, justification: str) -> None:
    cls = {"Strong Hire": "strong", "Hire": "hire", "Shortlisted": "hire", "Consider": "consider"}.get(recommendation, "nohire")
    render(
        f'<div class="iv-verdict {cls}"><small>Recommendation</small>'
        f'<div class="word">{_e(recommendation)}</div><p>{_e(justification)}</p></div>'
    )


def stats(items: List[tuple]) -> None:
    lis = "".join(f'<li><b>{_e(v)}</b><span>{_e(k)}</span></li>' for k, v in items)
    render(f'<ul class="iv-stats">{lis}</ul>')


def question_rows(question_scores: List[dict]) -> None:
    rows = []
    for i, q in enumerate(question_scores, 1):
        sc = None if q.get("skipped") or q.get("score") is None else float(q["score"])
        width = 0 if sc is None else sc * 10
        val = "Skipped" if q.get("skipped") else ("Not scored" if sc is None else f"{sc:g}")
        sub = [q.get("difficulty", "").capitalize() + " difficulty" if q.get("difficulty") else ""]
        if q.get("coverage"):
            sub.append(f"key points {q['coverage']}")
        if q.get("transcript_edited"):
            sub.append("transcript edited")
        meta = ", ".join(s for s in sub if s)
        rows.append(
            f'<div class="iv-qrow"><span class="n">{i}</span>'
            f'<span class="t">{_e(q.get("question", ""))}<small>{_e(meta)}</small></span>'
            f'<span class="s"><span class="iv-track"><i style="width:{width:.0f}%;background:{score_color(sc)}"></i></span>'
            f'<b>{val}</b></span></div>'
        )
    render("".join(rows))


def incident_rows(incidents: List[dict]) -> None:
    """Proctoring log for the report: time, what happened, and its effect."""
    if not incidents:
        render('<p class="iv-feedback">No proctoring incidents were recorded.</p>')
        return
    rows = []
    for inc in incidents:
        effect = "Ended the interview" if inc.get("fatal") else ("Warning" if inc.get("counted") else "Logged")
        color = "#B4432F" if inc.get("fatal") else ("#B7791F" if inc.get("counted") else "#5C7179")
        when = str(inc.get("at", ""))[11:19]
        q = f"Question {inc['question']}" if inc.get("question") else "Before the first question"
        rows.append(
            f'<div class="iv-qrow iv-inc"><span class="n" style="font-weight:600">{_e(when)}</span>'
            f'<span class="t">{_e(inc.get("label", ""))}<small>{_e(q)}</small></span>'
            f'<span class="s" style="display:block;text-align:right"><b style="color:{color}">{effect}</b></span></div>'
        )
    render("".join(rows))


def lists_block(groups: List[tuple]) -> None:
    cols = []
    for title, items in groups:
        if not items:
            continue
        lis = "".join(f"<li>{_e(x)}</li>" for x in items)
        head = f"<h4>{_e(title)}</h4>" if title else ""
        cols.append(f"<div>{head}<ul>{lis}</ul></div>")
    if cols:
        render(f'<div class="iv-cols">{"".join(cols)}</div>')
