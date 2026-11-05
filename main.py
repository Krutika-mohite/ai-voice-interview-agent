"""
Command-line runner (text mode). The voice interview runs in the Streamlit app:
    streamlit run app.py

This CLI drives the SAME LangGraph graph; answers are typed instead of spoken.
"""

import argparse
import uuid

from langgraph.types import Command

import config
from graph import build_graph, pending_interrupt, thread_config
from llm_utils import LLMError
from resume_loader import load_resume

DEFAULT_JD = "Looking for Python developer with AI and ML experience."


def _invoke_with_retry(app, payload, cfg):
    """Invoke the graph; on LLM failure let the user retry from the last checkpoint."""
    while True:
        try:
            return app.invoke(payload, cfg)
        except LLMError as e:
            print(f"\n[!] {e}")
            if input("Retry? [Y/n]: ").strip().lower() == "n":
                raise SystemExit(1)
            payload = None  # resume the failed step from the checkpoint


def main():
    parser = argparse.ArgumentParser(description="AI Resume Screening + Interview (CLI)")
    parser.add_argument("--resume", default="resume.txt", help="Path to resume (.pdf or .txt)")
    parser.add_argument("--jd", default=DEFAULT_JD, help="Job description text")
    parser.add_argument("--questions", type=int, default=config.MAX_INTERVIEW_QUESTIONS)
    args = parser.parse_args()

    app = build_graph()

    resume_text = load_resume(args.resume)
    if not resume_text.strip():
        print(f"Could not read any text from '{args.resume}'.")
        return

    cfg = thread_config(str(uuid.uuid4()))
    _invoke_with_retry(app, {"resume": resume_text, "jd": args.jd}, cfg)

    while True:
        pending = pending_interrupt(app, cfg)
        if pending is None:
            break

        if pending.get("type") == "start_interview":
            state = app.get_state(cfg).values
            print("\n SCREENING RESULT:\n")
            print(state.get("screening", ""))
            input(f"\nCandidate shortlisted. Press Enter to start a {args.questions}-question interview...")
            _invoke_with_retry(app, Command(resume={"max_questions": args.questions}), cfg)

        elif pending.get("type") == "answer":
            print(f"\nQuestion {pending['number']} of {pending['total']}:\n{pending['question']}")
            answer = input("\nYour answer (blank to skip): ").strip()
            _invoke_with_retry(
                app, Command(resume={"answer": answer, "skipped": not answer, "source": "text"}), cfg
            )
            print("Answer recorded. (All answers are scored in the final report.)")

    response = app.get_state(cfg).values
    print("\n FINAL RESULT:\n")
    print(response["final"])


if __name__ == "__main__":
    main()
