"""
Voice-interview diagnostics. Run from the project folder (with the venv active):

    python diagnose.py                 # full check
    python diagnose.py --wav my.wav    # also transcribe one of your own recordings
    python diagnose.py --skip-llm      # only check the voice pipeline

Checks, in order:
  1. Python and package versions
  2. LLM reachability (and Ollama context window)
  3. Speech model loading
  4. TTS -> STT round trip with known sentences, measuring word accuracy
  5. Grading calibration: a strong, a weak and a wrong answer to the same question
     must be scored in that order, with clear gaps between them
Prints PASS / WARN / FAIL with a suggested fix for each problem.
"""

import argparse
import difflib
import json
import platform
import re
import sys
import time
import urllib.request
from importlib import metadata

RESULTS = []


def report(status: str, name: str, detail: str = "", fix: str = "") -> None:
    RESULTS.append(status)
    mark = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[status]
    print(f"{mark} {name}" + (f" - {detail}" if detail else ""))
    if fix and status in {"WARN", "FAIL"}:
        print(f"       Fix: {fix}")


def word_accuracy(expected: str, heard: str) -> float:
    a = re.findall(r"[a-z0-9]+", expected.lower())
    b = re.findall(r"[a-z0-9]+", heard.lower())
    if not a:
        return 0.0
    matched = sum(m.size for m in difflib.SequenceMatcher(None, a, b).get_matching_blocks())
    return matched / len(a)


def check_packages() -> None:
    print("\n== 1. Environment ==")
    report("INFO", "Python", f"{platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info >= (3, 14):
        report("WARN", "Python version", "very new Python - some audio packages may lack wheels",
               "use Python 3.11 or 3.12 from python.org")
    for pkg in ["langgraph", "langchain-ollama", "streamlit", "faster-whisper", "ctranslate2",
                "av", "onnxruntime", "edge-tts", "gTTS", "numpy", "pydantic"]:
        try:
            report("PASS", pkg, metadata.version(pkg))
        except metadata.PackageNotFoundError:
            report("FAIL", pkg, "not installed", "python -m pip install -r requirements.txt")


def check_llm() -> bool:
    print("\n== 2. LLM ==")
    import config
    from llm_utils import LLMError, safe_invoke

    report("INFO", "Provider", f"{config.LLM_PROVIDER} / "
           f"{config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.LLM_MODEL}")
    if config.LLM_PROVIDER == "ollama":
        try:
            with urllib.request.urlopen(config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags", timeout=5) as r:
                models = [m.get("name", "") for m in json.load(r).get("models", [])]
            if any(m.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for m in models):
                report("PASS", "Ollama server", f"running, model '{config.OLLAMA_MODEL}' available")
            else:
                report("FAIL", "Ollama model", f"'{config.OLLAMA_MODEL}' not pulled (have: {', '.join(models) or 'none'})",
                       f"ollama pull {config.OLLAMA_MODEL}")
                return False
        except Exception as exc:  # noqa: BLE001
            report("FAIL", "Ollama server", f"not reachable at {config.OLLAMA_BASE_URL} ({type(exc).__name__})",
                   "start Ollama (`ollama serve`) and keep it running")
            return False
        report("PASS" if config.OLLAMA_NUM_CTX >= 8192 else "WARN", "Context window",
               f"num_ctx={config.OLLAMA_NUM_CTX}", "set OLLAMA_NUM_CTX=8192 in .env")
    try:
        t = time.time()
        safe_invoke("Reply with the single word OK.")
        report("PASS", "LLM responds", f"{time.time() - t:.1f}s")
        return True
    except LLMError as exc:
        report("FAIL", "LLM responds", str(exc))
        return False


def check_stt(wav_path: str = "") -> None:
    print("\n== 3. Speech recognition ==")
    import config
    import voice

    report("INFO", "STT", f"{config.STT_PROVIDER}, model {config.WHISPER_MODEL_PATH or config.WHISPER_MODEL_SIZE}")
    t = time.time()
    ok, msg = voice.preload_model()
    if not ok:
        report("FAIL", "Speech model load", msg)
        print("       (The full technical error is printed above this line.)")
        return
    report("PASS", "Speech model load", f"{time.time() - t:.1f}s")

    print("\n== 4. Voice round trip (TTS -> STT) ==")
    sentences = [
        ("A tuple in Python is immutable, while a list is mutable and can be changed after creation.",
         "list vs tuple in Python"),
        ("FAISS stores embeddings in a vector index so similar documents can be retrieved quickly for RAG.",
         "vector search and RAG"),
    ]
    for text, topic in sentences:
        try:
            audio, _ = voice.synthesize(text)
        except voice.TTSError as exc:
            report("WARN", "Text-to-speech", exc.user_message,
                   "check internet access (edge-tts / gTTS need it); questions are still shown as text")
            return
        try:
            t = time.time()
            res = voice.transcribe_detailed(audio, question=topic, topic=topic)
            acc = word_accuracy(text, res.text)
            status = "PASS" if acc >= 0.9 else "WARN" if acc >= 0.75 else "FAIL"
            report(status, "Round trip", f"{acc:.0%} words correct, confidence {res.confidence:.0%}, "
                   f"{time.time() - t:.1f}s\n       heard: \"{res.text}\"",
                   "use a larger model: WHISPER_MODEL_SIZE=medium.en (slower, more accurate)")
        except voice.VoiceError as exc:
            report("FAIL", "Round trip", exc.user_message)

    if wav_path:
        print("\n== 4b. Your recording ==")
        try:
            with open(wav_path, "rb") as f:
                data = f.read()
            info = voice.prepare_audio(data)[1]
            report("INFO", "Audio", f"{info.duration_s:.1f}s, {info.sample_rate} Hz, level {info.rms:.0f}, "
                   f"gain x{info.gain:.1f}")
            res = voice.transcribe_detailed(data)
            report("PASS" if not res.low_confidence else "WARN", "Transcript",
                   f"confidence {res.confidence:.0%}\n       \"{res.text}\"",
                   "speak closer to the microphone / reduce background noise / use a headset")
        except voice.VoiceError as exc:
            report("FAIL", "Your recording", exc.user_message)
        except OSError as exc:
            report("FAIL", "Your recording", f"cannot open file ({exc})")


def check_grading() -> None:
    print("\n== 5. Grading calibration ==")
    from evaluation_agent import answer_evaluation_agent
    from llm_utils import LLMError

    question = "What is the difference between a list and a tuple in Python, and when would you use each?"
    key_points = [
        "Lists are mutable; tuples are immutable",
        "Tuples are hashable (if their items are) so they can be dict keys or set members",
        "Tuples are slightly faster and use less memory",
        "Use tuples for fixed records or constants, lists for collections that change",
    ]
    answers = {
        "strong": ("A list is mutable, so you can append, remove or change items, while a tuple is "
                   "immutable once created. Because tuples are immutable they are hashable when their "
                   "elements are, so you can use them as dictionary keys or put them in a set. Tuples are "
                   "also a bit faster and lighter in memory. I use a tuple for a fixed record such as a "
                   "coordinate or a database row, and a list when the collection grows or changes, like "
                   "items in a shopping cart."),
        "weak": "A list uses square brackets and a tuple uses round brackets. Both store several values.",
        "wrong": ("A tuple is mutable and a list is immutable, and tuples cannot be used as dictionary keys. "
                  "Lists are always faster than tuples."),
    }
    scores = {}
    for label, ans in answers.items():
        state = {"jd": "Python developer", "current_question": question, "current_difficulty": "medium",
                 "current_topic": "Python data structures", "current_key_points": key_points,
                 "qa_history": [], "pending_answer": {"answer": ans, "skipped": False, "source": "text"}}
        try:
            t = time.time()
            entry = answer_evaluation_agent(state)["qa_history"][-1]
            scores[label] = entry["score"]
            cov = sum(1 for c in entry["key_point_coverage"] if c["status"] == "covered")
            report("INFO", f"{label:6s} answer", f"score {entry['score']}/10, key points covered {cov}/"
                   f"{len(key_points)}, errors found {len(entry['factual_errors'])}, {time.time() - t:.1f}s")
        except LLMError as exc:
            report("FAIL", f"{label} answer", str(exc))
            return
    s, w, x = scores["strong"], scores["weak"], scores["wrong"]
    if s >= 7 and w <= 5 and x <= 3.5 and s - w >= 2:
        report("PASS", "Calibration", f"strong {s} > weak {w} > wrong {x}")
    else:
        report("WARN" if s > w > x or s > x else "FAIL", "Calibration",
               f"strong {s}, weak {w}, wrong {x} (expected strong >= 7, weak <= 5, wrong <= 3.5)",
               "the LLM is too weak for reliable grading: use a stronger model, e.g. "
               "OLLAMA_MODEL=qwen2.5:14b or llama3.1:8b, or LLM_PROVIDER=openai")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default="", help="path to one of your own recordings to transcribe")
    ap.add_argument("--skip-llm", action="store_true", help="skip LLM and grading checks")
    args = ap.parse_args()

    check_packages()
    llm_ok = False if args.skip_llm else check_llm()
    check_stt(args.wav)
    if llm_ok:
        check_grading()

    fails, warns = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\nSummary: {fails} failed, {warns} warnings.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
