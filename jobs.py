"""
Background jobs for long graph steps (screening, writing the next question, scoring).

Why: Streamlit re-runs the page script whenever something on the page changes - for
example when the proctoring camera reports a warning while the candidate is waiting.
A re-run can interrupt whatever the script was doing at that moment. If that was a
1-2 minute LLM call inside the graph, the step was killed halfway ("The previous step
did not finish"). Running the step in a worker thread means page re-runs can never
interrupt it: the page only waits for the result, and any re-run simply waits again.

One job at a time per interview (thread_id). Pure Python - no Streamlit imports.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger("jobs")


@dataclass
class Job:
    future: Future
    label: str
    error: Optional[str] = field(default=None)


class JobRunner:
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="graph-job")
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, key: str, fn: Callable[[], None], label: str) -> Job:
        """Start fn for this interview unless a job is already running (then return it)."""
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and not job.future.done():
                return job
            job = Job(future=self._executor.submit(fn), label=label)
            self._jobs[key] = job
            return job

    def get(self, key: Optional[str]) -> Optional[Job]:
        if not key:
            return None
        with self._lock:
            return self._jobs.get(key)

    def running(self, key: Optional[str]) -> bool:
        job = self.get(key)
        return job is not None and not job.future.done()

    def wait(self, key: Optional[str], timeout: Optional[float] = None) -> None:
        job = self.get(key)
        if job is not None:
            try:
                job.future.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - reported by collect()
                pass

    def collect(self, key: Optional[str], error_message: Callable[[BaseException], str]):
        """For a finished job: remove it and return (finished, error text or None).

        Returns (False, None) if there is no job or it is still running.
        """
        with self._lock:
            job = self._jobs.get(key) if key else None
            if job is None or not job.future.done():
                return False, None
            self._jobs.pop(key, None)
        exc = job.future.exception()
        return True, (error_message(exc) if exc is not None else None)
