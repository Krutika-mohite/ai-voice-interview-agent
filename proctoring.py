"""
Proctoring: camera checks + fullscreen / tab-switch enforcement.

Browser side (proctor_component/): MediaPipe face and hand models run in the
candidate's browser; no video leaves the computer. The component reports only
events such as {"type": "no_face", "label": "Face not visible"}.

Server side (this file): ProctorSession decides what an event means:
  - camera violations (no face, more than one face, looking away, face covered)
    count as warnings; the Nth (PROCTOR_MAX_VIOLATIONS, default 3) ends the interview
  - leaving fullscreen, switching tab/window or reloading the page ends it immediately
The decision is made here, not in the browser, and every incident is logged for the
final report.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("proctoring")

CAMERA_TYPES = {"no_face", "multiple_faces", "looking_away", "face_covered"}
FATAL_TYPES = {"left_fullscreen", "tab_switch", "window_switch", "page_reload"}
COMPONENT_DIR = Path(__file__).parent / "proctor_component"


class ProctorSession:
    """Server-side state for one proctored interview (pure Python, unit-testable)."""

    def __init__(self, thread_id: str, max_violations: Optional[int] = None):
        self.thread_id = thread_id
        self.max_violations = max(1, int(max_violations or config.PROCTOR_MAX_VIOLATIONS))
        self.seen_ids: set = set()
        self.violations = 0
        self.incidents: List[Dict[str, Any]] = []
        self.armed = False
        self.ended_reason: Optional[str] = None
        self.last_warning: Optional[str] = None

    # ------------------------------------------------------------------ #
    def arm(self) -> None:
        self.armed = True

    def record(self, ev_type: str, label: str, question: Optional[int], fatal: bool = False,
               counted: bool = False, at: Optional[str] = None) -> Dict[str, Any]:
        incident = {
            "type": ev_type, "label": label, "question": question, "fatal": fatal,
            "counted": counted, "at": at or datetime.now(timezone.utc).isoformat(),
        }
        self.incidents.append(incident)
        return incident

    def process(self, value: Optional[dict], question: Optional[int], paused: bool = False) -> Dict[str, Any]:
        """Apply new events from the component.

        paused=True while the app has a technical problem: events are logged as notes and
        never count (the candidate is not penalised for the app's own failure).
        Returns {"warnings": [labels of new counted warnings], "terminate": reason or None}.
        """
        result = {"warnings": [], "terminate": None}
        if not self.armed or self.ended_reason or not isinstance(value, dict):
            return result
        if value.get("nonce") and value.get("nonce") != self.thread_id:
            return result   # stale value from a previous interview
        if paused:
            for ev in value.get("events") or []:
                ev_id = ev.get("id")
                if ev_id and ev_id not in self.seen_ids:
                    self.seen_ids.add(ev_id)
                    self.record(str(ev.get("type", "")),
                                f"{ev.get('label', '')} (ignored: technical problem at the time)",
                                question, at=ev.get("at"))
            return result

        for ev in value.get("events") or []:
            ev_id = ev.get("id")
            if not ev_id or ev_id in self.seen_ids:
                continue
            self.seen_ids.add(ev_id)
            ev_type = str(ev.get("type", ""))
            label = str(ev.get("label") or ev_type.replace("_", " ").capitalize())

            if ev.get("fatal") or ev_type in FATAL_TYPES:
                self.record(ev_type, label, question, fatal=True, at=ev.get("at"))
                return self._end(label, result)

            if ev_type in CAMERA_TYPES:
                self.violations += 1
                self.record(ev_type, label, question, counted=True, at=ev.get("at"))
                if self.violations >= self.max_violations:
                    return self._end(f"{self.max_violations} proctoring warnings, the last one: {label.lower()}", result)
                self.last_warning = label
                result["warnings"].append(label)
            else:
                self.record(ev_type, label, question, at=ev.get("at"))   # logged, not counted
        return result

    def page_reloaded(self, question: Optional[int]) -> Dict[str, Any]:
        """The interview page was reloaded or opened elsewhere while proctored."""
        self.record("page_reload", "The interview page was reloaded or opened in another tab",
                    question, fatal=True)
        return self._end("the interview page was reloaded or opened in another tab", {"warnings": [], "terminate": None})

    def _end(self, reason: str, result: Dict[str, Any]) -> Dict[str, Any]:
        self.ended_reason = reason
        self.armed = False
        result["terminate"] = reason
        logger.info("Proctoring ended interview %s: %s", self.thread_id[:8], reason)
        return result


# --------------------------------------------------------------------------- #
# Streamlit component
# --------------------------------------------------------------------------- #
_COMPONENT = None


def _component():
    global _COMPONENT
    if _COMPONENT is None:
        import streamlit.components.v1 as components

        _COMPONENT = components.declare_component("interview_proctor", path=str(COMPONENT_DIR))
    return _COMPONENT


def _settings() -> Dict[str, Any]:
    return {
        "thresholds": {
            "noFaceSeconds": config.PROCTOR_NO_FACE_SECONDS,
            "multiFaceSeconds": config.PROCTOR_MULTI_FACE_SECONDS,
            "lookAwaySeconds": config.PROCTOR_LOOK_AWAY_SECONDS,
            "faceCoverSeconds": config.PROCTOR_FACE_COVER_SECONDS,
            "yawDegrees": config.PROCTOR_YAW_DEGREES,
            "pitchDegrees": config.PROCTOR_PITCH_DEGREES,
            "cooldown": config.PROCTOR_COOLDOWN_SECONDS,
            "repeatAfter": config.PROCTOR_REPEAT_SECONDS,
        },
        "focusGraceSeconds": config.PROCTOR_FOCUS_GRACE_SECONDS,
        "returnSeconds": config.PROCTOR_RETURN_SECONDS,
        "bundleUrl": config.PROCTOR_MEDIAPIPE_BUNDLE,
        "wasmUrl": config.PROCTOR_MEDIAPIPE_WASM,
        "faceModel": config.PROCTOR_FACE_MODEL_URL,
        "handModel": config.PROCTOR_HAND_MODEL_URL,
        "rulesText": [
            "Stay in fullscreen. Leaving fullscreen, switching tab or window, or reloading the page ends the interview immediately.",
            f"Keep only your face in view and look at the screen. Each problem lasting a few seconds is a warning; "
            f"warning {config.PROCTOR_MAX_VIOLATIONS} ends the interview.",
        ],
    }


def render_setup(thread_id: str):
    """Camera check + 'Enter fullscreen and start'. Returns the component value."""
    return _component()(mode="setup", nonce=thread_id, settings=_settings(),
                        key=f"proctor_setup_{thread_id}", default=None)


def render_exit(thread_id: str) -> None:
    """Leave browser fullscreen when the interview is over (invisible, zero height)."""
    _component()(mode="exit", nonce=thread_id, settings={}, key=f"proctor_exit_{thread_id}", default=None)


def render_monitor(session: ProctorSession, paused: bool = False, camera_paused: bool = False):
    """Small live strip shown during the interview. Returns the component value.

    paused=True suspends all checks (technical problem); when it turns False again the
    candidate is asked to return to fullscreen within a minute.
    camera_paused=True suspends only the camera warnings (candidate waiting for the AI).
    """
    return _component()(mode="monitor", nonce=session.thread_id, settings=_settings(),
                        warnings=session.violations, max_warnings=session.max_violations, paused=paused,
                        camera_paused=camera_paused,
                        key=f"proctor_monitor_{session.thread_id}", default=None)
