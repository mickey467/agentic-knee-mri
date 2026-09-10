"""Shared conversation checkpoints and completed-report artifacts.

Checkpointer singletons live here so per-request graph builds still share
history keyed by thread_id = "{study_id}:{session_id}".
"""

from langgraph.checkpoint.memory import MemorySaver

REPORT_CHECKPOINTER = MemorySaver()
CHAT_CHECKPOINTER = MemorySaver()

# thread_id -> {"study_id": str, "report": str}
REPORT_STORE: dict[str, dict[str, str]] = {}


def thread_id(study_id: str, session_id: str) -> str:
    return f"{study_id}:{session_id or 'default'}"


def save_report(study_id: str, session_id: str, report: str) -> None:
    REPORT_STORE[thread_id(study_id, session_id)] = {"study_id": study_id, "report": report}


def load_report(study_id: str, session_id: str):
    return REPORT_STORE.get(thread_id(study_id, session_id))
