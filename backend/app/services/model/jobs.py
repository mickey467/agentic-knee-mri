"""Background inference jobs: run the ensemble off-request so long CPU runs
never hold an HTTP connection open (proxies/tunnels time those out).

Single worker thread: CPU inference is memory-heavy; jobs run sequentially.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def submit(study_id: str, series_uids: Optional[list] = None) -> str:
    """Queue an inference job. Returns job_id (status starts as 'running')."""
    from app.services.model.inference import run_inference

    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "study_id": study_id,
                         "status": "running", "result": None, "error": None}

    def _run():
        try:
            result = run_inference(study_id, series_uids=series_uids)
        except Exception as e:  # noqa: BLE001 — surfaced via job status
            with _lock:
                _jobs[job_id].update(status="failed", error=str(e))
        else:
            with _lock:
                _jobs[job_id].update(status="done", result=result)

    _executor.submit(_run)
    return job_id


def get(job_id: str) -> Optional[dict[str, Any]]:
    """Return the job record, or None for unknown ids."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def reset() -> None:
    """Clear the job store (tests). Does not cancel running jobs."""
    with _lock:
        _jobs.clear()
