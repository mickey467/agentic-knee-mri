"""Phase 1: report generation endpoint (demographics interrupt + resume)."""

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

router = APIRouter(prefix="/studies", tags=["agent"])


class ReportRequest(BaseModel):
    session_id: str = "default"
    resume_payload: Optional[dict[str, Any]] = None


def _config_for(study_id: str, session_id: str) -> tuple[str, dict]:
    from app.services.agent.store import thread_id

    tid = thread_id(study_id, session_id or "default")
    return tid, {"configurable": {"thread_id": tid}}


@router.post("/{study_id}/report")
async def generate_report(study_id: str, body: ReportRequest):
    from app.services.agent.failover import ainvoke_with_failover
    from app.services.agent.graph import build_report_graph, get_interrupt_values
    from app.services.agent.prompts import missing_report_sections, report_task_message
    from app.services.agent.provider import LLMNotConfigured
    from app.services.agent.store import save_report
    from app.services.agent.trace import last_ai_text, summarize_trace
    from app.services.dicom.service import study_manager

    if not study_manager.get_study(study_id):
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")

    _, config = _config_for(study_id, body.session_id)
    if body.resume_payload is not None:
        payload = Command(resume=body.resume_payload)
    else:
        payload = {
            "messages": [HumanMessage(content=report_task_message(study_id))],
            "study_id": study_id,
        }
    try:
        _, model_used, graph = await ainvoke_with_failover(build_report_graph, payload, config)
    except LLMNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        from app.services.agent.provider import is_capacity_error

        if is_capacity_error(e):
            raise HTTPException(
                status_code=503,
                detail="NIM model is overloaded right now (ResourceExhausted). Please retry shortly.",
            )
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    pauses = get_interrupt_values(graph, config)
    if pauses:
        first = pauses[0] if isinstance(pauses[0], dict) else {}
        questions = first.get("ask") or ["patient_age", "patient_sex"]
        return {"status": "paused_for_demographics", "study_id": study_id, "questions": questions}

    snapshot = graph.get_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    report = last_ai_text(messages)
    if not report:
        raise HTTPException(status_code=500, detail="Agent finished without producing a report.")

    save_report(study_id, body.session_id, report)
    warnings: list[str] = []
    missing = missing_report_sections(report)
    if missing:
        warnings.append(f"Report is missing sections: {', '.join(missing)}")
    trace = summarize_trace(messages)
    if any(t.get("tool") == "get_predictions" and "failed" in t.get("result", "").lower() for t in trace):
        warnings.append("inference_unavailable: findings are preliminary")

    return {
        "status": "ok",
        "study_id": study_id,
        "model": model_used,
        "report": report,
        "warnings": warnings,
        "tool_trace": trace,
    }


@router.post("/{study_id}/report/stream")
async def stream_report(study_id: str, body: ReportRequest):
    """SSE variant of report generation: streams tool progress events.

    Events: tool ({tool, phase}), model ({model, note}), paused ({questions}),
    report ({report, warnings, model, tool_trace}), error ({detail}).
    Same pause/resume + failover semantics as POST /report.
    """
    from app.services.dicom.service import study_manager

    if not study_manager.get_study(study_id):
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    return StreamingResponse(
        _report_events(study_id, body.session_id, body.resume_payload),
        media_type="text/event-stream",
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _report_events(study_id: str, session_id: str, resume_payload):
    from app.services.agent.graph import build_report_graph, get_interrupt_values
    from app.services.agent.prompts import missing_report_sections, report_task_message
    from app.services.agent.provider import (
        LLMNotConfigured,
        get_fallback_model,
        has_fallback,
        is_capacity_error,
    )
    from app.services.agent.store import save_report
    from app.services.agent.trace import last_ai_text, summarize_trace

    try:
        graph = build_report_graph()
    except LLMNotConfigured as e:
        yield _sse("error", {"detail": str(e)})
        return

    _, config = _config_for(study_id, session_id or "default")
    if resume_payload is not None:
        payload = Command(resume=resume_payload)
    else:
        payload = {
            "messages": [HumanMessage(content=report_task_message(study_id))],
            "study_id": study_id,
        }

    seen: set = set()
    model_used = "primary"

    async def _drain(g):
        async for _ns, chunk in g.astream(payload, config, stream_mode="updates", subgraphs=True):
            if not isinstance(chunk, dict):
                continue
            for _node, update in chunk.items():
                msgs = update.get("messages", []) if isinstance(update, dict) else []
                for m in msgs or []:
                    mtype = getattr(m, "type", None)
                    if mtype == "ai" and getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            key = ("call", tc.get("id"))
                            if key not in seen:
                                seen.add(key)
                                yield _sse("tool", {"tool": tc.get("name"), "phase": "start"})
                    elif mtype == "tool":
                        key = ("result", getattr(m, "tool_call_id", None))
                        if key not in seen:
                            seen.add(key)
                            yield _sse("tool", {"tool": getattr(m, "name", "tool"), "phase": "end"})

    try:
        async for evt in _drain(graph):
            yield evt
    except Exception as e:
        if not is_capacity_error(e) or not has_fallback():
            yield _sse("error", {"detail": f"Report generation failed: {e}"})
            return
        model_used = "fallback"
        yield _sse("model", {"model": "fallback", "note": "primary overloaded, retrying"})
        try:
            graph = build_report_graph(model=get_fallback_model())
        except LLMNotConfigured as e2:
            yield _sse("error", {"detail": str(e2)})
            return
        try:
            async for evt in _drain(graph):
                yield evt
        except Exception as e2:
            yield _sse("error", {"detail": f"Report generation failed: {e2}"})
            return

    pauses = get_interrupt_values(graph, config)
    if pauses:
        first = pauses[0] if isinstance(pauses[0], dict) else {}
        yield _sse("paused", {"questions": first.get("ask") or ["patient_age", "patient_sex"]})
        return

    snapshot = graph.get_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    report = last_ai_text(messages)
    if not report:
        yield _sse("error", {"detail": "Agent finished without producing a report."})
        return

    save_report(study_id, session_id, report)
    warnings: list[str] = []
    missing = missing_report_sections(report)
    if missing:
        warnings.append(f"Report is missing sections: {', '.join(missing)}")
    trace = summarize_trace(messages)
    if any(t.get("tool") == "get_predictions" and "failed" in t.get("result", "").lower() for t in trace):
        warnings.append("inference_unavailable: findings are preliminary")
    yield _sse("report", {
        "status": "ok",
        "study_id": study_id,
        "model": model_used,
        "report": report,
        "warnings": warnings,
        "tool_trace": trace,
    })


@router.get("/{study_id}/report")
def get_report(study_id: str, session_id: str = "default"):
    from app.services.agent.store import load_report

    entry = load_report(study_id, session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="No completed report for this session yet.")
    return {"status": "ok", "study_id": study_id, "report": entry["report"]}
