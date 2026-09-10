"""Phase 2: free chat over a completed report (RAG + visualizer + study info)."""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.api.endpoints.reports import _config_for

router = APIRouter(prefix="/studies", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@router.post("/{study_id}/chat")
async def chat(study_id: str, body: ChatRequest):
    from app.services.agent.graph import build_chat_graph
    from app.services.agent.provider import LLMNotConfigured
    from app.services.agent.store import load_report
    from app.services.agent.trace import extract_viz_commands, last_ai_text, summarize_trace
    from app.services.dicom.service import study_manager

    if not study_manager.get_study(study_id):
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")

    if not body.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty.")

    if not load_report(study_id, body.session_id):
        return {
            "status": "needs_report",
            "study_id": study_id,
            "reply": "Please generate the report for this study first, then we can discuss it.",
        }

    from app.services.agent.failover import ainvoke_with_failover

    _, config = _config_for(study_id, body.session_id)
    payload = {"messages": [HumanMessage(content=f"(Study context: {study_id})\n{body.message}")]}
    try:
        _, model_used, graph = await ainvoke_with_failover(build_chat_graph, payload, config)
    except LLMNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        from app.services.agent.provider import is_capacity_error

        if is_capacity_error(e):
            raise HTTPException(
                status_code=503,
                detail="Models are overloaded right now (ResourceExhausted). Please retry shortly.",
            )
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    snapshot = graph.get_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    reply = last_ai_text(messages)
    return {
        "status": "ok",
        "study_id": study_id,
        "model": model_used,
        "reply": reply,
        "viz_commands": extract_viz_commands(messages),
        "tool_trace": summarize_trace(messages[-8:]),
    }
