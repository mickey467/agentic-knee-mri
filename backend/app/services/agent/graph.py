"""LangGraph wiring: demographics gate + report/chat ReAct agents."""

from typing import NotRequired

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import interrupt

from app.services.agent.prompts import CHAT_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT
from app.services.agent.provider import get_chat_model
from app.services.agent.store import CHAT_CHECKPOINTER, REPORT_CHECKPOINTER
from app.services.agent.tools import REPORT_RAG_BUDGET, build_tools


class AgentState(MessagesState):
    study_id: NotRequired[str]
    patient_age: NotRequired[str]
    patient_sex: NotRequired[str]


def _clean(value) -> str:
    return str(value).strip() if value is not None else ""


def demographics_gate(state: AgentState) -> dict:
    """Ensure age/sex are known before the report agent runs.

    Precedence: values already in state > DICOM headers > user answer on resume >
    "Not provided" (never blocks forever; single pass, no re-interrupt loop).

    Values the model cannot see in DICOM headers (user-supplied on resume) are
    injected as a message — otherwise the agent writes "Not provided" despite
    the user having answered.
    """
    from app.services.dicom.service import study_manager

    study = study_manager.get_study(state.get("study_id") or "")
    dicom_age = _clean((study or {}).get("patient_age"))
    dicom_sex = _clean((study or {}).get("patient_sex"))
    age = _clean(state.get("patient_age")) or dicom_age
    sex = _clean(state.get("patient_sex")) or dicom_sex
    if age and sex:
        if age != dicom_age or sex != dicom_sex:
            return {
                "patient_age": age,
                "patient_sex": sex,
                "messages": [_demographics_message(age, sex)],
            }
        return {"patient_age": age, "patient_sex": sex}

    missing = [k for k, v in (("patient_age", age), ("patient_sex", sex)) if not v]
    answers = interrupt({"ask": missing, "study_id": state.get("study_id")})
    answers = answers if isinstance(answers, dict) else {}

    def _pick(key: str, fallback: str) -> str:
        v = _clean(answers.get(key))
        if (not v or v.lower() in {"unknown", "n/a", "none", "not provided"}) and not fallback:
            return "Not provided"
        return v or fallback or "Not provided"

    final_age = _pick("patient_age", age)
    final_sex = _pick("patient_sex", sex)
    update: dict = {"patient_age": final_age, "patient_sex": final_sex}
    if final_age != dicom_age or final_sex != dicom_sex:
        update["messages"] = [_demographics_message(final_age, final_sex)]
    return update


def _demographics_message(age: str, sex: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(
        content=(
            "Patient demographics for this report (user-provided, authoritative — "
            f"use these exact values in Study Info): age {age}, sex {sex}."
        )
    )


def build_report_graph(model=None):
    """Report agent: demographics gate → full-tool ReAct agent. Shared checkpointer.

    The inner agent only sees messages (study binding travels in the task
    message), so its default state schema applies; the parent graph keeps the
    study/demographics fields.
    """
    model = model or get_chat_model()
    inner = create_agent(
        model,
        tools=build_tools(chat_only=False, rag_budget=REPORT_RAG_BUDGET),
        system_prompt=REPORT_SYSTEM_PROMPT,
    )

    async def agent_node(state: AgentState, config) -> dict:
        result = await inner.ainvoke({"messages": state.get("messages") or []}, config)
        seen = len(state.get("messages") or [])
        return {"messages": result["messages"][seen:]}

    builder = StateGraph(AgentState)
    builder.add_node("demographics_gate", demographics_gate)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "demographics_gate")
    builder.add_edge("demographics_gate", "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=REPORT_CHECKPOINTER)


def build_chat_graph(model=None):
    """Phase-2 chat agent: restricted tools (RAG + visualizer + study info)."""
    model = model or get_chat_model()
    return create_agent(
        model,
        tools=build_tools(chat_only=True),
        system_prompt=CHAT_SYSTEM_PROMPT,
        checkpointer=CHAT_CHECKPOINTER,
    )


def get_interrupt_values(graph, config) -> list:
    """Return pending interrupt values for a thread (empty when the run completed)."""
    snapshot = graph.get_state(config)
    values = []
    for task in snapshot.tasks or []:
        for iv in getattr(task, "interrupts", None) or []:
            values.append(getattr(iv, "value", iv))
    return values
