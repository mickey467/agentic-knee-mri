"""Milestone 4 tests: agent tools, demographics gate, report/chat endpoints.

All offline: LLM is stubbed with GenericFakeChatModel; no network, no weights.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import app.services.agent.graph as graph_mod
from app.services.agent.graph import (
    build_chat_graph,
    build_report_graph,
    demographics_gate,
    get_interrupt_values,
)
from app.services.agent.prompts import missing_report_sections
from app.services.agent.store import REPORT_STORE, load_report, save_report
from app.services.agent.tools import build_tools
from app.services.agent.trace import extract_viz_commands, last_ai_text, summarize_trace

REPORT_MD = """## Study Info
Demo study.

## Findings
- Effusion likely.

## Impression
Needs radiologist review.

## Evidence
- Effusion: joint fluid accumulation per knowledge base.
"""


def _tool_call(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class ScriptedChatModel(BaseChatModel):
    """Offline stand-in: cycles scripted responses; tolerates bind_tools."""

    _it: object = PrivateAttr()

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_it", iter(list(script)))

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            message = next(self._it)
        except StopIteration:
            message = AIMessage(content="Script exhausted.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _fake(script):
    return ScriptedChatModel(script)


# ---------------------------------------------------------------------------
# Tool adapters + prompts + trace helpers (no graph needed)
# ---------------------------------------------------------------------------

def test_build_tools_sets():
    full, chat = build_tools(False), build_tools(True)
    assert sorted(t.name for t in full) == sorted(
        ["get_study_info", "list_series", "get_predictions", "find_series", "search_medical_knowledge"]
    )
    assert sorted(t.name for t in chat) == sorted(
        ["get_study_info", "list_series", "find_series", "search_medical_knowledge"]
    )
    assert "get_predictions" not in {t.name for t in chat}


def test_rag_budget_enforced(monkeypatch):
    from app.services.agent.tools import REPORT_RAG_BUDGET, make_search_tool

    calls = []
    monkeypatch.setattr(
        "app.services.agent.tools._knowledge",
        lambda query, k=4: (calls.append(query), f"passage for {query}")[1],
    )
    tool = make_search_tool(REPORT_RAG_BUDGET)
    assert tool.name == "search_medical_knowledge"
    for i in range(REPORT_RAG_BUDGET):
        assert f"passage for q{i}" in tool.invoke({"query": f"q{i}"})
    assert len(calls) == REPORT_RAG_BUDGET
    over = tool.invoke({"query": "one too many"})
    assert "budget exhausted" in over.lower()
    assert len(calls) == REPORT_RAG_BUDGET, "no retrieval past the budget"


def test_rag_budget_fresh_per_factory(monkeypatch):
    from app.services.agent.tools import make_search_tool

    monkeypatch.setattr(
        "app.services.agent.tools._knowledge", lambda query, k=4: "passage"
    )
    assert "passage" in make_search_tool(1).invoke({"query": "a"})
    assert "passage" in make_search_tool(1).invoke({"query": "a"})


def test_build_tools_rag_budget_swaps_search(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.services.agent.tools._knowledge",
        lambda query, k=4: (calls.append(query), "passage")[1],
    )
    budgeted = {t.name: t for t in build_tools(False, rag_budget=2)}
    assert set(budgeted) == {t.name for t in build_tools(False)}
    assert "passage" in budgeted["search_medical_knowledge"].invoke({"query": "x"})
    assert "passage" in budgeted["search_medical_knowledge"].invoke({"query": "y"})
    assert "budget exhausted" in budgeted["search_medical_knowledge"].invoke({"query": "z"}).lower()
    assert calls == ["x", "y"]


def test_tool_adapter_invokes():
    tools = {t.name: t for t in build_tools(False)}
    assert "Study: acl" in tools["get_study_info"].invoke({"study_id": "acl"})
    assert "menisc" in tools["search_medical_knowledge"].invoke({"query": "meniscus"}).lower()
    assert "not found" in tools["get_study_info"].invoke({"study_id": "nope"}).lower()


def test_missing_report_sections():
    assert missing_report_sections(REPORT_MD) == []
    assert "evidence" in missing_report_sections("## Study Info\n## Findings\n## Impression\n")


def test_trace_and_viz_helpers():
    messages = [
        _tool_call("find_series", {"study_id": "acl", "orientation": "Sagittal"}, "c1"),
        ToolMessage(
            content="Series with 'Sagittal' orientation in study 'acl':\nSeries 3: Sag T2 FS",
            name="find_series",
            tool_call_id="c1",
        ),
        AIMessage(content="Here is the sagittal view."),
    ]
    trace = summarize_trace(messages)
    assert trace[0] == {"tool": "find_series", "args": {"study_id": "acl", "orientation": "Sagittal"}}
    assert trace[1]["tool"] == "find_series"
    commands = extract_viz_commands(messages)
    assert commands == [
        {"tool": "find_series", "orientation": "Sagittal",
         "series_number": 3, "series_description": "Sag T2 FS",
         "slice_number": None}
    ]
    assert last_ai_text(messages) == "Here is the sagittal view."
    assert extract_viz_commands([AIMessage(content="no tools here")]) == []


# ---------------------------------------------------------------------------
# Demographics gate
# ---------------------------------------------------------------------------

def test_gate_passes_through_complete_state():
    out = demographics_gate(
        {"messages": [], "study_id": "acl", "patient_age": "45", "patient_sex": "M"}
    )
    assert out["patient_age"] == "45" and out["patient_sex"] == "M"
    # Age came from the user (absent in DICOM) so the model must see it in messages.
    injected = " ".join(getattr(m, "content", "") for m in out.get("messages", []))
    assert "45" in injected and "M" in injected


def test_gate_skips_message_when_dicom_complete(monkeypatch):
    from app.services.dicom import service as svc

    real_get = svc.study_manager.get_study
    monkeypatch.setattr(
        svc.study_manager, "get_study",
        lambda sid: {**(real_get(sid) or {}), "patient_age": "50Y", "patient_sex": "F"},
    )
    out = demographics_gate({"messages": [], "study_id": "acl"})
    assert out == {"patient_age": "50Y", "patient_sex": "F"}


def test_gate_pauses_when_demographics_missing():
    graph = build_report_graph(model=_fake([AIMessage(content="hi")]))
    config = {"configurable": {"thread_id": "gate-test-nope"}}
    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="Generate the KNEE MRI REPORT.")], "study_id": "nope"},
            config,
        )
    )
    pauses = get_interrupt_values(graph, config)
    assert pauses, "expected the demographics gate to interrupt"
    assert pauses[0]["ask"] == ["patient_age", "patient_sex"]


def test_gate_resume_flows_into_report():
    script = [
        _tool_call("get_study_info", {"study_id": "nope"}, "c1"),
        _tool_call("get_predictions", {"study_id": "nope"}, "c2"),
        AIMessage(content=REPORT_MD),
    ]
    graph = build_report_graph(model=_fake(script))
    config = {"configurable": {"thread_id": "gate-resume-nope"}}
    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="Generate the KNEE MRI REPORT.")], "study_id": "nope"},
            config,
        )
    )
    assert get_interrupt_values(graph, config), "expected a pause before resume"
    from langgraph.types import Command

    result = asyncio.run(graph.ainvoke(Command(resume={"patient_age": "50", "patient_sex": "F"}), config))
    assert not get_interrupt_values(graph, config)
    assert "## Evidence" in last_ai_text(result["messages"])
    assert result["patient_age"] == "50" and result["patient_sex"] == "F"


# ---------------------------------------------------------------------------
# Endpoints (stubbed LLM via monkeypatched provider)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _stub_provider(monkeypatch, script):
    # One shared iterator per test so multi-invoke flows (e.g. empty-report
    # retry) progress through the script instead of restarting it.
    shared = iter(list(script))

    def _build():
        model = ScriptedChatModel([])
        object.__setattr__(model, "_it", shared)
        return model

    monkeypatch.setattr(graph_mod, "get_chat_model", _build)


def _study_missing_demographics():
    from app.services.dicom.service import study_manager

    for s in study_manager.get_all_studies_summary():
        if not (s.get("patient_sex") and s.get("patient_age")):
            return s["study_id"]
    pytest.skip("no demo study with missing demographics")


def test_report_pauses_for_demographics(client, monkeypatch):
    _stub_provider(monkeypatch, [AIMessage(content="unreached")])
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    sid = _study_missing_demographics()
    resp = client.post(f"/api/studies/{sid}/report", json={"session_id": "t1"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "paused_for_demographics"
    from app.services.dicom.service import study_manager

    study = study_manager.get_study(sid)
    expected = [
        key
        for key, val in (("patient_age", study.get("patient_age")), ("patient_sex", study.get("patient_sex")))
        if not val
    ]
    assert body["questions"] == expected


def test_report_resume_produces_report(client, monkeypatch):
    sid = _study_missing_demographics()
    _stub_provider(
        monkeypatch,
        [
            _tool_call("get_study_info", {"study_id": sid}, "c1"),
            _tool_call("get_predictions", {"study_id": sid}, "c2"),
            _tool_call("search_medical_knowledge", {"query": "effusion"}, "c3"),
            AIMessage(content=REPORT_MD),
        ],
    )
    paused = client.post(f"/api/studies/{sid}/report", json={"session_id": "t2"})
    assert paused.json()["status"] == "paused_for_demographics"
    resp = client.post(
        f"/api/studies/{sid}/report",
        json={"session_id": "t2", "resume_payload": {"patient_age": "45", "patient_sex": "M"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert "## Evidence" in body["report"]
    assert any(t["tool"] == "get_predictions" for t in body["tool_trace"])
    assert load_report(sid, "t2")["report"] == body["report"]

    got = client.get(f"/api/studies/{sid}/report", params={"session_id": "t2"})
    assert got.status_code == 200 and got.json()["report"] == body["report"]


def test_report_resume_recovers_from_empty_response(client, monkeypatch):
    sid = _study_missing_demographics()
    _stub_provider(monkeypatch, [AIMessage(content=""), AIMessage(content=REPORT_MD)])
    paused = client.post(f"/api/studies/{sid}/report", json={"session_id": "s4"})
    assert paused.json()["status"] == "paused_for_demographics"
    resp = client.post(
        f"/api/studies/{sid}/report",
        json={"session_id": "s4", "resume_payload": {"patient_age": "45", "patient_sex": "M"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert "## Evidence" in body["report"]


def test_report_503_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    resp = client.post("/api/studies/acl/report", json={"session_id": "t3"})
    assert resp.status_code == 503


def test_report_404_unknown_study(client):
    assert client.post("/api/studies/nope/report", json={}).status_code == 404


def test_chat_needs_report_first(client):
    resp = client.post("/api/studies/acl/chat", json={"message": "hi", "session_id": "fresh"})
    assert resp.json()["status"] == "needs_report"


def test_chat_answers_with_viz_commands(client, monkeypatch):
    save_report("acl", "t4", REPORT_MD)
    _stub_provider(
        monkeypatch,
        [
            _tool_call("find_series", {"study_id": "acl", "orientation": "Sagittal"}, "c1"),
            AIMessage(content="Showing the sagittal series."),
        ],
    )
    resp = client.post(
        "/api/studies/acl/chat",
        json={"message": "show me the sagittal view", "session_id": "t4"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["reply"] == "Showing the sagittal series."
    assert body["viz_commands"], "expected viewer commands from the find_series call"
    assert all(c["tool"] == "find_series" for c in body["viz_commands"])


def test_chat_injects_report_on_fresh_thread(client, monkeypatch):
    from app.services.agent.graph import build_chat_graph
    from app.services.agent.store import thread_id

    save_report("acl", "t7", REPORT_MD)
    _stub_provider(monkeypatch, [AIMessage(content="On it.")])
    resp = client.post(
        "/api/studies/acl/chat", json={"message": "summarize", "session_id": "t7"}
    )
    assert resp.json()["status"] == "ok"
    probe = build_chat_graph(model=_fake([AIMessage(content="unused")]))
    snap = probe.get_state({"configurable": {"thread_id": thread_id("acl", "t7")}})
    texts = [getattr(m, "content", "") or "" for m in snap.values.get("messages", [])]
    assert any("Completed KNEE MRI REPORT" in t for t in texts)
    assert any("## Evidence" in t for t in texts)


def test_chat_404_and_422(client):
    assert client.post("/api/studies/nope/chat", json={"message": "hi"}).status_code == 404
    save_report("acl", "t5", REPORT_MD)
    assert client.post("/api/studies/acl/chat", json={"message": "  "}).status_code == 422


def teardown_module():
    REPORT_STORE.clear()


def test_sse_heartbeat_during_quiet_stretch():
    import asyncio

    from app.api.endpoints.reports import _with_heartbeat

    async def slow_gen():
        await asyncio.sleep(0.05)
        yield "event: tool\ndata: {}\n\n"

    async def collect():
        out = []
        async for item in _with_heartbeat(slow_gen(), interval=0.01):
            out.append(item)
        return out

    items = asyncio.run(collect())
    assert any(i.startswith(": ping") for i in items)
    assert items[-1].startswith("event: tool")


def _sse_events(text):
    """Parse an SSE stream into [(event, data)] pairs."""
    import json

    out = []
    for block in text.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if event:
            out.append((event, data))
    return out


def test_report_stream_pauses(client, monkeypatch):
    _stub_provider(monkeypatch, [AIMessage(content="unreached")])
    sid = _study_missing_demographics()
    resp = client.post(f"/api/studies/{sid}/report/stream", json={"session_id": "s1"})
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    events = _sse_events(resp.text)
    paused = [d for e, d in events if e == "paused"]
    assert paused and paused[0]["questions"]
    assert set(paused[0]["questions"]) <= {"patient_age", "patient_sex"}
    assert not [e for e, _ in events if e == "report"]


def test_report_stream_resume_produces_report(client, monkeypatch):
    sid = _study_missing_demographics()
    _stub_provider(
        monkeypatch,
        [
            _tool_call("get_study_info", {"study_id": sid}, "c1"),
            _tool_call("get_predictions", {"study_id": sid}, "c2"),
            AIMessage(content=REPORT_MD),
        ],
    )
    paused = client.post(f"/api/studies/{sid}/report/stream", json={"session_id": "s2"})
    assert any(e == "paused" for e, _ in _sse_events(paused.text))
    resp = client.post(
        f"/api/studies/{sid}/report/stream",
        json={"session_id": "s2", "resume_payload": {"patient_age": "45", "patient_sex": "M"}},
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    tools = [(d["tool"], d["phase"]) for e, d in events if e == "tool"]
    assert ("get_study_info", "start") in tools
    assert ("get_predictions", "end") in tools
    reports = [d for e, d in events if e == "report"]
    assert len(reports) == 1
    assert "## Evidence" in reports[0]["report"]
    assert reports[0]["model"] == "primary"
