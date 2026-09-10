"""LangChain tool adapters over the MCP tool implementations.

The MCPServer factories in app/services/mcp remain the canonical servers;
these thin @tool wrappers let the LangGraph agent call the same logic
in-process (fast, offline-testable, no transport overhead).
"""

from langchain_core.tools import tool as lc_tool

from app.services.mcp.dicom import get_study_info_text as _study_info
from app.services.mcp.dicom import list_series_text as _list_series
from app.services.mcp.model import get_predictions_text as _predictions
from app.services.mcp.rag import search_knowledge_text as _knowledge
from app.services.mcp.visualization import find_series_text as _find_series


@lc_tool
def get_study_info(study_id: str) -> str:
    """Get structured information about a knee MRI study, including patient demographics."""
    return _study_info(study_id)


@lc_tool
def list_series(study_id: str) -> str:
    """List all series within a knee MRI study with descriptions and orientations."""
    return _list_series(study_id)


@lc_tool
def get_predictions(study_id: str) -> str:
    """Run knee MRI abnormality detection inference on a study (12-class probabilities)."""
    return _predictions(study_id)


@lc_tool
def find_series(study_id: str, orientation: str) -> str:
    """Find series in a knee MRI study by orientation (Sagittal, Coronal, Axial)."""
    return _find_series(study_id, orientation)


@lc_tool
def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base about knee abnormalities and clinical insights."""
    return _knowledge(query)


def make_search_tool(budget: int):
    """Build a budgeted search_medical_knowledge tool (per report run).

    Each factory call gets a fresh counter, so concurrent runs never share
    budget. Calls beyond `budget` return an exhaustion notice instead of
    retrieving — the agent then writes Evidence from what it already has.
    """
    from langchain_core.tools import StructuredTool

    used = 0

    def _run(query: str) -> str:
        nonlocal used
        if used >= budget:
            return (
                f"RAG lookup budget exhausted ({budget} used). "
                "Write Evidence from the passages already retrieved."
            )
        used += 1
        return _knowledge(query)

    return StructuredTool.from_function(
        func=_run,
        name="search_medical_knowledge",
        description="Search the medical knowledge base about knee abnormalities and clinical insights.",
    )


FULL_TOOLS = [get_study_info, list_series, get_predictions, find_series, search_medical_knowledge]

# Phase 2 chat scope: RAG + visualizer + study info. No prediction re-runs.
CHAT_TOOLS = [get_study_info, list_series, find_series, search_medical_knowledge]

REPORT_RAG_BUDGET = 4


def build_tools(chat_only: bool = False, rag_budget: int | None = None):
    """Return the tool list for the report agent (all) or chat agent (restricted).

    Pass rag_budget to swap in a per-run budgeted search tool (report runs).
    """
    tools = list(CHAT_TOOLS) if chat_only else list(FULL_TOOLS)
    if rag_budget is not None:
        tools = [make_search_tool(rag_budget) if t.name == "search_medical_knowledge" else t
                 for t in tools]
    return tools
