"""Milestone 3 tests: the 4 MCP servers expose tools and execute independently."""

import asyncio

from app.services.mcp.dicom import (
    create_dicom_mcp_server,
    get_study_info_text,
    list_series_text,
)
from app.services.mcp.model import create_model_mcp_server, get_predictions_text
from app.services.mcp.rag import create_rag_mcp_server, search_knowledge_text
from app.services.mcp.visualization import create_vis_mcp_server, find_series_text


def _tool_names(server):
    return [t.name for t in asyncio.run(server.list_tools())]


def test_dicom_server_tools_registered():
    assert set(_tool_names(create_dicom_mcp_server())) == {"get_study_info", "list_series"}


def test_model_server_tools_registered():
    assert _tool_names(create_model_mcp_server()) == ["get_predictions"]


def test_vis_server_tools_registered():
    assert _tool_names(create_vis_mcp_server()) == ["find_series"]


def test_rag_server_tools_registered():
    assert _tool_names(create_rag_mcp_server()) == ["search_medical_knowledge"]


def test_dicom_tools_execute():
    info = get_study_info_text("acl")
    assert "Study: acl" in info
    assert "Series Count:" in info
    series = list_series_text("acl")
    assert "Series in study 'acl':" in series
    assert get_study_info_text("does_not_exist") == "Study 'does_not_exist' not found"


def test_vis_tool_executes():
    for orientation in ("Sagittal", "Coronal", "Axial"):
        result = find_series_text("acl", orientation)
        assert "acl" in result
    assert "not found" in find_series_text("does_not_exist", "Sagittal").lower()


def test_rag_mock_fallback(monkeypatch, tmp_path):
    """With no ingested store, the RAG tool falls back to the mock KB."""
    from app.services.rag import retriever

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "empty"))
    retriever.reset_state()
    try:
        assert "anterior cruciate ligament" in search_knowledge_text("ACL tear").lower()
        assert "No specific match" in search_knowledge_text("unrelated xyz query")
    finally:
        retriever.reset_state()


def test_rag_faiss_path_when_ingested():
    """Against the real ingested store (skipped when not built)."""
    from app.services.rag import retriever
    from app.services.rag.index import INDEX_FILENAME

    if not (retriever.data_dir() / INDEX_FILENAME).exists():
        import pytest

        pytest.skip("FAISS store not ingested")
    retriever.reset_state()
    try:
        result = search_knowledge_text("ACL tear reconstruction")
        assert "Retrieved from knee PDF library" in result
        assert ".pdf p." in result
    finally:
        retriever.reset_state()


def test_model_tool_handles_missing_weights_gracefully():
    # Offline CI has no transformers weights: must return an error string, not raise.
    result = get_predictions_text("does_not_exist")
    assert isinstance(result, str)
    assert "not found" in result.lower() or "failed" in result.lower()


def test_call_tool_end_to_end_dicom_and_rag():
    async def _run():
        dicom_server = create_dicom_mcp_server()
        result = await dicom_server.call_tool("get_study_info", {"study_id": "acl"})
        assert not result.is_error
        assert "Study: acl" in result.content[0].text

        rag_server = create_rag_mcp_server()
        result = await rag_server.call_tool("search_medical_knowledge", {"query": "meniscus"})
        assert not result.is_error
        assert "menisc" in result.content[0].text.lower()

    asyncio.run(_run())
