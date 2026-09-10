"""Visualization MCP server: find series by orientation for the viewer."""

import logging

from mcp.server.mcpserver import MCPServer

from app.services.dicom.service import study_manager

logger = logging.getLogger(__name__)


def find_series_text(study_id: str, orientation: str) -> str:
    """Find series in a study matching an anatomical orientation."""
    study = study_manager.get_study(study_id)
    if not study:
        return f"Study '{study_id}' not found"
    orientation_cap = (orientation or "").capitalize()
    matching = [
        f"Series {s.get('series_number')}: {s.get('series_description', 'Unknown')}"
        for s in study.get("series", [])
        if s.get("orientation", "") == orientation_cap
    ]
    if not matching:
        return f"No series found with orientation '{orientation}' in study '{study_id}'."
    return f"Series with '{orientation}' orientation in study '{study_id}':\n" + "\n".join(matching)


def create_vis_mcp_server() -> MCPServer:
    """Create and initialize the Visualization MCP server."""
    server = MCPServer(
        name="knee-vis-mcp",
        title="Knee MRI Visualization MCP Server",
        description="Helps the agent find and display knee MRI series by orientation",
        version="1.0.0",
    )

    @server.tool()
    def find_series(study_id: str, orientation: str) -> str:
        """Find series in a knee MRI study by orientation (Sagittal, Coronal, Axial)."""
        return find_series_text(study_id, orientation)

    return server
