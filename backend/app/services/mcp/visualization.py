"""Visualization MCP server: find series by orientation for the viewer."""

import logging

from mcp.server.mcpserver import MCPServer

from app.services.dicom.service import study_manager

logger = logging.getLogger(__name__)


def find_series_text(study_id: str, orientation: str, slice_number=None) -> str:
    """Find series in a study matching an anatomical orientation.

    slice_number is passed straight through for the viewer (parsed back out
    downstream); the tool itself matches on orientation only.
    """
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
    text = f"Series with '{orientation}' orientation in study '{study_id}':\n" + "\n".join(matching)
    if slice_number is not None:
        text += f"\nRequested slice: {slice_number}"
    return text


def create_vis_mcp_server() -> MCPServer:
    """Create and initialize the Visualization MCP server."""
    server = MCPServer(
        name="knee-vis-mcp",
        title="Knee MRI Visualization MCP Server",
        description="Helps the agent find and display knee MRI series by orientation",
        version="1.0.0",
    )

    @server.tool()
    def find_series(study_id: str, orientation: str, slice_number: int | None = None) -> str:
        """Find series in a knee MRI study by orientation (Sagittal, Coronal, Axial).

        Pass slice_number through whenever the user names a specific slice.
        """
        return find_series_text(study_id, orientation, slice_number)

    return server
