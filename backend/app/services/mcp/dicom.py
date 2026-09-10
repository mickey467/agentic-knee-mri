"""DICOM MCP server: study metadata and series listing tools."""

import logging

from mcp.server.mcpserver import MCPServer

from app.services.dicom.service import study_manager

logger = logging.getLogger(__name__)


def get_study_info_text(study_id: str) -> str:
    """Build a concise text summary for a study (plain function for testing)."""
    study = study_manager.get_study(study_id)
    if not study:
        return f"Study '{study_id}' not found"
    patient_name = study.get("patient_name", "Anonymized")
    patient_sex = study.get("patient_sex", "")
    patient_age = study.get("patient_age", "")
    has_demographics = bool(patient_sex and patient_age)
    return (
        f"Study: {study.get('study_id', 'unknown')}\n"
        f"Patient Name: {patient_name}\n"
        f"Patient Sex: {patient_sex or 'Not specified'}\n"
        f"Patient Age: {patient_age or 'Not specified'}\n"
        f"Modality: {study.get('modality', 'MR')}\n"
        f"Body Part: {study.get('body_part', 'KNEE')}\n"
        f"Total Files: {study.get('total_files', 0)}\n"
        f"Series Count: {len(study.get('series', []))}\n"
        f"Has Complete Demographics: {has_demographics}"
    )


def list_series_text(study_id: str) -> str:
    """Build a text listing of the series in a study (plain function for testing)."""
    study = study_manager.get_study(study_id)
    if not study:
        return f"Study '{study_id}' not found"
    series_list = study.get("series", [])
    if not series_list:
        return "No series found in this study."
    lines = [f"Series in study '{study_id}':"]
    for s in series_list:
        desc = s.get("series_description", "Unknown series")
        orient = s.get("orientation", "Unknown")
        slices = s.get("slice_count", 0)
        series_num = s.get("series_number", "N/A")
        lines.append(f"  - Series {series_num}: {desc} ({orient}), {slices} slices")
    return "\n".join(lines)


def create_dicom_mcp_server() -> MCPServer:
    """Create and initialize the DICOM MCP server."""
    server = MCPServer(
        name="knee-dicom-mcp",
        title="Knee MRI DICOM MCP Server",
        description="Provides DICOM metadata and study information for knee MRI studies",
        version="1.0.0",
    )

    @server.tool()
    def get_study_info(study_id: str) -> str:
        """Get structured information about a specific knee MRI study,
        including patient demographics, series overview, and metadata."""
        return get_study_info_text(study_id)

    @server.tool()
    def list_series(study_id: str) -> str:
        """List all series within a knee MRI study with their descriptions and orientations."""
        return list_series_text(study_id)

    return server
