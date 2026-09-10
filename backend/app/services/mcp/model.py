"""Knee Model MCP server: DINOv2-LoRA + Model G inference tool."""

import logging

from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)


def get_predictions_text(study_id: str) -> str:
    """Run ensemble inference on a study and format the result as text."""
    try:
        from app.services.model.inference import run_inference

        result = run_inference(study_id)
    except FileNotFoundError as e:
        return f"Study '{study_id}' not found or weights missing: {e}"
    except Exception as e:
        return f"Inference failed: {e}"
    probs = result.get("probabilities", {})
    predicted = result.get("predicted_labels", [])
    raw = result.get("raw_probs", [])
    lines = [
        f"Study: {study_id}",
        f"Predicted Labels: {', '.join(predicted) if predicted else 'None above threshold'}",
        f"Raw Probabilities: {raw}",
        "Probabilities:",
    ]
    lines.extend(f"  {label}: {prob}" for label, prob in probs.items())
    return "\n".join(lines)


def create_model_mcp_server() -> MCPServer:
    """Create and initialize the Knee Model MCP server."""
    server = MCPServer(
        name="knee-model-mcp",
        title="Knee MRI Model MCP Server",
        description="Runs DINOv2-LoRA + Model G ensemble inference for knee abnormality detection",
        version="1.0.0",
    )

    @server.tool()
    def get_predictions(study_id: str) -> str:
        """Run knee MRI abnormality detection inference on a study.

        Returns 12-class sigmoid probabilities for: ACL, MCL, Medial Meniscus,
        Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis,
        Baker's cyst, Contusion, Fracture.
        """
        return get_predictions_text(study_id)

    return server
