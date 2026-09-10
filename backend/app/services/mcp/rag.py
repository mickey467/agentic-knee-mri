"""Medical RAG MCP server: FAISS retrieval over knee PDFs, mock fallback."""

import logging

from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE = {
    "acl": "The anterior cruciate ligament (ACL) is one of the key stabilizing ligaments of the knee. "
    "ACL tears are common sports injuries that often require surgical reconstruction "
    "followed by extensive rehabilitation.",
    "meniscus": "The menisci are C-shaped cartilage discs that cushion the knee joint. "
    "Medial and lateral meniscus tears are common and can cause pain, swelling, "
    "and locking of the knee joint.",
    "oa": "Osteoarthritis (OA) is a degenerative joint disease characterized by gradual "
    "breakdown of cartilage. Knee OA commonly affects the medial compartment, lateral "
    "compartment, and patellofemoral joint.",
    "effusion": "Joint effusion, also known as 'water on the knee,' refers to excess fluid "
    "accumulation in the knee joint space. It can indicate inflammation, injury, "
    "or underlying pathology.",
    "synovitis": "Synovitis is inflammation of the synovial membrane that lines the joint "
    "cavity. It can cause swelling, pain, and warmth in the affected joint.",
    "baker": "A Baker's cyst (popliteal cyst) is a fluid-filled swelling that causes a bulge "
    "and a feeling of tightness behind the knee. It often accompanies knee joint "
    "issues like arthritis or meniscus tears.",
    "contusion": "A bone contusion is a bruise to the bone that is more severe than a bruise "
    "but less severe than a fracture. It can cause pain, swelling, and tenderness.",
    "fracture": "A knee fracture involves a break in one of the bones forming the knee joint, "
    "including the femur, tibia, patella, or distal femur. Fractures require "
    "immediate medical attention and may need surgical intervention.",
}


def search_knowledge_text(query: str, k: int = 4) -> str:
    """Search the FAISS knee library; fall back to the mock KB when unavailable."""
    try:
        from app.services.rag.retriever import KnowledgeUnavailable, search_knowledge
    except ImportError:
        return _mock_search(query)
    try:
        passages = search_knowledge(query, k=k)
    except KnowledgeUnavailable:
        logger.info("FAISS RAG unavailable; using mock knowledge base.")
        return _mock_search(query)
    lines = [f"Query: '{query}'", "Retrieved from knee PDF library:"]
    for i, p in enumerate(passages, start=1):
        snippet = p["text"][:600]
        lines.append(f"[{i}] {p['source']} p.{p['page']} (score {p['score']}): {snippet}")
    return "\n".join(lines)


def _mock_search(query: str) -> str:
    """Static fallback knowledge base (plain function for testing)."""
    q = (query or "").lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in q:
            return f"Query: '{query}'\nKnowledge: {value}"
    return (
        f"Knowledge search for: '{query}'\n"
        "No specific match found in the mock knowledge base. "
        "Try queries about: ACL, meniscus, OA, effusion, synovitis, Baker's cyst, "
        "contusion, or fracture."
    )


def create_rag_mcp_server() -> MCPServer:
    """Create and initialize the Medical RAG MCP server."""
    server = MCPServer(
        name="knee-rag-mcp",
        title="Knee MRI Medical RAG MCP Server",
        description="Provides medical knowledge and information about knee abnormalities",
        version="1.0.0",
    )

    @server.tool()
    def search_medical_knowledge(query: str) -> str:
        """Search the medical knowledge base for information about knee abnormalities, findings, and clinical insights."""
        return search_knowledge_text(query)

    return server
