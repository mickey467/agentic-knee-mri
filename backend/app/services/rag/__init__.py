"""Local FAISS RAG over curated knee MRI PDFs."""

from app.services.rag.retriever import KnowledgeUnavailable, search_knowledge

__all__ = ["KnowledgeUnavailable", "search_knowledge"]
