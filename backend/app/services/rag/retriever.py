"""Lazy FAISS retriever over the ingested knee PDFs.

Everything is lazy so imports stay cheap and failures degrade gracefully:
callers catch KnowledgeUnavailable and fall back to the mock knowledge base.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = BACKEND_DIR / "rag_data"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class KnowledgeUnavailable(RuntimeError):
    """Raised when the FAISS store or embedder cannot serve a query."""


def data_dir() -> Path:
    return Path(os.environ.get("RAG_DATA_DIR", str(DEFAULT_DATA_DIR)))


def model_name() -> str:
    return os.environ.get("RAG_EMBED_MODEL", DEFAULT_MODEL)


_embedder = None
_store = None  # (index, chunks, meta)


def reset_state() -> None:
    """Drop cached embedder/index (used by tests)."""
    global _embedder, _store
    _embedder = None
    _store = None


def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise KnowledgeUnavailable(f"sentence-transformers not installed: {e}")
        try:
            _embedder = SentenceTransformer(model_name(), device="cpu")
        except Exception as e:
            raise KnowledgeUnavailable(f"Could not load embedder '{model_name()}': {e}")
    return _embedder


def get_store():
    global _store
    if _store is None:
        from app.services.rag.index import load_store

        try:
            _store = load_store(data_dir())
        except Exception as e:
            raise KnowledgeUnavailable(str(e))
        meta_model = _store[2].get("model")
        if meta_model and meta_model != model_name():
            logger.warning("RAG index built with '%s' but runtime model is '%s'",
                           meta_model, model_name())
    return _store


def search_knowledge(query: str, k: int = 4) -> list[dict]:
    """Return top-k passages: [{text, source, page, score}]."""
    if not (query or "").strip():
        raise KnowledgeUnavailable("Empty query.")
    embedder = get_embedder()
    index, chunks, _ = get_store()
    try:
        q = np.asarray(embedder.encode([query], normalize_embeddings=True), dtype=np.float32)
        scores, ids = index.search(q, min(k, index.ntotal))
    except Exception as e:
        raise KnowledgeUnavailable(f"FAISS search failed: {e}")
    results = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        results.append({"text": c["text"], "source": c["source"],
                        "page": c["page"], "score": round(float(score), 4)})
    if not results:
        raise KnowledgeUnavailable("No passages retrieved.")
    return results
