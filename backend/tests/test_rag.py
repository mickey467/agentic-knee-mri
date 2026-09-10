"""RAG tests: chunking, FAISS roundtrip, retriever (stubbed), fallback paths.

All offline: the real SentenceTransformer model is never loaded here
(live ingestion is validated by scripts/ingest_rag.py itself).
"""

import json

import numpy as np
import pytest

from app.services.rag import retriever
from app.services.rag.chunking import Chunk, clean_text, content_hash, split_chunks
from app.services.rag.index import build_index, load_store, save_store


@pytest.fixture(autouse=True)
def _clean_retriever():
    retriever.reset_state()
    yield
    retriever.reset_state()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_clean_text_collapses_whitespace_and_dehyphenates():
    assert clean_text("recon-\nstruction  of   the knee\n\n\n\nACL") == "reconstruction of the knee\n\nACL"


def test_split_chunks_overlap_and_bounds():
    text = "\n\n".join(f"Paragraph {i} " + ("word " * 60) for i in range(6))
    chunks = split_chunks(text, "doc.pdf", 2, size=300, overlap=60)
    assert len(chunks) >= 6
    assert all(len(c.text) <= 300 + 60 for c in chunks)
    assert all(c.source == "doc.pdf" and c.page == 2 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    for i in range(6):
        assert f"Paragraph {i}" in joined


def test_content_hash_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


# ---------------------------------------------------------------------------
# Index roundtrip
# ---------------------------------------------------------------------------

def test_build_save_load_roundtrip(tmp_path):
    chunks = [
        Chunk(text="acl ligament tear", source="a.pdf", page=1),
        Chunk(text="meniscus cartilage", source="a.pdf", page=2),
        Chunk(text="fracture bone", source="b.pdf", page=1),
    ]
    vecs = np.eye(3, dtype=np.float32)  # already unit norm
    index = build_index(vecs)
    save_store(tmp_path, index, chunks, "stub-model")
    index2, loaded, meta = load_store(tmp_path)
    assert meta == {"model": "stub-model", "dim": 3, "count": 3}
    assert [c["text"] for c in loaded] == [c.text for c in chunks]
    scores, ids = index2.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), 1)
    assert ids[0][0] == 0 and scores[0][0] == pytest.approx(1.0)


def test_load_store_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_store(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Retriever with stubbed embedder + tiny store
# ---------------------------------------------------------------------------

class _KeywordEmbedder:
    """Deterministic 3-d embeddings so nearest-neighbor ranking is exact."""

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            t = t.lower()
            v = np.array([
                1.0 if "acl" in t else 0.0,
                1.0 if "meniscus" in t else 0.0,
                1.0 if "fracture" in t else 0.0,
            ], dtype=np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v) or 1.0
                v = v / n
            vecs.append(v)
        return np.stack(vecs)


def _make_store(tmp_path):
    chunks = [
        Chunk(text="acl ligament tear reconstruction", source="acl.pdf", page=1),
        Chunk(text="meniscus cartilage tear", source="men.pdf", page=3),
        Chunk(text="fracture bone break", source="frac.pdf", page=2),
    ]
    emb = _KeywordEmbedder()
    save_store(tmp_path, build_index(emb.encode([c.text for c in chunks])), chunks, "stub")
    return tmp_path


def test_search_returns_ranked_cited_passages(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DATA_DIR", str(_make_store(tmp_path)))
    monkeypatch.setattr(retriever, "get_embedder", lambda: _KeywordEmbedder())
    hits = retriever.search_knowledge("acl tear symptoms", k=2)
    assert hits[0]["source"] == "acl.pdf" and hits[0]["page"] == 1
    assert hits[0]["score"] > hits[1]["score"]
    assert len(hits) == 2


def test_search_unavailable_without_store(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(retriever.KnowledgeUnavailable):
        retriever.search_knowledge("acl")


def test_search_empty_query_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DATA_DIR", str(_make_store(tmp_path)))
    monkeypatch.setattr(retriever, "get_embedder", lambda: _KeywordEmbedder())
    with pytest.raises(retriever.KnowledgeUnavailable):
        retriever.search_knowledge("   ")


def test_model_mismatch_warns_but_serves(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("RAG_DATA_DIR", str(_make_store(tmp_path)))
    monkeypatch.setenv("RAG_EMBED_MODEL", "other-model")
    monkeypatch.setattr(retriever, "get_embedder", lambda: _KeywordEmbedder())
    with caplog.at_level("WARNING"):
        hits = retriever.search_knowledge("fracture")
    assert hits[0]["source"] == "frac.pdf"
    assert "built with 'stub'" in caplog.text


# ---------------------------------------------------------------------------
# Live store (only when ingested; exercises the real embedder download path
# separately from CI — ingestion itself is the live validation)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_live_store_answers_acl():
    from app.services.rag.index import INDEX_FILENAME

    if not (retriever.data_dir() / INDEX_FILENAME).exists():
        pytest.skip("FAISS store not ingested")
    hits = retriever.search_knowledge("ACL tear reconstruction", k=2)
    assert hits and "acl" in hits[0]["source"].lower()
    assert json.dumps(hits)  # JSON-serializable for the API
