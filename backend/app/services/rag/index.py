"""FAISS index build / persist / load for knee knowledge chunks."""

import json
from pathlib import Path

import numpy as np

from app.services.rag.chunking import Chunk

INDEX_FILENAME = "knee_faiss.index"
CHUNKS_FILENAME = "chunks.jsonl"
META_FILENAME = "meta.json"


def build_index(embeddings: np.ndarray):
    """Cosine-similarity index (inner product over L2-normalized vectors)."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_store(data_dir: Path, index, chunks: list[Chunk], model_name: str) -> None:
    import faiss

    data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(data_dir / INDEX_FILENAME))
    with open(data_dir / CHUNKS_FILENAME, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": c.text, "source": c.source, "page": c.page}) + "\n")
    with open(data_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "dim": index.d, "count": len(chunks)}, f)


def load_store(data_dir: Path):
    """Return (index, chunks, meta). Raises FileNotFoundError when not ingested."""
    import faiss

    index_path = data_dir / INDEX_FILENAME
    chunks_path = data_dir / CHUNKS_FILENAME
    meta_path = data_dir / META_FILENAME
    for p in (index_path, chunks_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"RAG store not ingested (missing {p.name} in {data_dir})")
    index = faiss.read_index(str(index_path))
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, chunks, meta
