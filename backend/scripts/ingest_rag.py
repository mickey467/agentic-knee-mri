"""Ingest knee PDFs into a FAISS store for the RAG retriever.

Usage:
    python scripts/ingest_rag.py --pdf-dir "C:\\path\\to\\pdfs" [--out rag_data] [--batch 32]

Output: <out>/{knee_faiss.index, chunks.jsonl, meta.json}
Duplicate PDFs (identical extracted text) are ingested once.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.rag.chunking import chunk_pdf  # noqa: E402
from app.services.rag.index import save_store  # noqa: E402
from app.services.rag.retriever import DEFAULT_MODEL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knee PDFs into FAISS RAG store.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing .pdf files")
    parser.add_argument("--out", default=str(BACKEND_DIR / "rag_data"), help="Output store directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        raise SystemExit(1)

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedder '{args.model}' (first run downloads weights)...")
    embedder = SentenceTransformer(args.model, device="cpu")

    all_chunks, seen_hashes, skipped = [], set(), []
    t0 = time.time()
    for pdf in pdfs:
        try:
            chunks, digest = chunk_pdf(pdf)
        except Exception as e:
            print(f"  SKIP {pdf.name}: unreadable ({e})")
            skipped.append(pdf.name)
            continue
        if digest in seen_hashes:
            print(f"  SKIP {pdf.name}: duplicate content")
            skipped.append(pdf.name)
            continue
        seen_hashes.add(digest)
        all_chunks.extend(chunks)
        print(f"  {pdf.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("No chunks extracted; nothing to index.")
        raise SystemExit(1)

    print(f"Embedding {len(all_chunks)} chunks (batch={args.batch})...")
    embeddings = np.asarray(
        embedder.encode([c.text for c in all_chunks], batch_size=args.batch,
                        normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )

    from app.services.rag.index import build_index

    index = build_index(embeddings)
    out = Path(args.out)
    save_store(out, index, all_chunks, args.model)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s: {len(all_chunks)} chunks, dim={index.d}, "
          f"{len(pdfs) - len(skipped)}/{len(pdfs)} PDFs -> {out}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
