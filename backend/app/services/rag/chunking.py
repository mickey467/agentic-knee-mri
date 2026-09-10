"""PDF text extraction and overlap chunking for the knee knowledge base."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs for a PDF. Empty pages are skipped."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # De-hyphenate line breaks: "recon-\nstruction" -> "reconstruction"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_chunks(text: str, source: str, page: int,
                 size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for para in paras:
        if len(para) > size:
            if current:
                chunks.append(Chunk(text=current, source=source, page=page))
                current = ""
            # Hard-split oversized paragraph with overlap
            start = 0
            while start < len(para):
                piece = para[start:start + size]
                chunks.append(Chunk(text=piece, source=source, page=page))
                if start + size >= len(para):
                    break
                start += size - overlap
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > size and current:
            chunks.append(Chunk(text=current, source=source, page=page))
            # Overlap: carry the tail of the finished chunk forward
            current = f"{current[-overlap:]}\n\n{para}" if len(current) > overlap else para
            if len(current) > size:  # pathological; flush and restart
                chunks.append(Chunk(text=current[:size], source=source, page=page))
                current = current[size - overlap:]
        else:
            current = candidate
    if current:
        chunks.append(Chunk(text=current, source=source, page=page))
    return chunks


def chunk_pdf(pdf_path: Path) -> tuple[list[Chunk], str]:
    """Chunk one PDF. Returns (chunks, content_hash) for dedupe."""
    source = pdf_path.name
    full_text_parts = []
    chunks: list[Chunk] = []
    for page_num, text in extract_pages(pdf_path):
        full_text_parts.append(text)
        chunks.extend(split_chunks(text, source, page_num))
    return chunks, content_hash("\n".join(full_text_parts))
