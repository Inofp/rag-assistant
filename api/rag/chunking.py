from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict


@dataclass
class DocChunk:
    doc_id: str
    title: str
    source_path: str
    text: str


def read_markdown_files(root: str, limit: int | None = None) -> List[Path]:
    p = Path(root)
    files = sorted([x for x in p.rglob("*.md") if x.is_file()])
    if limit:
        files = files[: int(limit)]
    return files


def chunk_markdown(text: str, max_chars: int = 900) -> List[str]:
    parts = []
    buf = []
    size = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") and buf:
            parts.append(" ".join(buf).strip())
            buf, size = [], 0
        if size + len(line) + 1 > max_chars and buf:
            parts.append(" ".join(buf).strip())
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        parts.append(" ".join(buf).strip())
    return [p for p in parts if len(p) > 40]


def build_chunks(docs_root: str, limit: int | None = None) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for fp in read_markdown_files(docs_root, limit=limit):
        raw = fp.read_text(encoding="utf-8")
        title = fp.stem.replace("_", " ").strip()
        pieces = chunk_markdown(raw)
        for i, piece in enumerate(pieces):
            doc_id = f"{fp.stem}:{i}"
            chunks.append(DocChunk(doc_id=doc_id, title=title, source_path=str(fp), text=piece))
    return chunks