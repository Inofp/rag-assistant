from __future__ import annotations

from typing import List, Dict
from api.rag.retriever import RetrievedChunk


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c.title} ({c.doc_id})\n{c.text}")
    return "\n\n".join(parts).strip()


def build_messages(system: str, history: List[Dict[str, str]], user: str, context: str, summary: str | None) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": system}]
    if summary:
        msgs.append({"role": "system", "content": f"Краткий контекст диалога:\n{summary}".strip()})
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    if context:
        msgs.append({"role": "system", "content": f"Контекст из базы знаний:\n{context}".strip()})
    msgs.append({"role": "user", "content": user})
    return msgs


SYSTEM_RAG = """Ты — ассистент корпоративного сайта. Отвечай кратко и по делу.
Если в контексте нет информации, не выдумывай. Попроси уточнение или предложи оставить контакты.
Всегда добавляй список источников в конце, используя ссылки вида [1], [2] по порядку контекста."""