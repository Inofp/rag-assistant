from __future__ import annotations

from dataclasses import dataclass
from typing import List
from api.rag.retriever import RetrievedChunk


@dataclass
class GateDecision:
    ok: bool
    reason: str


def decide(chunks: List[RetrievedChunk], min_score: float) -> GateDecision:
    if not chunks:
        return GateDecision(ok=False, reason="empty")
    best = max(c.score for c in chunks)
    if best < float(min_score):
        return GateDecision(ok=False, reason="low_score")
    return GateDecision(ok=True, reason="ok")