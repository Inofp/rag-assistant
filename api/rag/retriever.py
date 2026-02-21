from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import hashlib
import json
import time

from api.stores.qdrant_store import QdrantStore
from api.stores.redis_store import RedisConversationStore
from api.rag.embeddings import EmbeddingClient


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    score: float
    source_path: str
    text: str


class Retriever:
    def __init__(
        self,
        qdrant: QdrantStore,
        embedder: EmbeddingClient,
        cache: RedisConversationStore,
        cache_ttl_seconds: int,
        top_k: int,
    ) -> None:
        self.qdrant = qdrant
        self.embedder = embedder
        self.cache = cache
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.top_k = int(top_k)

    def _cache_key(self, query: str) -> str:
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        return f"retr:{h}"

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = int(top_k or self.top_k)
        key = self._cache_key(query)

        cached = self.cache.r.get(key)
        if cached:
            try:
                payload = json.loads(cached)
                out = [
                    RetrievedChunk(
                        doc_id=x["doc_id"],
                        title=x["title"],
                        score=float(x["score"]),
                        source_path=x["source_path"],
                        text=x["text"],
                    )
                    for x in payload
                ]
                return out[:k]
            except Exception:
                pass

        vec = self.embedder.embed([query])[0]
        hits = self.qdrant.search(vec, top_k=k)
        out = [
            RetrievedChunk(
                doc_id=h["doc_id"],
                title=h["title"],
                score=float(h["score"]),
                source_path=h["source_path"],
                text=h["text"],
            )
            for h in hits
        ]
        self.cache.r.setex(key, self.cache_ttl_seconds, json.dumps([x.__dict__ for x in out], ensure_ascii=False))
        return out