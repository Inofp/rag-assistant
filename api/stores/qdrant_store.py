from __future__ import annotations

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class QdrantStore:
    def __init__(self, url: str, collection: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, vector_size: int, recreate: bool) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists and recreate:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def upsert(self, points: List[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for h in hits:
            payload = h.payload or {}
            out.append(
                {
                    "score": float(h.score),
                    "doc_id": str(payload.get("doc_id", "")),
                    "title": str(payload.get("title", "")),
                    "source_path": str(payload.get("source_path", "")),
                    "text": str(payload.get("text", "")),
                }
            )
        return out