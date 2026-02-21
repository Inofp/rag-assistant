from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingClient:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model_name = model
        self._model = SentenceTransformer(model)

    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> List[List[float]]:
        xs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [x.tolist() for x in xs]