from __future__ import annotations

from functools import lru_cache

from api.settings import settings
from api.app.metrics import Metrics
from api.stores.redis_store import RedisConversationStore
from api.stores.qdrant_store import QdrantStore
from api.rag.embeddings import EmbeddingClient
from api.intent.routing import IntentRouter
from api.rag.pipeline import ChatPipeline


class Deps:
    @lru_cache
    def metrics(self) -> Metrics:
        return Metrics()

    @lru_cache
    def redis(self) -> RedisConversationStore:
        return RedisConversationStore(settings.redis_url, settings.redis_ttl_seconds, settings.chat_max_turns)

    @lru_cache
    def qdrant(self) -> QdrantStore:
        return QdrantStore(settings.qdrant_url, settings.qdrant_collection)

    @lru_cache
    def embedder(self) -> EmbeddingClient:
        return EmbeddingClient(settings.embedding_provider, settings.embedding_model)

    @lru_cache
    def intent_router(self) -> IntentRouter:
        return IntentRouter()

    def pipeline(self) -> ChatPipeline:
        return ChatPipeline(
            settings=settings,
            metrics=self.metrics(),
            conv_store=self.redis(),
            qdrant=self.qdrant(),
            embedder=self.embedder(),
            intent_router=self.intent_router(),
        )


deps = Deps()