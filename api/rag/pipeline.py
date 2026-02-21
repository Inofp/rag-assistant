from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Dict, List

import httpx

from api.settings import Settings
from api.app.metrics import Metrics
from api.schemas.chat import ChatRequest, ChatResponse, Source
from api.schemas.search import SearchRequest, SearchResponse
from api.schemas.ingest import IngestRequest, IngestResponse
from api.stores.redis_store import RedisConversationStore
from api.stores.qdrant_store import QdrantStore
from api.rag.embeddings import EmbeddingClient
from api.intent.routing import IntentRouter
from api.rag.chunking import build_chunks
from api.rag.retriever import Retriever
from api.rag.gating import decide
from api.rag.prompts import SYSTEM_RAG, build_context, build_messages

from qdrant_client.http import models as qm


log = logging.getLogger("api")


class ChatPipeline:
    def __init__(
        self,
        settings: Settings,
        metrics: Metrics,
        conv_store: RedisConversationStore,
        qdrant: QdrantStore,
        embedder: EmbeddingClient,
        intent_router: IntentRouter,
    ) -> None:
        self.s = settings
        self.m = metrics
        self.conv = conv_store
        self.qdrant = qdrant
        self.embedder = embedder
        self.intent_router = intent_router
        self.retriever = Retriever(
            qdrant=qdrant,
            embedder=embedder,
            cache=conv_store,
            cache_ttl_seconds=settings.retriever_cache_ttl_seconds,
            top_k=settings.retriever_top_k,
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = perf_counter()
        intent = self.intent_router.route(req.message)
        self.m.inc(f"intent_{intent}")

        self.conv.append(req.conversation_id, "user", req.message)
        history = self.conv.history(req.conversation_id)
        summary = self.conv.get_summary(req.conversation_id)

        if intent == "CTA":
            answer = self._cta_answer()
            self.conv.append(req.conversation_id, "assistant", answer)
            self.m.observe_ms("chat_total", (perf_counter() - t0) * 1000.0)
            return ChatResponse(conversation_id=req.conversation_id, intent=intent, answer=answer, sources=[], debug={"mode": "cta"})

        if intent == "OPERATOR":
            answer = self._operator_answer()
            self.conv.append(req.conversation_id, "assistant", answer)
            self.m.observe_ms("chat_total", (perf_counter() - t0) * 1000.0)
            return ChatResponse(conversation_id=req.conversation_id, intent=intent, answer=answer, sources=[], debug={"mode": "operator"})

        r0 = perf_counter()
        chunks = self.retriever.retrieve(req.message)
        self.m.observe_ms("retrieve_ms", (perf_counter() - r0) * 1000.0)

        gate = decide(chunks, self.s.retriever_min_score)
        if not gate.ok:
            self.m.inc("fallback")
            answer = self._fallback_answer(gate.reason)
            self.conv.append(req.conversation_id, "assistant", answer)
            self.m.observe_ms("chat_total", (perf_counter() - t0) * 1000.0)
            return ChatResponse(
                conversation_id=req.conversation_id,
                intent="RAG",
                answer=answer,
                sources=[],
                debug={"mode": "fallback", "reason": gate.reason},
            )

        context = build_context(chunks)
        msgs = build_messages(SYSTEM_RAG, history, req.message, context, summary)

        g0 = perf_counter()
        answer = await self._generate(msgs)
        self.m.observe_ms("generate_ms", (perf_counter() - g0) * 1000.0)

        sources = [Source(doc_id=c.doc_id, title=c.title, score=c.score, source_path=c.source_path) for c in chunks]
        self.conv.append(req.conversation_id, "assistant", answer)

        self.m.observe_ms("chat_total", (perf_counter() - t0) * 1000.0)
        log.info("chat", extra={"extra": {"conversation_id": req.conversation_id, "intent": intent, "sources": len(sources)}})
        return ChatResponse(
            conversation_id=req.conversation_id,
            intent="RAG",
            answer=answer,
            sources=sources,
            debug={"mode": "rag", "gate": gate.reason},
        )

    async def search(self, req: SearchRequest) -> SearchResponse:
        chunks = self.retriever.retrieve(req.query, top_k=req.top_k)
        results = [Source(doc_id=c.doc_id, title=c.title, score=c.score, source_path=c.source_path) for c in chunks]
        return SearchResponse(query=req.query, results=results)

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        chunks = build_chunks(req.docs_path, limit=req.limit)
        dim = self.embedder.dim()
        self.qdrant.ensure_collection(vector_size=dim, recreate=req.recreate)

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        points = []
        for idx, (c, v) in enumerate(zip(chunks, vectors)):
            points.append(
                qm.PointStruct(
                    id=idx,
                    vector=v,
                    payload={
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "source_path": c.source_path,
                        "text": c.text,
                    },
                )
            )
        if points:
            self.qdrant.upsert(points)
        return IngestResponse(indexed_chunks=len(points), collection=self.qdrant.collection)

    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        if not self.s.llm_api_key:
            return "LLM_API_KEY не задан."
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"}
        payload: Dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": messages,
            "max_tokens": self.s.llm_max_output_tokens,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=self.s.llm_timeout_seconds) as client:
            r = await client.post(self._llm_url(), headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()

    def _llm_url(self) -> str:
        base = (self.s.llm_base_url or "").rstrip("/")
        if base:
            return f"{base}/chat/completions"
        return "https://api.openai.com/v1/chat/completions"

    def _cta_answer(self) -> str:
        return "Могу подготовить КП. Оставь телефон или email и укажи: объём (тонны), ширина/толщина, сплав/состояние, город доставки."

    def _operator_answer(self) -> str:
        return "Понял. Передам менеджеру. Оставь контакты и кратко опиши задачу (объём, спецификация, сроки)."

    def _fallback_answer(self, reason: str) -> str:
        if reason == "low_score":
            return "Не нашёл точного ответа в базе знаний. Уточни параметры (сплав, толщина, ширина, объём, город), или оставь контакты - менеджер поможет."
        return "Похоже, информации не хватает. Уточни вопрос или оставь контакты для связи."