from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl
from typing import List, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_name: str = "rag-assistant-template"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    cors_origins: str = "http://localhost:3000"

    redis_url: str = "redis://redis:6379/0"
    redis_ttl_seconds: int = 1209600
    chat_max_turns: int = 14

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "kb"

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "openai_compatible"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 18
    llm_max_output_tokens: int = 320

    retriever_top_k: int = 6
    retriever_min_score: float = 0.28
    retriever_cache_ttl_seconds: int = 21600

    rate_limit_rpm: int = 120

    def cors_list(self) -> List[str]:
        raw = (self.cors_origins or "").strip()
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]


settings = Settings()