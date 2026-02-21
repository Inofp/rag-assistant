from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    metadata: Optional[Dict[str, Any]] = None


class Source(BaseModel):
    doc_id: str
    title: str
    score: float
    source_path: str


class ChatResponse(BaseModel):
    conversation_id: str
    intent: str
    answer: str
    sources: List[Source] = []
    debug: Dict[str, Any] = {}