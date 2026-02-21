from pydantic import BaseModel, Field
from typing import List
from api.schemas.chat import Source


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = 6


class SearchResponse(BaseModel):
    query: str
    results: List[Source]