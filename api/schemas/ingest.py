from pydantic import BaseModel, Field
from typing import Optional


class IngestRequest(BaseModel):
    docs_path: str = Field(default="data/docs")
    recreate: bool = False
    limit: Optional[int] = None


class IngestResponse(BaseModel):
    indexed_chunks: int
    collection: str