from pydantic import BaseModel
from typing import Dict, Any


class EvalReport(BaseModel):
    retrieval: Dict[str, Any]
    latency: Dict[str, Any]