from fastapi import APIRouter, Depends

from api.schemas.chat import ChatRequest, ChatResponse
from api.schemas.search import SearchRequest, SearchResponse
from api.schemas.ingest import IngestRequest, IngestResponse
from api.app.deps import deps
from api.rag.pipeline import ChatPipeline

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, p: ChatPipeline = Depends(deps.pipeline)):
    return await p.chat(req)


@router.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest, p: ChatPipeline = Depends(deps.pipeline)):
    return await p.search(req)


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, p: ChatPipeline = Depends(deps.pipeline)):
    return await p.ingest(req)