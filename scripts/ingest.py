import asyncio
from api.app.deps import deps
from api.schemas.ingest import IngestRequest


async def main():
    p = deps.pipeline()
    r = await p.ingest(IngestRequest(docs_path="data/docs", recreate=True))
    print(r.model_dump())


if __name__ == "__main__":
    asyncio.run(main())