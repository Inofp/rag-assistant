import asyncio
import httpx
import redis
from qdrant_client import QdrantClient

from api.settings import settings


async def check_api():
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("http://localhost:8000/health")
        r.raise_for_status()
        return r.json()


def check_redis():
    r = redis.Redis.from_url(settings.redis_url)
    return r.ping()


def check_qdrant():
    c = QdrantClient(url=settings.qdrant_url)
    return c.get_collections()


async def main():
    print("Checking API...")
    api = await check_api()
    print("API:", api)

    print("Checking Redis...")
    print("Redis:", check_redis())

    print("Checking Qdrant...")
    print("Qdrant:", check_qdrant())

    print("All checks passed")


if __name__ == "__main__":
    asyncio.run(main())