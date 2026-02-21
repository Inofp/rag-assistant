import asyncio
import time
import httpx
import numpy as np


async def run(n: int = 40):
    url = "http://localhost:8000/api/chat"
    lat = []
    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(n):
            t0 = time.perf_counter()
            r = await client.post(url, json={"conversation_id": "load", "message": "Какие условия поставки и сроки?"})
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000.0)
    xs = np.array(lat)
    out = {"n": int(xs.size), "p50_ms": float(np.percentile(xs, 50)), "p95_ms": float(np.percentile(xs, 95))}
    print(out)


if __name__ == "__main__":
    asyncio.run(run())