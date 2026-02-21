from __future__ import annotations

import json
from typing import List, Dict, Any, Optional
import redis


class RedisConversationStore:
    def __init__(self, redis_url: str, ttl_seconds: int, max_turns: int) -> None:
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = int(ttl_seconds)
        self.max_turns = int(max_turns)

    def _key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:msgs"

    def _sum_key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:summary"

    def append(self, conversation_id: str, role: str, content: str) -> None:
        key = self._key(conversation_id)
        item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        pipe = self.r.pipeline()
        pipe.rpush(key, item)
        pipe.ltrim(key, -self.max_turns * 2, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def history(self, conversation_id: str) -> List[Dict[str, str]]:
        key = self._key(conversation_id)
        xs = self.r.lrange(key, 0, -1) or []
        out = []
        for x in xs:
            try:
                out.append(json.loads(x))
            except Exception:
                continue
        return out

    def set_summary(self, conversation_id: str, summary: str) -> None:
        key = self._sum_key(conversation_id)
        self.r.setex(key, self.ttl_seconds, summary)

    def get_summary(self, conversation_id: str) -> Optional[str]:
        key = self._sum_key(conversation_id)
        v = self.r.get(key)
        return v if v else None