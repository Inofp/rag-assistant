from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Any
import threading
import math


@dataclass
class Timer:
    t0: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(perf_counter())

    def ms(self) -> float:
        return (perf_counter() - self.t0) * 1000.0


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._timings_ms: Dict[str, list[float]] = {}

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe_ms(self, name: str, value: float) -> None:
        with self._lock:
            self._timings_ms.setdefault(name, []).append(float(value))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            timings = {}
            for k, xs in self._timings_ms.items():
                if not xs:
                    continue
                ys = sorted(xs)
                def pct(p: float) -> float:
                    if not ys:
                        return math.nan
                    i = int(round((p / 100.0) * (len(ys) - 1)))
                    i = max(0, min(len(ys) - 1, i))
                    return ys[i]
                timings[k] = {"count": len(ys), "p50_ms": pct(50), "p95_ms": pct(95)}
            return {"counters": dict(self._counters), "timings_ms": timings}