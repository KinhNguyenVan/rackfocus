"""Đếm latency theo stage. Percentile tính từ reservoir cố định, không giữ vô hạn."""
from __future__ import annotations

import threading
import time
from collections import Counter, deque

import numpy as np


class Metrics:
    """Thread-safe. Giữ tối đa `capacity` mẫu gần nhất cho mỗi stage."""

    def __init__(self, capacity: int = 4096) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, deque] = {}
        self._capacity = capacity
        self._counters: Counter[str] = Counter()
        self._by_strategy: Counter[str] = Counter()
        self._started = time.time()

    def observe(self, stage: str, ms: float) -> None:
        with self._lock:
            dq = self._samples.get(stage)
            if dq is None:
                dq = self._samples[stage] = deque(maxlen=self._capacity)
            dq.append(ms)

    def observe_many(self, timings: dict[str, float]) -> None:
        for stage, ms in timings.items():
            self.observe(stage, ms)

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def incr_strategy(self, strategy: int) -> None:
        with self._lock:
            self._by_strategy[str(strategy)] += 1

    def percentiles(self, stage: str = "total_ms") -> dict[str, float]:
        with self._lock:
            data = list(self._samples.get(stage, ()))
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
        arr = np.asarray(data, dtype=np.float64)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "n": len(data)}

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            strategies = dict(self._by_strategy)
            stages = list(self._samples)
        return {
            "uptime_sec": int(time.time() - self._started),
            "counters": counters,
            "by_strategy": strategies,
            "stages": {s: self.percentiles(s) for s in stages},
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._counters.clear()
            self._by_strategy.clear()


metrics = Metrics()
