"""In-process counters + simple latency histogram for /metrics.

Deliberately tiny — no Prometheus client dependency. Just enough to confirm the
gateway is doing work; real observability happens at the platform layer.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total: dict[str, int] = defaultdict(int)
        self.errors_total: dict[str, int] = defaultdict(int)
        self.input_tokens_total: int = 0
        self.output_tokens_total: int = 0
        # Coarse latency buckets (ms): <100, <500, <1000, <5000, <30000, +Inf
        self._buckets_ms = (100, 500, 1000, 5000, 30000)
        self.ttfb_ms_histogram: list[int] = [0] * (len(self._buckets_ms) + 1)
        self.total_ms_histogram: list[int] = [0] * (len(self._buckets_ms) + 1)

    def record_request(
        self,
        *,
        route: str,
        status_code: int,
        ttfb_ms: float | None,
        total_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        key = f"{route}:{status_code}"
        with self._lock:
            self.requests_total[key] += 1
            if status_code >= 400:
                self.errors_total[key] += 1
            self.input_tokens_total += int(input_tokens or 0)
            self.output_tokens_total += int(output_tokens or 0)
            if ttfb_ms is not None:
                self._add_to_histogram(self.ttfb_ms_histogram, ttfb_ms)
            self._add_to_histogram(self.total_ms_histogram, total_ms)

    def _add_to_histogram(self, hist: list[int], value_ms: float) -> None:
        for idx, bound in enumerate(self._buckets_ms):
            if value_ms < bound:
                hist[idx] += 1
                return
        hist[-1] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": dict(self.requests_total),
                "errors_total": dict(self.errors_total),
                "input_tokens_total": self.input_tokens_total,
                "output_tokens_total": self.output_tokens_total,
                "ttfb_ms_histogram": {
                    "buckets_ms": list(self._buckets_ms) + ["+Inf"],
                    "counts": list(self.ttfb_ms_histogram),
                },
                "total_ms_histogram": {
                    "buckets_ms": list(self._buckets_ms) + ["+Inf"],
                    "counts": list(self.total_ms_histogram),
                },
            }


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.first_byte: float | None = None

    def mark_first_byte(self) -> None:
        if self.first_byte is None:
            self.first_byte = time.perf_counter()

    def ttfb_ms(self) -> float | None:
        if self.first_byte is None:
            return None
        return (self.first_byte - self.start) * 1000.0

    def total_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0
