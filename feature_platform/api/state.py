"""Serving state: a singleton registry, index, cache and metrics collector.

The API process builds these once at startup.  Reloading is explicit (spec /
index changes) so that serving behaviour can never silently change halfway
through a request.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from feature_platform import spec
from feature_platform.online.cache import CachedOnlineStore, DependencyCache
from feature_platform.online.engine import OnlineEngine, OnlineIndex


@dataclass
class LatencyRecorder:
    """Rolling latency percentiles per endpoint (no external deps)."""

    max_samples: int = 20_000
    samples: dict[str, deque] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, name: str, micros: float) -> None:
        with self._lock:
            dq = self.samples.setdefault(name, deque(maxlen=self.max_samples))
            dq.append(micros)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            out = {}
            for name, dq in self.samples.items():
                if not dq:
                    continue
                arr = np.fromiter(dq, dtype=float)
                out[name] = {
                    "n": int(len(arr)),
                    "p50_us": round(float(np.percentile(arr, 50)), 1),
                    "p95_us": round(float(np.percentile(arr, 95)), 1),
                    "p99_us": round(float(np.percentile(arr, 99)), 1),
                    "max_us": round(float(arr.max()), 1),
                    "mean_us": round(float(arr.mean()), 1),
                }
            return out


class ServiceState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.registry: spec.Registry | None = None
        self.index: OnlineIndex | None = None
        self.engine: OnlineEngine | None = None
        self.store: CachedOnlineStore | None = None
        self.encodings: dict[str, Any] = {}
        self.latency = LatencyRecorder()
        self.started_at = time.time()
        self.requests = 0
        self.unknown_category_events: dict[str, int] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self.registry = spec.load_default()
            self.encodings = spec.load_encodings()
            self.index = OnlineIndex(self.registry)
            self.engine = OnlineEngine(self.registry, self.index, self.encodings)
            self.store = CachedOnlineStore(self.engine, self.registry, DependencyCache())
            self.started_at = time.time()

    # -- convenience -----------------------------------------------------
    def score(self, view: str, entity: str, as_of: Any, use_cache: bool = True) -> dict[str, Any]:
        if self.store is None:
            raise RuntimeError("service not initialised")
        t0 = time.perf_counter_ns()
        if use_cache:
            result, hit = self.store.score(view, entity, as_of)
        else:
            result = self.engine.score(view, entity, as_of)  # type: ignore[union-attr]
            hit = False
        us = (time.perf_counter_ns() - t0) / 1000
        self.latency.record("score", us)
        self.requests += 1
        result["cache_hit"] = hit
        result["latency_us"] = round(us, 3)
        for feat, is_unknown in (result.get("unknown_category") or {}).items():
            if is_unknown:
                self.unknown_category_events[feat] = self.unknown_category_events.get(feat, 0) + 1
        return result

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests": self.requests,
            "spec_fingerprint": self.registry.fingerprint() if self.registry else None,
            "views": list(self.registry.views) if self.registry else [],
            "entities_indexed": len(self.index.entities()) if self.index else 0,
            "latency": self.latency.summary(),
            "cache": self.store.cache.snapshot() if self.store else {},
            "unknown_category_events": self.unknown_category_events,
        }


_state: ServiceState | None = None


def get_state() -> ServiceState:
    global _state
    if _state is None:
        _state = ServiceState()
    return _state
