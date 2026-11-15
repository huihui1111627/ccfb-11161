"""Per-feature online feature cache with dependency-aware invalidation.

Requirement 8: *cache feature computation results; when source data changes,
invalidate only the affected part.*

Cache granularity is one **feature value**, not one feature vector:

* key      ``(spec_fingerprint, view, feature, entity, as_of_bucket)``
* deps     the physical sources that feature alone reads, computed from the
           spec (a derived feature inherits its inputs' sources)
* eviction bumping a source version evicts exactly the entries whose dependency
           set contains that source.  A new loan therefore does not throw away
           a cached ``credit_score`` that only reads the profile table.

Two further properties matter for correctness:

* **TTL buckets** -- ``as_of`` is bucketed by the view TTL, so all requests in
  a window share one entry while an explicit past instant keeps its own.
* **Single-flight** -- concurrent requests for the same key compute once.

A vector score is served by fetching each feature from the cache and filling
only the misses in one engine pass.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable

from feature_platform import spec


@dataclass
class CacheEntry:
    value: Any
    source_versions: dict[str, int]
    created_ns: int
    ttl_seconds: int
    hits: int = 0

    def expired(self, now_ns: int) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (now_ns - self.created_ns) > self.ttl_seconds * 1_000_000_000


class DependencyCache:
    """LRU cache of feature values keyed by (view, feature, entity, bucket)."""

    def __init__(self, max_entries: int = 100_000):
        self.max_entries = max_entries
        self._data: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._source_versions: dict[str, int] = {}
        self._inflight: dict[Hashable, threading.Event] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "computed": 0,
            "invalidated_source_change": 0,
            "invalidated_lru": 0,
            "invalidated_ttl": 0,
            "coalesced": 0,
        }

    # -- versions --------------------------------------------------------
    def bump_source(self, source: str) -> int:
        with self._lock:
            v = self._source_versions.get(source, 0) + 1
            self._source_versions[source] = v
            doomed = [k for k, e in self._data.items() if source in e.source_versions]
            for k in doomed:
                del self._data[k]
            self.stats["invalidated_source_change"] += len(doomed)
            return v

    def bump_sources(self, sources: list[str]) -> dict[str, int]:
        return {s: self.bump_source(s) for s in sources}

    def versions(self) -> dict[str, int]:
        with self._lock:
            return dict(self._source_versions)

    # -- keys ------------------------------------------------------------
    @staticmethod
    def make_key(fingerprint: str, view: str, feature: str, entity: str,
                 as_of_ns: int, ttl_seconds: int) -> Hashable:
        bucket = as_of_ns // (ttl_seconds * 1_000_000_000) if ttl_seconds > 0 else as_of_ns
        return (fingerprint, view, feature, entity, bucket)

    # -- cache primitives ------------------------------------------------
    def get(self, key: Hashable) -> tuple[bool, Any]:
        now = time.time_ns()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return False, None
            if entry.expired(now):
                del self._data[key]
                self.stats["invalidated_ttl"] += 1
                self.stats["misses"] += 1
                return False, None
            entry.hits += 1
            self._data.move_to_end(key)
            self.stats["hits"] += 1
            return True, entry.value

    def put(self, key: Hashable, value: Any, deps: set[str], ttl_seconds: int) -> None:
        with self._lock:
            while len(self._data) >= self.max_entries:
                self._data.popitem(last=False)
                self.stats["invalidated_lru"] += 1
            self._data[key] = CacheEntry(
                value=value,
                source_versions={s: self._source_versions.get(s, 0) for s in deps},
                created_ns=time.time_ns(),
                ttl_seconds=ttl_seconds,
            )

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any],
                       deps: set[str], ttl_seconds: int,
                       single_flight: bool = True) -> tuple[Any, bool]:
        hit, value = self.get(key)
        if hit:
            return value, True
        if not single_flight:
            value = compute()
            self.stats["computed"] += 1
            self.put(key, value, deps, ttl_seconds)
            return value, False
        with self._lock:
            ev = self._inflight.get(key)
            owner = ev is None
            if owner:
                ev = threading.Event()
                self._inflight[key] = ev
        if not owner:
            assert ev is not None
            ev.wait(timeout=10.0)
            self.stats["coalesced"] += 1
            hit, value = self.get(key)
            if hit:
                return value, True
        try:
            value = compute()
            with self._lock:
                self.stats["computed"] += 1
            self.put(key, value, deps, ttl_seconds)
            return value, False
        finally:
            if owner:
                with self._lock:
                    self._inflight.pop(key, None)
                assert ev is not None
                ev.set()

    # -- introspection ---------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.stats["hits"] + self.stats["misses"]
            return {
                **self.stats,
                "entries": len(self._data),
                "hit_rate": (self.stats["hits"] / total) if total else 0.0,
                "source_versions": dict(self._source_versions),
                "max_entries": self.max_entries,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def entries_by_feature(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for (fp, view, feature, entity, bucket) in self._data:
                key = f"{view}.{feature}"
                out[key] = out.get(key, 0) + 1
            return out


class CachedOnlineStore:
    """Serving facade: per-feature cache in front of :class:`OnlineEngine`."""

    def __init__(self, engine, registry: spec.Registry, cache: DependencyCache | None = None):
        self.engine = engine
        self.reg = registry
        self.cache = cache or DependencyCache()
        self.fingerprint = registry.fingerprint()
        # feature -> physical sources it reads, derived from the spec
        self._feature_deps: dict[tuple[str, str], set[str]] = {}
        for view in registry.views.values():
            for f in view.features:
                self._feature_deps[(view.name, f.name)] = registry.dependencies(view, f)

    def feature_deps(self, view: str, feature: str) -> set[str]:
        return self._feature_deps.get((view, feature), set())

    def score(self, view_name: str, entity_id: str, as_of) -> tuple[dict[str, Any], bool]:
        from feature_platform.online.engine import ns_to_ts, parse_as_of

        view = self.reg.views[view_name]
        as_of_ns = parse_as_of(as_of)
        ttl = view.online_ttl_seconds
        values: dict[str, Any] = {}
        hits = 0
        misses = 0
        for f in view.ordered():
            key = self.cache.make_key(self.fingerprint, view_name, f.name, entity_id, as_of_ns, ttl)
            hit, value = self.cache.get(key)
            if hit:
                values[f.name] = value
                hits += 1
            else:
                misses += 1
        if misses:
            computed = self.engine._compute(view, entity_id, as_of_ns, only={
                f.name for f in view.ordered() if f.name not in values
            })
            # fill cache for the misses only
            for name, value in computed.items():
                if name in values:
                    continue
                values[name] = value
                key = self.cache.make_key(self.fingerprint, view_name, name, entity_id, as_of_ns, ttl)
                self.cache.put(key, value, self.feature_deps(view_name, name), ttl)
                with self.cache._lock:
                    self.cache.stats["computed"] += 1
            self._last_raw_categories = getattr(self.engine, "_last_raw_categories", {})
        else:
            self._last_raw_categories = {}

        ordered_values = {f.name: values[f.name] for f in view.features}
        encoded = {}
        raw_categories = {}
        unknown = {}
        for f in view.features:
            if f.encoder is not None:
                encoded[f.name] = ordered_values[f.name]
                raw = self._last_raw_categories.get(f.name)
                raw_categories[f.name] = raw
                unknown[f.name] = raw is not None and raw not in f.encoder.dictionary
        return {
            "entity_id": entity_id,
            "as_of": ns_to_ts(as_of_ns).strftime("%Y-%m-%d %H:%M:%S"),
            "view": view_name,
            "features": ordered_values,
            "encoded": encoded,
            "category_raw": raw_categories,
            "unknown_category": unknown,
            "cache_hit": misses == 0,
            "cache_feature_hits": hits,
            "cache_feature_misses": misses,
        }, misses == 0
