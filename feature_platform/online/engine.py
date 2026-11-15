"""In-memory online feature store.

Loads the same source tables into columnar NumPy indexes at startup and
evaluates the *same* declarative spec at request time from a single user's
perspective.  The point-in-time predicates are re-implemented here in Python
deliberately independently of the SQL compiler -- agreement between the two is
the invariant the parity audit checks, so sharing code would make the test
vacuous.

Index layout (per entity):

    entity -> {
        "applications": {"t": sorted event_time (int64 ns), "rows": dict of arrays},
        ...
    }

Queries use ``np.searchsorted`` for the time range, then vectorised masks for
knowledge_time / valid_until / dedupe, then the aggregate.  Single-user
latency is a few hundred microseconds.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from feature_platform import config, expr, spec


# --------------------------------------------------------------------------
# Time helpers -- every time value inside the online engine is int64 nanoseconds
# --------------------------------------------------------------------------

_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")


_NULL_NS = np.iinfo(np.int64).min   # numpy renders NaT as INT64_MIN in ns view


def to_ns(values) -> np.ndarray:
    """Convert naive timestamps / strings / datetimes to int64 nanoseconds.

    NaT becomes ``_NULL_NS`` (max int64) so that all comparisons and masks stay
    integer/vectorised.  Callers that need "is this NULL" use ``_is_null_ns``.
    """
    if isinstance(values, np.ndarray) and values.dtype == np.int64:
        return values
    arr = pd.to_datetime(pd.Series(values) if not isinstance(values, pd.Series) else values,
                         errors="coerce", utc=False)
    out = arr.to_numpy(dtype="datetime64[ns]")
    return out.view("int64").astype(np.int64)


def _is_null_ns(arr: np.ndarray) -> np.ndarray:
    return arr == _NULL_NS


def ns_to_ts(ns: int) -> pd.Timestamp:
    """Exact Timestamp from int nanoseconds (no precision loss / warnings)."""
    return pd.Timestamp(np.datetime64(int(ns), "ns"))


def parse_as_of(value: Any) -> int:
    if value is None:
        raise ValueError("as_of required")
    if isinstance(value, (int, np.integer)):
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


# --------------------------------------------------------------------------
# Entity index
# --------------------------------------------------------------------------

@dataclass
class EntityFrame:
    """Columnar block for one entity in one source."""
    event_ns: np.ndarray
    knowledge_ns: np.ndarray
    valid_ns: np.ndarray | None
    columns: dict[str, np.ndarray]
    order: np.ndarray            # indices sorted by event_ns
    dedupe_key: np.ndarray | None = None
    dedupe_order_ns: np.ndarray | None = None
    deferred_known_ns: dict[str, np.ndarray] | None = None
    deferred_cleared_ns: dict[str, np.ndarray] | None = None

    def slice_between(self, lo: int, hi: int) -> np.ndarray:
        """Indices with lo <= event_ns < hi, in event order."""
        a = np.searchsorted(self.event_ns, lo, side="left")
        b = np.searchsorted(self.event_ns, hi, side="left")
        return self.order[a:b]


class OnlineIndex:
    """All sources, indexed by entity_id, loaded from DuckDB once."""

    def __init__(self, registry: spec.Registry, con=None):
        self.reg = registry
        self.by_source: dict[str, dict[str, EntityFrame]] = {}
        self.loaded_at: dict[str, int] = {}
        self._lock = threading.RLock()
        self._load(con)

    # -- loading ---------------------------------------------------------
    def _load(self, con) -> None:
        import duckdb

        own = con is None
        if own:
            con = duckdb.connect(str(config.DB_PATH), read_only=True)
        try:
            for name, src in self.reg.sources.items():
                df = con.execute(f"SELECT * FROM {src.relation_sql}").fetchdf()
                self._index_source(name, src, df)
                self.loaded_at[name] = int(pd.Timestamp.utcnow().value)
        finally:
            if own:
                con.close()

    def _index_source(self, name: str, src: spec.SourceSpec, df: pd.DataFrame) -> None:
        if df.empty:
            self.by_source[name] = {}
            return
        # normalise NULLs to Python None (pandas would otherwise re-introduce NaT)
        for c in df.columns:
            col = df[c]
            if col.dtype == object:
                df[c] = col.where(col.notna(), None)
            elif str(col.dtype).startswith("datetime"):
                df[c] = pd.Series([None if pd.isna(v) else v for v in col], index=col.index, dtype=object)
        event_ns = to_ns(df[src.event_time])
        knowledge_ns = to_ns(df[src.knowledge_time])
        valid_ns = to_ns(df[src.valid_until]) if src.valid_until else None

        deferred_known: dict[str, np.ndarray] = {}
        deferred_cleared: dict[str, np.ndarray] = {}
        for col, dspec in (src.deferred or {}).items():
            if col in df.columns and dspec.known_at in df.columns:
                deferred_known[col] = to_ns(df[dspec.known_at])
                if dspec.cleared_at and dspec.cleared_at in df.columns:
                    deferred_cleared[col] = to_ns(df[dspec.cleared_at])

        dedupe_key = None
        dedupe_order = None
        if src.dedupe_key and src.dedupe_key in df.columns:
            dedupe_key = df[src.dedupe_key].to_numpy(dtype=object)
            dedupe_order = to_ns(df[src.dedupe_order]) if src.dedupe_order else knowledge_ns

        cols = {c: df[c].to_numpy(dtype=object) for c in df.columns}
        entity_vals = df[src.entity_key].to_numpy(dtype=object)

        # per-entity frames are sorted by event time individually below
        per_entity: dict[str, EntityFrame] = {}
        pos = pd.Series(np.arange(len(entity_vals))).groupby(entity_vals).apply(lambda s: s.to_numpy())
        for entity, idx in pos.items():
            ev = event_ns[idx]
            local_order = np.argsort(ev, kind="stable")
            per_entity[str(entity)] = EntityFrame(
                event_ns=ev[local_order],
                knowledge_ns=knowledge_ns[idx][local_order],
                valid_ns=valid_ns[idx][local_order] if valid_ns is not None else None,
                columns={c: cols[c][idx][local_order] for c in cols},
                order=np.arange(len(idx)),
                dedupe_key=dedupe_key[idx][local_order] if dedupe_key is not None else None,
                dedupe_order_ns=dedupe_order[idx][local_order] if dedupe_order is not None else None,
                deferred_known_ns={k: v[idx][local_order] for k, v in deferred_known.items()},
                deferred_cleared_ns={k: v[idx][local_order] for k, v in deferred_cleared.items()},
            )
        self.by_source[name] = per_entity

    # -- querying --------------------------------------------------------
    def get(self, source: str, entity_id: str) -> EntityFrame | None:
        return self.by_source.get(source, {}).get(entity_id)

    def entities(self, source: str = "applications") -> list[str]:
        return list(self.by_source.get(source, {}).keys())

    def entities_as_of(self, as_of_ns: int, limit: int) -> list[str]:
        """Entities that had at least one application ingested by as_of."""
        frame_map = self.by_source.get("applications", {})
        out = []
        for ent, fr in frame_map.items():
            k = fr.knowledge_ns
            if len(k) and k.min() <= as_of_ns:
                out.append(ent)
                if len(out) >= limit:
                    break
        return out

    def snapshot_versions(self) -> dict[str, int]:
        return dict(self.loaded_at)


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------

class OnlineEngine:
    def __init__(self, registry: spec.Registry, index: OnlineIndex | None = None,
                 encodings: dict[str, Any] | None = None):
        self.reg = registry
        self.index = index or OnlineIndex(registry)
        self.encodings = encodings if encodings is not None else spec.load_encodings()
        self._lock = threading.RLock()
        self._last_raw_categories: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def score(self, view_name: str, entity_id: str, as_of: Any) -> dict[str, Any]:
        view = self.reg.views[view_name]
        as_of_ns = parse_as_of(as_of)
        start = time.perf_counter_ns()
        values = self._compute(view, entity_id, as_of_ns)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        encoded = {}
        raw_categories = {}
        for f in view.features:
            if f.encoder is not None:
                encoded[f.name] = values[f.name]
                raw_categories[f.name] = self._last_raw_categories.get(f.name)
        return {
            "entity_id": entity_id,
            "as_of": ns_to_ts(as_of_ns).strftime("%Y-%m-%d %H:%M:%S"),
            "view": view_name,
            "features": values,
            "encoded": encoded,
            "category_raw": raw_categories,
            "unknown_category": {
                f.name: raw_categories[f.name] is not None
                and raw_categories[f.name] not in f.encoder.dictionary  # type: ignore[union-attr]
                for f in view.features if f.encoder is not None
            },
            "latency_us": round(elapsed_us, 3),
        }

    def score_many(self, view_name: str, requests: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        return [self.score(view_name, e, a) for e, a in requests]

    # ------------------------------------------------------------------
    def _compute(self, view: spec.ViewSpec, entity_id: str, as_of_ns: int,
                 only: set[str] | None = None,
                 prefilled: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate a view's features in dependency order.

        ``only`` restricts evaluation to a subset (plus whatever those depend
        on); ``prefilled`` supplies already-computed values (used by the
        per-feature cache and by tests).
        """
        ctx = {"as_of": ns_to_ts(as_of_ns), "entity_id": entity_id}
        values: dict[str, Any] = dict(prefilled or {})

        # expand `only` with the transitive feature dependencies so derived
        # features always find their inputs
        needed = set(only) if only is not None else None
        if needed is not None:
            by_name = view.by_name()
            stack = list(needed)
            while stack:
                name = stack.pop()
                f = by_name.get(name)
                if f is None or f.kind != "derived":
                    continue
                for ref in expr.referenced_columns(f.expr_node):
                    if ref in by_name and ref not in needed:
                        needed.add(ref)
                        stack.append(ref)

        for f in view.ordered():
            if needed is not None and f.name not in needed:
                continue
            if f.name in values:
                continue
            if f.kind == "window":
                raw = self._window(f, entity_id, as_of_ns, ctx)
            elif f.kind == "asof_attr":
                raw = self._asof_attr(f, entity_id, as_of_ns, ctx)
            elif f.kind == "static_attr":
                raw = self._static_attr(f, entity_id, as_of_ns, ctx)
            elif f.kind == "constant":
                raw = f.value
            elif f.kind == "derived":
                raw = self._derived(f, values, ctx)
            else:  # pragma: no cover
                raw = None
            value = self._finish(f, raw)
            if f.encoder is not None:
                # parity compares encoded codes; expose the raw category too
                self._last_raw_categories[f.name] = raw
                value = self._encode(view.name, f, raw)
            values[f.name] = value
        return values

    # ------------------------------------------------------------------
    def feature(self, view_name: str, feature: str, entity_id: str, as_of: Any) -> Any:
        """Compute a single feature at an instant (used by the feature cache)."""
        view = self.reg.views[view_name]
        as_of_ns = parse_as_of(as_of)
        values = self._compute(view, entity_id, as_of_ns, only={feature})
        return values.get(feature)

    def _finish(self, f: spec.FeatureSpec, value: Any) -> Any:
        if f.encoder is not None:
            return value  # raw category; encoded with the frozen dictionary below
        if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            value = pd.Timestamp(value).to_pydatetime()
        if value is None:
            if f.fill is not None:
                value = f.fill
            return _cast_scalar(value, f.type)
        return _cast_scalar(value, f.type)

    def _encode(self, view_name: str, f: spec.FeatureSpec, value: Any) -> int:
        key = f"{view_name}.{f.name}"
        enc = self.encodings.get("encodings", {}).get(key)
        if enc is None:
            return f.encoder.code_for(value)  # type: ignore[union-attr]
        if value is None:
            return int(enc["default_code"])
        return int(enc["mapping"].get(str(value), enc["default_code"]))

    def _raw_category(self, view_name: str, f: spec.FeatureSpec, code: int) -> str | None:
        key = f"{view_name}.{f.name}"
        enc = self.encodings.get("encodings", {}).get(key)
        if enc is None:
            return None
        for cat, c in enc["mapping"].items():
            if int(c) == int(code):
                return cat
        return None

    # -- source helpers --------------------------------------------------
    def _rows(self, source: str, entity_id: str, as_of_ns: int,
              lo_ns: int | None = None) -> tuple[EntityFrame, np.ndarray]:
        fr = self.index.get(source, entity_id)
        if fr is None or len(fr.event_ns) == 0:
            empty = np.array([], dtype=np.int64)
            return EntityFrame(empty, empty, None, {}, empty), np.array([], dtype=np.int64)
        lo = lo_ns if lo_ns is not None else np.iinfo(np.int64).min
        idx = fr.slice_between(lo, as_of_ns)
        return fr, idx

    def _visible(self, src: spec.SourceSpec, fr: EntityFrame, idx: np.ndarray, as_of_ns: int) -> np.ndarray:
        """Apply the point-in-time guards; returns a boolean mask over idx."""
        if len(idx) == 0:
            return np.zeros(0, dtype=bool)
        mask = fr.event_ns[idx] < as_of_ns
        mask &= fr.knowledge_ns[idx] <= as_of_ns
        if fr.valid_ns is not None:
            v = fr.valid_ns[idx]
            mask &= _is_null_ns(v) | (v > as_of_ns)
        return mask

    def _dedupe_mask(self, fr: EntityFrame, idx: np.ndarray, as_of_ns: int) -> np.ndarray:
        """Keep, per dedupe_key, the restatement that was current at as_of.

        The online frame is sorted by event time, not by knowledge time, so we
        cannot simply take the row with the largest dedupe_order: an older row
        may carry a later knowledge time and must still win when both are
        known.  Group by key and pick max(order) among rows known at as_of.
        """
        if fr.dedupe_key is None or fr.dedupe_order_ns is None or len(idx) == 0:
            return np.ones(len(idx), dtype=bool)
        keys = fr.dedupe_key[idx]
        orders = fr.dedupe_order_ns[idx]
        known = fr.knowledge_ns[idx] <= as_of_ns
        best: dict[Any, int] = {}
        for k, o, kn in zip(keys, orders, known):
            if not kn:
                continue
            cur = best.get(k)
            if cur is None or o > cur:
                best[k] = int(o)
        return np.array([best.get(k) == int(o) for k, o in zip(keys, orders)], dtype=bool)

    def _window(self, f: spec.FeatureSpec, entity_id: str, as_of_ns: int, ctx: dict) -> Any:
        src = self.reg.sources[f.source]  # type: ignore[index]
        w = f.window_obj
        lo = as_of_ns - w.seconds * 1_000_000_000
        fr, idx = self._rows(f.source, entity_id, as_of_ns, lo)  # type: ignore[arg-type]
        if len(idx) == 0:
            return None
        mask = self._visible(src, fr, idx, as_of_ns)
        if len(mask) and f.where is not None:
            where = self._eval_where(f, fr, idx, as_of_ns, ctx)
            mask &= where
        mask &= self._dedupe_mask(fr, idx, as_of_ns)
        sel = idx[mask]
        if len(sel) == 0:
            return None
        agg_value = self._aggregate(f, fr, sel, as_of_ns)


        if f.transform == "days_since" and agg_value is not None:
            return _calendar_days(int(pd.Timestamp(agg_value).value), as_of_ns)
        return agg_value

    def _eval_where(self, f: spec.FeatureSpec, fr: EntityFrame, idx: np.ndarray,
                    as_of_ns: int, ctx: dict) -> np.ndarray:
        if f.where_node is None:
            return np.ones(len(idx), dtype=bool)
        columns = {name: np.asarray(arr, dtype=object)[idx].copy() for name, arr in fr.columns.items()}
        # hide deferred values that were not yet known at as_of
        for col, known in (fr.deferred_known_ns or {}).items():
            if col not in columns:
                continue
            not_known = known[idx] > as_of_ns
            if col in (fr.deferred_cleared_ns or {}):
                cleared = fr.deferred_cleared_ns[col][idx]
                not_known |= (cleared != _NULL_NS) & (cleared <= as_of_ns)
            if not_known.any():
                columns[col] = columns[col].astype(object)
                columns[col][not_known] = None
        out = expr.evaluate(f.where_node, columns, len(idx), ctx)  # type: ignore[arg-type]
        return np.array([bool(x) for x in out], dtype=bool)

    def _masked_values(self, fr: EntityFrame, sel: np.ndarray, as_of_ns: int) -> dict[str, np.ndarray]:
        """Column values with deferred (not-yet-known / revoked) entries NULLed."""
        out = {name: np.asarray(fr.columns[name][sel], dtype=object) for name in fr.columns}
        for col, known in (fr.deferred_known_ns or {}).items():
            if col not in out:
                continue
            vals = out[col].copy()
            hide = known[sel] > as_of_ns
            if col in (fr.deferred_cleared_ns or {}):
                cleared = fr.deferred_cleared_ns[col][sel]
                hide = hide | ((cleared != _NULL_NS) & (cleared <= as_of_ns))
            if hide.any():
                vals[hide] = None
            out[col] = vals
        return out

    def _aggregate(self, f: spec.FeatureSpec, fr: EntityFrame, sel: np.ndarray, as_of_ns: int) -> Any:
        agg = f.agg
        masked = self._masked_values(fr, sel, as_of_ns)
        if agg == "count":
            return int(len(sel))
        values = masked[f.column]  # type: ignore[index]
        if agg == "count_distinct":
            return int(len({v for v in values if v is not None and not _is_null_scalar(v)}))
        clean = [v for v in values if not (v is None or _is_null_scalar(v))]
        if not clean:
            return None
        if clean and isinstance(clean[0], (pd.Timestamp, np.datetime64, datetime)):
            # timestamp-valued aggregates (min/max of an event time)
            ts = pd.to_datetime(pd.Series(clean))
            if agg == "min":
                return ts.min().to_pydatetime()
            if agg == "max":
                return ts.max().to_pydatetime()
            raise ValueError(f"{f.name}: cannot {agg} a timestamp column")
        if agg in ("last", "first"):
            # ordered by event time, not by value
            ev = fr.event_ns[sel]
            raw = masked[f.column]  # type: ignore[index]
            keep = np.array([not (v is None or _is_null_scalar(v)) for v in raw])
            if not keep.any():
                return None
            ev = ev[keep]
            order = np.argsort(ev, kind="stable")
            pick = order[-1] if agg == "last" else order[0]
            return raw[keep][pick]
        if agg in ("sum", "avg"):
            try:
                numeric = [float(v) for v in clean]
            except (TypeError, ValueError):
                raise ValueError(f"{f.name}: {agg} requires a numeric column, got {clean[0]!r}")
        else:  # min / max on an ordered type
            numeric = [float(v) for v in clean] if isinstance(clean[0], (int, float)) else None
        if numeric is None:
            vals = sorted(clean)
            return vals[0] if agg == "min" else vals[-1]
        if agg == "sum":
            return float(sum(numeric))
        if agg == "avg":
            return float(sum(numeric) / len(numeric))
        if agg == "min":
            return float(min(numeric))
        if agg == "max":
            return float(max(numeric))
        raise ValueError(f"unsupported agg {agg!r}")

    def _asof_attr(self, f: spec.FeatureSpec, entity_id: str, as_of_ns: int, ctx: dict) -> Any:
        src = self.reg.sources[f.source]  # type: ignore[index]
        fr, idx = self._rows(f.source, entity_id, as_of_ns)  # type: ignore[arg-type]
        if len(idx) == 0:
            return None
        mask = self._visible(src, fr, idx, as_of_ns)
        sel = idx[mask]
        if len(sel) == 0:
            return None
        ev = fr.event_ns[sel]
        pick = sel[int(np.argmax(ev))]          # latest snapshot before as_of
        dspec = src.deferred.get(f.attr) if src.deferred else None
        if dspec and fr.deferred_known_ns and f.attr in fr.deferred_known_ns:
            if fr.deferred_known_ns[f.attr][pick] > as_of_ns:
                return None
            if dspec.cleared_at and fr.deferred_cleared_ns and f.attr in fr.deferred_cleared_ns:
                cleared = fr.deferred_cleared_ns[f.attr][pick]
                if cleared != _NULL_NS and cleared <= as_of_ns:
                    return None
        v = fr.columns[f.attr][pick]  # type: ignore[index]
        return None if v is None or _is_null_scalar(v) else v

    def _static_attr(self, f: spec.FeatureSpec, entity_id: str, as_of_ns: int, ctx: dict) -> Any:
        src = self.reg.sources[f.source]  # type: ignore[index]
        fr, idx = self._rows(f.source, entity_id, as_of_ns)  # type: ignore[arg-type]
        if len(idx) == 0:
            return None
        mask = self._visible(src, fr, idx, as_of_ns)
        sel = idx[mask]
        if len(sel) == 0:
            return None
        pick = sel[0]
        v = fr.columns[f.attr][pick]  # type: ignore[index]
        if v is None or _is_null_scalar(v):
            return None
        if f.transform == "days_since":
            return _calendar_days(fr.event_ns[pick], as_of_ns)
        return v

    def _derived(self, f: spec.FeatureSpec, raw: dict[str, Any], ctx: dict) -> Any:
        columns = {}
        for name in expr.referenced_columns(f.expr_node):
            if name in raw:
                columns[name] = raw[name]
            else:
                raise ValueError(f"derived feature {f.name}: unknown reference {name!r}")
        out = expr.evaluate(f.expr_node, columns, 1, ctx)
        v = out[0]
        return None if v is None or _is_null_scalar(v) else v


def _calendar_days(from_ns: int, to_ns: int) -> int:
    """Whole calendar days between two instants, matching DuckDB date_diff('day', DATE, DATE)."""
    d0 = pd.Timestamp(int(from_ns)).normalize()
    d1 = pd.Timestamp(int(to_ns)).normalize()
    return int((d1 - d0) / pd.Timedelta(days=1))


def _cast_scalar(value: Any, ftype: str) -> Any:
    """Mirror DuckDB CAST semantics for the scalar types used by the spec."""
    if value is None:
        return None
    if ftype == "int":
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (pd.Timestamp, datetime)):
            raise TypeError("cannot cast timestamp to int")
        return int(float(value))          # DuckDB truncates toward zero
    if ftype == "float":
        if isinstance(value, (pd.Timestamp, datetime)):
            return float(pd.Timestamp(value).value)   # timestamp ns as DOUBLE
        return float(value)
    if ftype == "bool":
        return bool(value)
    return str(value)


def _is_null_scalar(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False
