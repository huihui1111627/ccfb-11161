"""Online/offline consistency monitor.

Requirement 5: sample the same ``(user, instant)`` on both paths and compare.

The monitor is intentionally *not* the parity test.  Parity tests run at
application time over training rows.  This monitor asks the serving question:
"at arbitrary instants -- including instants with no application at all --
does the batch engine and the online engine return the same vector?"  It runs
on a schedule, writes a per-day roll-up to DuckDB, and exposes both the
current mismatch rate and a day-by-day history for the dashboard.

Comparison policy
-----------------
* exact mode   as_of = instant, both engines
* skew mode    online is asked at as_of + jitter (the realistic case where the
               request lands seconds after the batch snapshot)
* tolerance    abs/rel 1e-9 for floats, exact for ints/strings/bools, NULL==NULL

Everything is recorded with the request, both values and the spec fingerprint,
so a mismatch can be reproduced exactly from the dashboard.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd

from feature_platform import config, spec
from feature_platform.audit.parity import _jsonable, _values_equal
from feature_platform.offline.compiler import OfflineCompiler, build_spine_sql
from feature_platform.online.engine import OnlineEngine, OnlineIndex


@dataclass
class ConsistencyRun:
    view: str
    mode: str
    n_requests: int
    n_features: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    per_feature: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at: str = ""
    duration_ms: float = 0.0
    fingerprint: str = ""

    @property
    def cells(self) -> int:
        return self.n_requests * self.n_features

    @property
    def mismatch_rate(self) -> float:
        return len(self.mismatches) / self.cells if self.cells else 0.0

    @property
    def request_mismatch_rate(self) -> float:
        bad = {m["entity_id"] + "|" + m["as_of"] for m in self.mismatches}
        return len(bad) / self.n_requests if self.n_requests else 0.0

    def as_dict(self) -> dict:
        return {
            "view": self.view,
            "mode": self.mode,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "spec_fingerprint": self.fingerprint,
            "n_requests": self.n_requests,
            "n_features": self.n_features,
            "cells_compared": self.cells,
            "mismatches": len(self.mismatches),
            "cell_mismatch_rate": round(self.mismatch_rate, 8),
            "request_mismatch_rate": round(self.request_mismatch_rate, 6),
            "per_feature": {
                k: v for k, v in sorted(self.per_feature.items(), key=lambda kv: -kv[1]["mismatches"])
                if v["mismatches"]
            },
            "examples": self.mismatches[:25],
        }


def _pick_requests(con, reg: spec.Registry, n: int, seed: int, mode: str) -> list[tuple[str, str]]:
    """Sample (user, instant) pairs.

    Half at application instants (the highest-risk case: right on a boundary),
    half at random instants inside each user's active life (the serving case).
    """
    rng = random.Random(seed)
    users = con.execute(f"SELECT DISTINCT {reg.base_entity} FROM {reg.base_table}").fetchdf()
    all_users = users[reg.base_entity].tolist()
    rng.shuffle(all_users)
    apps = con.execute(
        f"SELECT {reg.base_entity} AS e, {reg.base_time} AS t FROM {reg.base_table}"
    ).fetchdf()
    by_entity = apps.groupby("e")["t"].apply(list).to_dict()

    out: list[tuple[str, str]] = []
    half = n // 2
    sample_apps = apps.sample(min(half, len(apps)), random_state=seed)
    for _, r in sample_apps.iterrows():
        out.append((str(r["e"]), str(pd.Timestamp(r["t"]))))
    for e in all_users[: n - len(out)]:
        times = by_entity.get(e)
        if not times:
            continue
        lo = min(times) - pd.Timedelta(days=365)
        hi = max(times) + pd.Timedelta(days=30)
        span = (hi - lo).total_seconds()
        if span <= 0:
            continue
        inst = lo + pd.Timedelta(seconds=rng.uniform(0, span))
        out.append((e, str(inst)))
    return out[:n]


def run_consistency(
    view_name: str = "user_risk_v1",
    n: int = 200,
    seed: int | None = None,
    mode: str = "exact",
    jitter_seconds: int = 2,
    registry: spec.Registry | None = None,
    con=None,
    engine: OnlineEngine | None = None,
) -> ConsistencyRun:
    reg = registry or spec.load_default()
    seed = seed if seed is not None else int(time.time())
    own = con is None
    if own:
        con = duckdb.connect(str(config.DB_PATH), read_only=True)
    view = reg.views[view_name]
    compiler = OfflineCompiler(reg)
    cv = compiler.compile_view(view)

    run = ConsistencyRun(
        view=view_name, mode=mode, n_requests=0, n_features=len(cv.feature_order),
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        fingerprint=reg.fingerprint(),
    )
    t0 = time.perf_counter()
    requests = _pick_requests(con, reg, n, seed, mode)
    rng = random.Random(seed + 1)

    # ---------------- offline: one batched SQL pass
    spine = build_spine_sql(reg, requests)
    off = con.execute(f"SELECT * FROM ({cv.sql.replace('FROM (SELECT * FROM _base) b', 'FROM (' + spine + ') b')}) ORDER BY CAST(entity_row_id AS BIGINT)").fetchdf()
    off = off.reset_index(drop=True)

    if engine is None:
        index = OnlineIndex(reg, con=con)
        engine = OnlineEngine(reg, index)

    per_feature = {f: {"mismatches": 0, "compared": 0} for f in cv.feature_order}
    for i, (entity, as_of) in enumerate(requests):
        row = off.iloc[i]
        as_of_ts = pd.Timestamp(as_of)
        online_as_of = as_of_ts
        if mode == "skew":
            online_as_of = as_of_ts + pd.Timedelta(seconds=rng.randint(-jitter_seconds, jitter_seconds))
        online = engine.score(view_name, entity, online_as_of)
        run.n_requests += 1
        for fname in cv.feature_order:
            a = _jsonable(row[fname])
            b = _jsonable(online["features"][fname])
            per_feature[fname]["compared"] += 1
            if not _values_equal(a, b, cv.feature_types[fname]):
                per_feature[fname]["mismatches"] += 1
                if len(run.mismatches) < 5000:
                    run.mismatches.append({
                        "entity_id": entity,
                        "as_of": as_of,
                        "online_as_of": str(online_as_of),
                        "feature": fname,
                        "offline": a,
                        "online": b,
                    })
    for fname, st in per_feature.items():
        st["mismatch_rate"] = st["mismatches"] / st["compared"] if st["compared"] else 0.0
    run.per_feature = per_feature
    run.duration_ms = (time.perf_counter() - t0) * 1000
    if own:
        con.close()
    return run


# --------------------------------------------------------------------------
# Persistence: daily roll-ups used by the dashboard
# --------------------------------------------------------------------------

def ensure_store(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS monitor_consistency (
            run_ts TIMESTAMP, view VARCHAR, mode VARCHAR, spec_fingerprint VARCHAR,
            n_requests INTEGER, n_features INTEGER, cells INTEGER,
            mismatches INTEGER, cell_mismatch_rate DOUBLE, request_mismatch_rate DOUBLE,
            duration_ms DOUBLE, details VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS monitor_drift (
            run_date DATE, view VARCHAR, feature VARCHAR, kind VARCHAR,
            metric VARCHAR, value DOUBLE, baseline DOUBLE, z_score DOUBLE,
            threshold DOUBLE, breached BOOLEAN, details VARCHAR
        )
    """)


def record_run(con, run: ConsistencyRun) -> None:
    ensure_store(con)
    con.execute(
        "INSERT INTO monitor_consistency VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [datetime.now(), run.view, run.mode, run.fingerprint, run.n_requests,
         run.n_features, run.cells, len(run.mismatches),
         run.mismatch_rate, run.request_mismatch_rate, run.duration_ms,
         json.dumps(run.mismatches[:50], default=str)],
    )


def history(con, days: int = 30) -> pd.DataFrame:
    return con.execute(f"""
        SELECT CAST(run_ts AS DATE) AS day, view, mode,
               count(*) AS runs, sum(cells) AS cells, sum(mismatches) AS mismatches,
               avg(cell_mismatch_rate) AS avg_cell_mismatch_rate,
               max(cell_mismatch_rate) AS max_cell_mismatch_rate,
               avg(duration_ms) AS avg_duration_ms
        FROM monitor_consistency
        WHERE run_ts >= current_date - INTERVAL {int(days)} DAY
        GROUP BY 1,2,3 ORDER BY 1 DESC, 2
    """).fetchdf()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", choices=["exact", "skew", "both"], default="both")
    args = ap.parse_args()
    con = duckdb.connect(str(config.DB_PATH), read_only=False)
    results = {}
    for mode in (["exact", "skew"] if args.mode == "both" else [args.mode]):
        run = run_consistency(n=args.n, mode=mode)
        record_run(con, run)
        results[mode] = run.as_dict()
        print(f"[{mode}] {run.n_requests} requests x {run.n_features} features "
              f"cells={run.cells} mismatches={len(run.mismatches)} "
              f"cell_rate={run.mismatch_rate:.6%} request_rate={run.request_mismatch_rate:.4%} "
              f"({run.duration_ms:.0f} ms)")
    out = config.REPORTS_DIR / "consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print("wrote", out)
    con.close()


if __name__ == "__main__":
    main()
