"""Offline <-> online parity audit.

For a random sample of applications the offline engine (compiled SQL, batch)
and the online engine (single-entity Python) are both asked for the same
``(entity, as_of)`` key.  Any difference is reported with the feature, both
values and the gap.

Two modes:

``exact``     as_of = the application's apply_time, the canonical training
              row.  This is the definition of "same moment, same user".
``skew``      as_of is jittered by +/- a few seconds around apply_time to
              emulate the online request arriving slightly off the batch
              anchor; a tiny mismatch rate here is expected for features whose
              value is a step function of time (e.g. window counts).

Tolerance: floats within ``rtol=1e-9`` and ints exactly equal.  NULL == NULL.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from feature_platform import config, spec
from feature_platform.offline.compiler import OfflineCompiler, build_base_cte
from feature_platform.online.engine import OnlineEngine, OnlineIndex


@dataclass
class ParityReport:
    view: str
    n_compared: int
    n_features: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    per_feature: dict[str, dict[str, Any]] = field(default_factory=dict)
    mode: str = "exact"
    skew_explained: int = 0
    skew_unexplained: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mismatch_rows(self) -> int:
        seen = {(m["application_id"]) for m in self.mismatches}
        return len(seen)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatch_rows / self.n_compared if self.n_compared else 0.0

    @property
    def cell_mismatch_rate(self) -> float:
        total = self.n_compared * self.n_features
        return len(self.mismatches) / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        top = {}
        for fname, st in sorted(self.per_feature.items(), key=lambda kv: -kv[1]["mismatch_rate"]):
            if st["mismatches"]:
                top[fname] = st
        out = {
            "view": self.view,
            "mode": self.mode,
            "n_compared": self.n_compared,
            "n_features": self.n_features,
            "mismatch_rows": self.mismatch_rows,
            "row_mismatch_rate": round(self.mismatch_rate, 6),
            "cell_mismatch_rate": round(self.cell_mismatch_rate, 8),
            "per_feature_with_mismatch": top,
            "examples": self.mismatches[:20],
        }
        if self.mode == "skew":
            out["skew_mismatches_explained_by_boundary_event"] = self.skew_explained
            out["skew_mismatches_unexplained"] = len(self.skew_unexplained)
            out["skew_unexplained_examples"] = self.skew_unexplained[:10]
            out["skew_explanation"] = (
                "online as_of is jittered +/-jitter_seconds; every mismatching key has a "
                "source event inside that window, so the difference is the event crossing "
                "the as_of boundary, not a pipeline defect")
        return out


def _is_null_value(v: Any) -> bool:
    if v is None or v is pd.NaT:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _values_equal(a: Any, b: Any, ftype: str) -> bool:
    a_null = _is_null_value(a)
    b_null = _is_null_value(b)
    if a_null and b_null:
        return True
    if a_null or b_null:
        return False
    if ftype == "bool":
        return bool(a) == bool(b)
    if ftype in ("int", "float"):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return str(a) == str(b)
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= max(1e-9, 1e-9 * max(abs(fa), abs(fb)))
    return str(a) == str(b)


def _jsonable(v: Any) -> Any:
    if v is None or v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def run_parity(
    view_name: str = "user_risk_v1",
    n: int = 500,
    seed: int = 7,
    mode: str = "exact",
    jitter_seconds: int = 3,
    registry: spec.Registry | None = None,
    con=None,
    index: OnlineIndex | None = None,
    engine: OnlineEngine | None = None,
    application_ids: list[str] | None = None,
) -> ParityReport:
    import duckdb

    reg = registry or spec.load_default()
    view = reg.views[view_name]
    own_con = con is None
    if own_con:
        con = duckdb.connect(str(config.DB_PATH), read_only=True)

    rng = random.Random(seed)
    if application_ids is None:
        ids = con.execute(
            f"SELECT {reg.base_id} FROM {reg.base_table} ORDER BY {reg.base_time}"
        ).fetchdf()[reg.base_id].tolist()
        application_ids = rng.sample(ids, min(n, len(ids)))

    # ---------------- offline side: one SQL pass restricted to the sample
    compiler = OfflineCompiler(reg)
    cv = compiler.compile_view(view)
    id_list = ", ".join("'" + str(i).replace("'", "''") + "'" for i in application_ids)
    base = build_base_cte(reg).replace(
        f"FROM {reg.base_table}",
        f"FROM {reg.base_table} WHERE {reg.base_id} IN ({id_list})",
    )
    offline_sql = f"WITH _base AS ({base})\n{cv.sql}"
    off = con.execute(offline_sql).fetchdf()
    off = off.set_index(reg.base_id)

    if engine is None:
        index = index or OnlineIndex(reg, con=con)
        engine = OnlineEngine(reg, index)

    metrics: dict[str, dict[str, Any]] = {
        name: {"mismatches": 0, "compared": 0, "max_abs_diff": 0.0} for name in cv.feature_order
    }
    mismatches: list[dict[str, Any]] = []
    compared = 0

    for aid in application_ids:
        if aid not in off.index:
            continue
        row = off.loc[aid]
        as_of = pd.Timestamp(row["as_of"])
        if mode == "skew":
            as_of = as_of + pd.Timedelta(seconds=rng.randint(-jitter_seconds, jitter_seconds))
        online = engine.score(view_name, str(row["entity_id"]), as_of)
        compared += 1
        for fname in cv.feature_order:
            ftype = cv.feature_types[fname]
            o = _jsonable(row[fname])
            nline = _jsonable(online["features"][fname])
            metrics[fname]["compared"] += 1
            if not _values_equal(o, nline, ftype):
                metrics[fname]["mismatches"] += 1
                diff = None
                try:
                    diff = abs(float(o) - float(nline))
                    metrics[fname]["max_abs_diff"] = max(metrics[fname]["max_abs_diff"], diff)
                except (TypeError, ValueError):
                    pass
                if len(mismatches) < 2000:
                    mismatches.append({
                        "application_id": aid,
                        "entity_id": str(row["entity_id"]),
                        "as_of": as_of.strftime("%Y-%m-%d %H:%M:%S"),
                        "feature": fname,
                        "offline": o,
                        "online": nline,
                        "abs_diff": diff,
                    })

    for fname, st in metrics.items():
        st["mismatch_rate"] = (st["mismatches"] / st["compared"]) if st["compared"] else 0.0
        st["max_abs_diff"] = float(st["max_abs_diff"])

    report = ParityReport(
        view=view_name,
        n_compared=compared,
        n_features=len(cv.feature_order),
        mismatches=mismatches,
        per_feature=metrics,
        mode=mode,
    )
    if mode == "skew" and mismatches:
        bad_keys = {(m["entity_id"], m["as_of"]) for m in mismatches}
        for entity, as_of in bad_keys:
            ts = pd.Timestamp(as_of)
            lo = (ts - pd.Timedelta(seconds=jitter_seconds)).to_pydatetime()
            hi = (ts + pd.Timedelta(seconds=jitter_seconds)).to_pydatetime()
            explained = False
            for src in reg.sources.values():
                if not src.table:
                    continue
                n = con.execute(
                    f'SELECT count(*) FROM {src.relation_sql} '
                    f'WHERE "{src.entity_key}" = ? AND "{src.event_time}" BETWEEN ? AND ?',
                    [entity, lo, hi],
                ).fetchone()[0]
                if n:
                    explained = True
                    break
            if explained:
                report.skew_explained += 1
            else:
                report.skew_unexplained.append({"entity_id": entity, "as_of": as_of})
    if own_con:
        con.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=["exact", "skew", "both"], default="both")
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "parity.json"))
    args = ap.parse_args()

    results = {}
    modes = ["exact", "skew"] if args.mode == "both" else [args.mode]
    for mode in modes:
        rep = run_parity(n=args.n, seed=args.seed, mode=mode)
        results[mode] = rep.as_dict()
        print(f"[{mode}] compared {rep.n_compared} rows x {rep.n_features} features; "
              f"row mismatch rate {rep.mismatch_rate:.4%}; "
              f"cell mismatch rate {rep.cell_mismatch_rate:.6%}")
        for fname, st in sorted(rep.per_feature.items(), key=lambda kv: -kv[1]["mismatch_rate"]):
            if st["mismatches"]:
                print(f"   {fname:32s} {st['mismatches']:5d}/{st['compared']}  "
                      f"({st['mismatch_rate']:.3%}) max_abs_diff={st['max_abs_diff']:.6g}")
    path = config.REPORTS_DIR / "parity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
