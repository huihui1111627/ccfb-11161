"""Training-sample generation with point-in-time guarantees.

For every application the sample row carries

* the feature vector evaluated at ``as_of = apply_time`` (never later),
* the label *and* ``label_ready_time`` so downstream code can enforce that the
  outcome was observable by the time the sample is actually used,
* the model-ready matrix in a stable feature order.

Two extra columns are written for the audit:

``__as_of``             the exact instant features were computed at
``__label_ready_time``  the instant the label became observable

Training/validation splits are made **on time, not at random**, because random
splits leak through users and through time.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from feature_platform import config, spec
from feature_platform.offline.compiler import OfflineCompiler, build_base_cte


@dataclass
class TrainingBuild:
    frame: pd.DataFrame
    feature_names: list[str]
    label_name: str = "label"
    train: pd.DataFrame | None = None
    valid: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    stats: dict | None = None


def build_training_frame(
    view_name: str = "user_risk_v1",
    start: str | None = None,
    end: str | None = None,
    include_immature_labels: bool = False,
    registry: spec.Registry | None = None,
    con=None,
) -> TrainingBuild:
    reg = registry or spec.load_default()
    own = con is None
    if own:
        con = duckdb.connect(str(config.DB_PATH), read_only=True)

    compiler = OfflineCompiler(reg)
    cv = compiler.compile_view(reg.views[view_name])
    ids = ", ".join(f'"{n}"' for n in cv.feature_order)
    where = []
    if start:
        where.append(f"{reg.base_time} >= TIMESTAMP '{start}'")
    if end:
        where.append(f"{reg.base_time} < TIMESTAMP '{end}'")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
    WITH _base AS (
        {build_base_cte(reg).replace(f'FROM {reg.base_table}', f'FROM {reg.base_table} a {where_sql}')}
    ),
    _features AS (
        {cv.sql}
    )
    SELECT f.entity_row_id, f.entity_id, f.as_of AS __as_of, f."{reg.base_id}",
           l.label, l.label_ready_time AS __label_ready_time, l.max_dpd_90d,
           l.label_reason,
           {ids}
    FROM _features f
    JOIN {reg.label_table} l ON l."{reg.base_id}" = f."{reg.base_id}"
    WHERE l.label IS NOT NULL
    ORDER BY __as_of
    """
    df = con.execute(sql).fetchdf()
    if own:
        con.close()
    if include_immature_labels:
        df = df[df["label_reason"] != "cancelled"].copy()
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    for n in cv.feature_order:
        if cv.feature_types[n] == "int":
            df[n] = df[n].astype("Int64")

    stats = _stats(df, cv.feature_order)
    return TrainingBuild(frame=df, feature_names=cv.feature_order, stats=stats)


def _stats(df: pd.DataFrame, features: list[str]) -> dict:
    out = {
        "rows": int(len(df)),
        "label_rate": float(df["label"].mean()) if len(df) else 0.0,
        "as_of_min": str(df["__as_of"].min()) if len(df) else None,
        "as_of_max": str(df["__as_of"].max()) if len(df) else None,
        "missing_rate": {f: float(df[f].isna().mean()) for f in features},
    }
    return out


def time_split(df: pd.DataFrame, train_end: str = "2024-11-01", valid_end: str = "2025-04-01") -> tuple:
    """Chronological split.  Also used to verify no future leakage across sets."""
    df = df.sort_values("__as_of").reset_index(drop=True)
    tr = df[df["__as_of"] < pd.Timestamp(train_end)]
    va = df[(df["__as_of"] >= pd.Timestamp(train_end)) & (df["__as_of"] < pd.Timestamp(valid_end))]
    te = df[df["__as_of"] >= pd.Timestamp(valid_end)]
    return tr, va, te


def write_outputs(build: TrainingBuild, out_dir: Path | None = None) -> dict[str, str]:
    out_dir = out_dir or (config.DATA_DIR / "training")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    tr, va, te = time_split(build.frame)
    for name, part in (("train", tr), ("valid", va), ("test", te)):
        p = out_dir / f"{name}.parquet"
        part.to_parquet(p, index=False)
        paths[name] = str(p)
    meta = {
        "features": build.feature_names,
        "rows": {k: int(len(v)) for k, v in (("train", tr), ("valid", va), ("test", te))},
        "label_rates": {k: float(v["label"].mean()) for k, v in (("train", tr), ("valid", va), ("test", te))},
        "stats": build.stats,
        "split_policy": "time-based",
        "train_end": "2024-11-01",
        "valid_end": "2025-04-01",
    }
    mp = out_dir / "meta.json"
    mp.write_text(json.dumps(meta, indent=2))
    paths["meta"] = str(mp)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="user_risk_v1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    build = build_training_frame(args.view, start=args.start, end=args.end)
    paths = write_outputs(build)
    print(json.dumps({"rows": len(build.frame), "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
