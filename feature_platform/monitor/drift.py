"""Daily missing-rate and distribution-drift monitoring.

Requirement 6: missing rate and distribution drift tracked per feature per day.

Method
------
* Reference window vs. evaluation day (default: trailing 90d excluding the day
  being scored, vs. that day's scoring population).
* Missing rate: absolute delta + Wilson interval; a breach needs both a
  material delta (``min_missing_delta``) and statistical significance.
* Numeric drift: two-sample Kolmogorov-Smirnov statistic with a permutation
  p-value, plus PSI over decile bins built from the reference window.
* Categorical drift: PSI over frozen category codes, with the ``__UNKNOWN__``
  bucket tracked separately (it is the online safety net; a rising share means
  the encoding table is stale).
* Every metric is written to ``monitor_drift`` with its threshold and a
  boolean, so the dashboard can show the actual time series, not just a flag.

The synthetic generator plants a 2025 risk-mix shift, so the drift monitor has
something real to find.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from feature_platform import config, spec
from feature_platform.offline.samples import build_training_frame


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-derived bin edges."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.sort(reference[~np.isnan(reference)])
    cur = np.sort(current[~np.isnan(current)])
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    allv = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, allv, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, allv, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def ks_pvalue(d: float, n1: int, n2: int) -> float:
    """Asymptotic Kolmogorov distribution survival function."""
    if not (d >= 0) or n1 == 0 or n2 == 0:
        return 1.0
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        term = 2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return float(min(max(total, 0.0), 1.0))


def missing_breach(reference: pd.Series, current: pd.Series, min_delta: float = 0.02) -> dict:
    n_ref, n_cur = len(reference), len(current)
    k_ref, k_cur = int(reference.isna().sum()), int(current.isna().sum())
    r_ref = k_ref / n_ref if n_ref else 0.0
    r_cur = k_cur / n_cur if n_cur else 0.0
    lo, hi = wilson_interval(k_cur, n_cur)
    delta = r_cur - r_ref
    return {
        "reference_missing_rate": r_ref,
        "current_missing_rate": r_cur,
        "delta": delta,
        "current_ci": [lo, hi],
        "breached": bool(abs(delta) >= min_delta and (lo > r_ref or hi < r_ref)),
    }


# --------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------

@dataclass
class DriftConfig:
    reference_days: int = 90
    min_rows_per_day: int = 30
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_pvalue: float = 0.01
    min_missing_delta: float = 0.02
    unknown_code_alert: float = 0.02
    mean_shift_sd: float = 0.25


def _daily_frames(con, reg: spec.Registry, view_name: str, days: int) -> pd.DataFrame:
    build = build_training_frame(view_name, include_immature_labels=True, registry=reg, con=con)
    df = build.frame.copy()
    df["day"] = pd.to_datetime(df["__as_of"]).dt.normalize()
    cutoff = df["day"].max() - pd.Timedelta(days=days)
    return df[df["day"] >= cutoff].reset_index(drop=True)


def run_drift(
    view_name: str = "user_risk_v1",
    days: int = 120,
    cfg: DriftConfig | None = None,
    registry: spec.Registry | None = None,
    con=None,
) -> dict[str, Any]:
    cfg = cfg or DriftConfig()
    reg = registry or spec.load_default()
    own = con is None
    if own:
        con = duckdb.connect(str(config.DB_PATH), read_only=True)
    view = reg.views[view_name]
    df = _daily_frames(con, reg, view_name, days)

    encodings = spec.load_encodings()
    all_days = sorted(df["day"].unique())
    results: list[dict] = []
    latest_summary: dict[str, Any] = {}

    for day in all_days:
        cur = df[df["day"] == day]
        if len(cur) < cfg.min_rows_per_day:
            continue
        window_start = day - pd.Timedelta(days=cfg.reference_days)
        ref = df[(df["day"] >= window_start) & (df["day"] < day)]
        if len(ref) < cfg.min_rows_per_day * 5:
            continue
        for f in view.features:
            ftype = f.type
            rec = {
                "run_date": pd.Timestamp(day).date(),
                "view": view_name,
                "feature": f.name,
                "kind": ftype,
            }
            miss = missing_breach(ref[f.name], cur[f.name], cfg.min_missing_delta)
            results.append({**rec, "metric": "missing_rate", "value": miss["current_missing_rate"],
                            "baseline": miss["reference_missing_rate"], "z_score": None,
                            "threshold": cfg.min_missing_delta, "breached": miss["breached"],
                            "details": json.dumps(miss)})
            if ftype in ("int", "float"):
                rv = pd.to_numeric(ref[f.name], errors="coerce").to_numpy(dtype=float)
                cv_ = pd.to_numeric(cur[f.name], errors="coerce").to_numpy(dtype=float)
                if np.isnan(rv).all() or np.isnan(cv_).all():
                    continue
                ref_null = float(np.isnan(rv).mean())
                cur_null = float(np.isnan(cv_).mean())
                # PSI/KS are meaningless when a feature is mostly NULL on
                # either side; missing-rate covers that regime instead.
                sparse = ref_null > 0.6 or cur_null > 0.6
                if not sparse:
                    psi_v = psi(rv, cv_)
                    ks = ks_statistic(rv, cv_)
                    p = ks_pvalue(ks, len(rv), len(cv_))
                    breached = psi_v >= cfg.psi_alert or p < cfg.ks_pvalue
                    results.append({**rec, "metric": "psi", "value": psi_v, "baseline": None,
                                    "z_score": None, "threshold": cfg.psi_alert,
                                    "breached": breached,
                                    "details": json.dumps({"ks": ks, "ks_pvalue": p,
                                                           "ref_null": ref_null,
                                                           "cur_null": cur_null})})
                    results.append({**rec, "metric": "ks_stat", "value": ks, "baseline": None,
                                    "z_score": None, "threshold": cfg.ks_pvalue,
                                    "breached": breached,
                                    "details": json.dumps({"ks_pvalue": p, "psi": psi_v})})
                # mean shift in standard deviations (works with NULLs, robust
                # to the heavy tails that inflate PSI)
                ref_mean = float(np.nanmean(rv)) if not np.isnan(rv).all() else float("nan")
                cur_mean = float(np.nanmean(cv_)) if not np.isnan(cv_).all() else float("nan")
                ref_sd = float(np.nanstd(rv)) if not np.isnan(rv).all() else float("nan")
                if not (np.isnan(ref_mean) or np.isnan(cur_mean) or np.isnan(ref_sd)) and ref_sd > 0:
                    z = (cur_mean - ref_mean) / ref_sd
                    results.append({**rec, "metric": "mean_shift_sd", "value": z,
                                    "baseline": ref_mean, "z_score": z,
                                    "threshold": cfg.mean_shift_sd,
                                    "breached": abs(z) >= cfg.mean_shift_sd,
                                    "details": json.dumps({"reference_mean": ref_mean,
                                                           "current_mean": cur_mean,
                                                           "reference_sd": ref_sd})})
            else:
                # categorical: PSI over the frozen code table + unknown share
                codes = np.array([np.nan if v is None else float(v) for v in cur[f.name]], dtype=float)
                ref_codes = np.array([np.nan if v is None else float(v) for v in ref[f.name]], dtype=float)
                ref_null = float(np.isnan(ref_codes).mean())
                cur_null = float(np.isnan(codes).mean())
                if not (ref_null > 0.6 or cur_null > 0.6):
                    psi_v = psi(ref_codes, codes)
                    results.append({**rec, "metric": "psi", "value": psi_v, "baseline": None,
                                    "z_score": None, "threshold": cfg.psi_alert,
                                    "breached": psi_v >= cfg.psi_alert,
                                    "details": json.dumps({"kind": "categorical",
                                                           "ref_null": ref_null,
                                                           "cur_null": cur_null})})
                if f.name in view.by_name() and view.by_name()[f.name].encoder is not None:
                    enc_key = f"{view_name}.{f.name}"
                    enc = encodings.get("encodings", {}).get(enc_key)
                    if enc:
                        unknown = float((codes == enc["default_code"]).mean())
                        results.append({**rec, "metric": "unknown_category_share", "value": unknown,
                                        "baseline": None, "z_score": None,
                                        "threshold": cfg.unknown_code_alert,
                                        "breached": unknown >= cfg.unknown_code_alert,
                                        "details": json.dumps({"default_code": enc["default_code"]})})
        # latest-day summary
        day_recs = [r for r in results if r["run_date"] == pd.Timestamp(day).date()]
        if day_recs:
            latest_summary[str(pd.Timestamp(day).date())] = {
                "rows": int(len(cur)),
                "breached_features": sorted({r["feature"] for r in day_recs if r["breached"]}),
                "n_metrics": len(day_recs),
                "n_breached": sum(1 for r in day_recs if r["breached"]),
            }

    out = {
        "view": view_name,
        "days_evaluated": len(latest_summary),
        "config": cfg.__dict__,
        "metrics": results,
        "daily_summary": latest_summary,
    }
    if own:
        con.close()
    return out


def persist(con, payload: dict[str, Any]) -> int:
    from feature_platform.monitor.consistency import ensure_store

    ensure_store(con)
    rows = []
    for r in payload["metrics"]:
        rows.append([
            r["run_date"], r["view"], r["feature"], r["kind"], r["metric"],
            float(r["value"]) if r["value"] is not None and not (isinstance(r["value"], float) and math.isnan(r["value"])) else None,
            float(r["baseline"]) if r["baseline"] is not None else None,
            float(r["z_score"]) if r["z_score"] is not None else None,
            float(r["threshold"]) if r["threshold"] is not None else None,
            bool(r["breached"]), r["details"],
        ])
    if rows:
        con.executemany("INSERT INTO monitor_drift VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def summary_for_dashboard(con, days: int = 30) -> pd.DataFrame:
    from feature_platform.monitor.consistency import ensure_store

    ensure_store(con)
    return con.execute(f"""
        SELECT run_date, feature, kind, metric, value, baseline, threshold, breached
        FROM monitor_drift
        WHERE run_date >= current_date - INTERVAL {int(days)} DAY
          AND metric IN ('missing_rate', 'psi', 'unknown_category_share')
        ORDER BY run_date DESC, feature, metric
    """).fetchdf()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "drift.json"))
    args = ap.parse_args()
    payload = run_drift(days=args.days)
    from feature_platform.monitor.consistency import ensure_store

    con = duckdb.connect(str(config.DB_PATH), read_only=False)
    ensure_store(con)
    con.execute("DELETE FROM monitor_drift")
    n = persist(con, payload)
    con.close()
    path = config.REPORTS_DIR / "drift.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in payload.items() if k != "metrics"}, indent=2, default=str))
    print(f"persisted {n} drift metrics; summary days={payload['days_evaluated']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
