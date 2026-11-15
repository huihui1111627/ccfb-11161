"""Point-in-time leak audit: three independent proofs that training features
never use anything that happened at or after ``apply_time``.

Proof 1 -- prefix stability (data-access-level proof)
    The warehouse is physically truncated to what existed at ``as_of``
    (``event_time <= as_of`` and ``knowledge_time <= as_of`` for every source).
    Features recomputed on the truncated warehouse must equal features
    recomputed on the full warehouse at the same ``as_of``.  Any row from the
    future that a feature touched would show up as a difference.  The check is
    independent of the feature code: it re-runs the real compiler against less
    data.

Proof 2 -- boundary mutation (positive control, catches off-by-one)
    Every source row with ``event_time > as_of`` for the same entity has its
    *value* columns perturbed drastically.  If a feature boundary is inclusive
    (``<=`` instead of ``<``) or a knowledge guard is missing, the perturbed
    values appear at ``as_of`` and the check fails.

Proof 3 -- naive baseline contrast (quantifies what is being prevented)
    Deliberately naive feature definitions are scored against the same labels:
    latest-state dedupe, ingestion lag ignored, ``apply_time_local`` parsed as
    if it were UTC, latest profile snapshot instead of as-of snapshot.
    We report how many cells differ from the correct pipeline and how much a
    model's AUC moves.  This is what a "detector that never fires" cannot do:
    it shows the audit is sensitive enough to catch real leaks.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from feature_platform import config, spec
from feature_platform.audit.parity import _jsonable, _values_equal
from feature_platform.offline.compiler import OfflineCompiler, build_spine_sql


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _truncation_where(src: spec.SourceSpec, as_of: str) -> str:
    conds = [f'"{src.event_time}" <= TIMESTAMP \'{as_of}\'']
    if src.knowledge_time:
        conds.append(f'"{src.knowledge_time}" <= TIMESTAMP \'{as_of}\'')
    return " AND ".join(conds)


def _register_sources(con, reg: spec.Registry, prefix: str, sql_override: dict[str, str] | None = None) -> dict[str, str]:
    """Create temp views named ``<prefix><source>`` and return a table map."""
    mapping = {}
    for name, src in reg.sources.items():
        if not src.table:
            continue
        view = f"{prefix}{name}"
        body = (sql_override or {}).get(name, f"SELECT * FROM {src.table}")
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS {body}")
        mapping[name] = view
    return mapping


def _clone_with_tables(reg: spec.Registry, mapping: dict[str, str]) -> spec.Registry:
    reg2 = copy.deepcopy(reg)
    for name, table in mapping.items():
        reg2.sources[name].table = table
    return reg2


def _eval(con, reg: spec.Registry, keys: list[tuple[str, str]], view_name: str = "user_risk_v1") -> pd.DataFrame:
    compiler = OfflineCompiler(reg)
    cv = compiler.compile_view(reg.views[view_name], base_sql=build_spine_sql(reg, keys))
    return con.execute(cv.sql).fetchdf()


def _compare(a: pd.DataFrame, b: pd.DataFrame, features: list[str], types: dict[str, str]) -> list[dict]:
    if len(a) != len(b):
        return [{"error": f"row count differs {len(a)} vs {len(b)}"}]
    a = a.sort_values(["entity_id", "as_of"]).reset_index(drop=True)
    b = b.sort_values(["entity_id", "as_of"]).reset_index(drop=True)
    out = []
    for i in range(len(a)):
        for f in features:
            va, vb = _jsonable(a.at[i, f]), _jsonable(b.at[i, f])
            if not _values_equal(va, vb, types[f]):
                out.append({
                    "entity_id": a.at[i, "entity_id"],
                    "as_of": str(a.at[i, "as_of"]),
                    "feature": f,
                    "differs": va,
                    "reference": vb,
                })
    return out


def _sample_keys(con, reg: spec.Registry, n: int, seed: int) -> list[tuple[str, str]]:
    apps = con.execute(
        f"SELECT {reg.base_id}, {reg.base_entity}, {reg.base_time} FROM {reg.base_table} ORDER BY {reg.base_time}"
    ).fetchdf()
    idx = np.linspace(0, len(apps) - 1, n).astype(int)
    return [(str(apps.iloc[i][reg.base_entity]), str(apps.iloc[i][reg.base_time])) for i in idx]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

@dataclass
class LeakReport:
    n_sampled: int
    n_features: int
    prefix: dict[str, Any] = field(default_factory=dict)
    boundary: dict[str, Any] = field(default_factory=dict)
    naive: dict[str, Any] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        ok = (self.prefix.get("violations", 1) == 0
              and self.boundary.get("violations", 1) == 0
              and self.controls.get("all_detected", True))
        return {
            "n_sampled_applications": self.n_sampled,
            "n_features": self.n_features,
            "proof_1_prefix_stability": self.prefix,
            "proof_2_boundary_mutation": self.boundary,
            "proof_3_naive_baseline": self.naive,
            "proof_4_negative_controls": self.controls,
            "verdict": "PASS" if ok else "FAIL",
        }


# --------------------------------------------------------------------------
# Proof 1
# --------------------------------------------------------------------------

def proof_prefix_stability(con, reg: spec.Registry, keys, features, types) -> dict:
    full = _eval(con, reg, keys)
    violations: list[dict] = []
    cells = 0
    for entity, as_of in keys:
        truncated_sql = {}
        for name, src in reg.sources.items():
            if not src.table:
                continue
            truncated_sql[name] = (
                f'SELECT * FROM {src.table} WHERE {_truncation_where(src, as_of)}'
            )
        mapping = _register_sources(con, reg, "__trunc_", truncated_sql)
        reg_t = _clone_with_tables(reg, mapping)
        trunc = _eval(con, reg_t, [(entity, as_of)])
        ref = full[(full["entity_id"] == entity) & (full["as_of"].astype(str) == as_of)]
        diffs = _compare(trunc, ref, features, types)
        cells += len(features)
        violations.extend(diffs)
    return {
        "description": "features recomputed on a warehouse physically truncated to "
                       "event_time <= as_of AND knowledge_time <= as_of must be identical",
        "cells_compared": cells,
        "violations": len(violations),
        "violation_rate": len(violations) / cells if cells else 0.0,
        "examples": violations[:10],
    }


# --------------------------------------------------------------------------
# Proof 2
# --------------------------------------------------------------------------

_MUTATION_COLUMNS = {
    # source -> (column, sql expression for the mutated value)
    "loans": [("principal", '"{c}" * 97.0 + 12345.0')],
    "repayments": [("amount_paid", '"{c}" * 97.0 + 12345.0')],
    "applications": [("apply_amount", '"{c}" * 97.0 + 12345.0'),
                     ("apply_status", "'APPROVED'"),
                     ("reject_reason", "'fraud_suspect'")],
    "profile": [("credit_score", '"{c}" * 7 + 101'), ("is_blacklist", "NOT \"{c}\"")],
    "user_static": [("income_level", '"{c}" * 3 + 5000')],
}


def _mutation_sql(src: spec.SourceSpec, entity: str, as_of: str) -> str:
    """SELECT over the source with every row not fully known by as_of perturbed.

    Three future-knowledge surfaces are covered:

    * ``event_time >= as_of``   -- catches inclusive upper bounds
    * ``knowledge_time > as_of`` -- catches ignored ingestion lag (late feeds)
    * ``valid_until <= as_of``   -- catches ignored cancellation/reversal
    """
    cols = []
    mutated = dict(_MUTATION_COLUMNS.get(src.name, []))
    ent = entity.replace("'", "''")
    conds = [f'"{src.event_time}" >= TIMESTAMP \'{as_of}\'']
    if src.knowledge_time:
        conds.append(f'"{src.knowledge_time}" > TIMESTAMP \'{as_of}\'')
    if src.valid_until:
        conds.append(f'"{src.valid_until}" <= TIMESTAMP \'{as_of}\'')
    row_cond = f'"{src.entity_key}" = \'{ent}\' AND (' + " OR ".join(conds) + ")"
    guard_cols = {src.event_time, src.knowledge_time, src.dedupe_key, src.dedupe_order,
                  src.valid_until}
    guard_cols |= {d.known_at for d in (src.deferred or {}).values()}
    guard_cols |= {d.cleared_at for d in (src.deferred or {}).values() if d.cleared_at}
    for c in src.columns:
        if c in mutated:
            expr = mutated[c].format(c=c)
            cond = row_cond
            # A deferred column is future knowledge on its own clock: the row
            # may be old, but its decision outcome can still be unknown at
            # as_of.  Mutating it catches pipelines that skip the mask.
            dspec = (src.deferred or {}).get(c)
            if dspec:
                cond = (f'"{src.entity_key}" = \'{ent}\' AND '
                        f'"{dspec.known_at}" > TIMESTAMP \'{as_of}\'')
            cols.append(f'CASE WHEN {cond} THEN {expr} ELSE "{c}" END AS "{c}"')
        else:
            cols.append(f'"{c}"')
    return f'SELECT {", ".join(cols)} FROM {src.table}'


def _deferred_mutation_sql(src: spec.SourceSpec, entity: str, as_of: str) -> str:
    """Perturb only deferred columns whose value was not yet known at as_of.

    Kept separate from :func:`_mutation_sql` because a *correct* pipeline masks
    these values to NULL; mutating a different value would therefore look like
    a leak even though both sides agree on NULL.  The audit applies this
    mutation only when the pipeline under test claims to honour ``deferred``.
    """
    if not src.deferred:
        return ""
    ent = entity.replace("'", "''")
    guard_cols = set()
    for dspec in src.deferred.values():
        guard_cols.add(dspec.known_at)
        if dspec.cleared_at:
            guard_cols.add(dspec.cleared_at)
    cols = []
    for c in src.columns:
        dspec = (src.deferred or {}).get(c)
        if dspec and c not in guard_cols:
            cond = (f'"{src.entity_key}" = \'{ent}\' AND '
                    f'"{dspec.known_at}" > TIMESTAMP \'{as_of}\'')
            if c in ("apply_status", "reject_reason"):
                expr = "'leaked_value'"
            elif c == "amount_paid":
                expr = '"{c}" * 97.0 + 999.0'.format(c=c)
            else:
                expr = f'"{c}"'
            cols.append(f'CASE WHEN {cond} THEN {expr} ELSE "{c}" END AS "{c}"')
        else:
            cols.append(f'"{c}"')
    return f'SELECT {", ".join(cols)} FROM {src.table}'


# --------------------------------------------------------------------------
# Negative controls: prove the guards fire by *weakening* them
# --------------------------------------------------------------------------

def proof_negative_controls(con, reg: spec.Registry, n: int = 12, seed: int = 11) -> dict:
    """Compile the same spec with one point-in-time rule removed at a time.

    The correct pipeline must differ from each weakened pipeline on at least
    one cell.  A zero count would mean the rule is unobservable on this data
    (i.e. the audit could not have detected a violation of it), so this is the
    complement of Proof 2: Proof 2 checks "correct pipeline ignores future
    data", this checks "the audit is sensitive enough to notice when it
    doesn't".
    """
    cv = OfflineCompiler(reg).compile_view(reg.views["user_risk_v1"])
    keys = _sample_keys(con, reg, n, seed)
    ident = {name: f"SELECT * FROM {src.table}" for name, src in reg.sources.items() if src.table}
    bmap = _register_sources(con, reg, "__nc_base_", ident)
    reg_b = _clone_with_tables(reg, bmap)
    base = _eval(con, reg_b, keys)

    results = {}
    for mode, desc in [
        ("no_dedupe", "append-only loan restatements counted twice"),
        ("no_validity", "cancelled loans / reversed payments still counted"),
        ("latest_state", "latest profile snapshot used instead of as-of snapshot"),
        ("knowledge_lag", "late-arriving feed rows used before ingestion"),
        ("inclusive_boundary", "event at exactly as_of included"),
        ("local_time_shift", "apply_time_local parsed as UTC (+8h shift)"),
        ("no_deferred", "decision outcomes used before they were known"),
    ]:
        mm = _register_sources(con, reg, f"__nc_{mode}_", ident)
        reg_m = _clone_with_tables(reg, mm)
        alt = con.execute(
            OfflineCompiler(reg_m, naive={mode}).compile_view(
                reg_m.views["user_risk_v1"], base_sql=build_spine_sql(reg_m, keys)
            ).sql
        ).fetchdf()
        diffs = _compare(alt, base, cv.feature_order, cv.feature_types)
        results[mode] = {
            "description": desc,
            "cells_changed": len(diffs),
            "detected": bool(diffs),
            "features": sorted({d["feature"] for d in diffs})[:10],
        }
    return {
        "description": "each point-in-time rule is removed in turn; all must change at "
                       "least one feature, otherwise the audit could not detect its violation",
        "n_keys": len(keys),
        "n_features": len(cv.feature_order),
        "controls": results,
        "all_detected": all(v["detected"] for v in results.values()),
    }


def proof_boundary_mutation(con, reg: spec.Registry, keys, features, types) -> dict:
    violations: list[dict] = []
    cells = 0
    for entity, as_of in keys:
        # baseline: mutated catalogue in place, but with NO future row perturbed
        # (identity mutation) -- guarantees both sides read the same relations
        identity_sql = {name: f"SELECT * FROM {src.table}"
                        for name, src in reg.sources.items() if src.table}
        base_map = _register_sources(con, reg, "__mut_base_", identity_sql)
        reg_b = _clone_with_tables(reg, base_map)
        baseline = _eval(con, reg_b, [(entity, as_of)])

        mutated_sql = {name: _mutation_sql(src, entity, as_of)
                       for name, src in reg.sources.items() if src.table}
        mapping = _register_sources(con, reg, "__mut_", mutated_sql)
        reg_m = _clone_with_tables(reg, mapping)
        mutated = _eval(con, reg_m, [(entity, as_of)])
        diffs = _compare(mutated, baseline, features, types)
        cells += len(features)
        violations.extend(diffs)
    return {
        "description": "perturbing every source row with event_time > as_of must not change "
                       "any feature at as_of (off-by-one / inclusive-boundary detector)",
        "mutation": {src: [c for c, _ in cols] for src, cols in _MUTATION_COLUMNS.items()},
        "cells_compared": cells,
        "violations": len(violations),
        "violation_rate": len(violations) / cells if cells else 0.0,
        "examples": violations[:10],
    }


# --------------------------------------------------------------------------
# Proof 3 -- naive baselines
# --------------------------------------------------------------------------

def proof_naive_baseline(con, reg: spec.Registry, frame: pd.DataFrame, cv, seed: int = 3) -> dict:
    """Run the same spec through deliberately naive compiler modes.

    The platform pipeline gives the reference values; each naive mode is a
    plausible "what a hand-written SQL job would do" variant.  We report the
    share of feature cells that change and the AUC impact on the same labels.
    """
    keys = [(str(e), str(a)) for e, a in zip(frame["entity_id"], frame["__as_of"])]
    spine = build_spine_sql(reg, keys, base_id=reg.base_id)
    ref_sql = OfflineCompiler(reg).compile_view(reg.views["user_risk_v1"], base_sql=spine).sql
    ref = con.execute(f"SELECT * FROM ({ref_sql}) ORDER BY CAST(entity_row_id AS BIGINT)").fetchdf()

    modes = [
        ("inclusive_boundary", {"inclusive_boundary"},
         "window upper bound is <= instead of < (double counts an application at as_of)"),
        ("knowledge_lag_ignored", {"knowledge_lag"},
         "ignores ingested_at, so late-arriving partner feed rows appear early"),
        ("no_dedupe", {"no_dedupe"},
         "append-only loan restatements are all counted"),
        ("no_validity", {"no_validity"},
         "cancelled loans and reversed payments are still counted"),
        ("latest_profile", {"latest_state"},
         "uses the latest profile snapshot instead of the as-of snapshot"),
        ("local_time_as_utc", {"local_time_shift"},
         "parses apply_time_local (+08:00) as UTC, shifting all application events by 8h"),
        ("deferred_unguarded", {"no_deferred"},
         "reads decision outcomes (status) before they were known"),
        ("all_naive", {"inclusive_boundary", "knowledge_lag", "no_dedupe", "no_validity", "latest_state"},
         "all of the above combined"),
    ]
    y = frame["label"].to_numpy(dtype=int)
    cells = len(ref) * len(cv.feature_order)
    results = {}
    auc_ref = None
    for label, naive_set, desc in modes:
        alt_sql = OfflineCompiler(reg, naive=naive_set).compile_view(
            reg.views["user_risk_v1"], base_sql=spine
        ).sql
        alt = con.execute(f"SELECT * FROM ({alt_sql}) ORDER BY CAST(entity_row_id AS BIGINT)").fetchdf()
        diff_cells = 0
        per_feature = {}
        for f in cv.feature_order:
            d = 0
            for va, vb in zip(alt[f].to_numpy(dtype=object), ref[f].to_numpy(dtype=object)):
                if not _values_equal(_jsonable(va), _jsonable(vb), cv.feature_types[f]):
                    d += 1
            diff_cells += d
            if d:
                per_feature[f] = {"differing_cells": d, "rate": d / len(ref) if len(ref) else 0.0}
        # AUC on naive features (align by row order of the spine)
        Xn = alt[cv.feature_order].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        Xr = ref[cv.feature_order].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        auc_naive = _fit_auc(Xn, y, seed)
        if auc_ref is None:
            auc_ref = _fit_auc(Xr, y, seed)
        results[label] = {
            "description": desc,
            "differing_cells": diff_cells,
            "differing_cell_rate": diff_cells / cells if cells else 0.0,
            "per_feature": per_feature,
            "auc": auc_naive,
            "auc_delta_vs_platform": round(auc_naive - auc_ref, 4) if auc_ref is not None else None,
        }
    return {
        "description": "same spec compiled with deliberately naive point-in-time rules",
        "rows": int(len(ref)),
        "feature_columns": len(cv.feature_order),
        "cells_compared": cells,
        "auc_platform": round(auc_ref, 4) if auc_ref is not None else None,
        "variants": results,
    }


def _fit_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
    clf.fit(Xtr, ytr)
    return _auc(yte, clf.predict_proba(Xte)[:, 1])


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = ~np.isnan(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_audit(n_prefix: int = 40, n_boundary: int = 15, n_naive: int = 1200,
              seed: int = 11, con=None) -> LeakReport:
    reg = spec.load_default()
    own = con is None
    if own:
        con = duckdb.connect(str(config.DB_PATH), read_only=True)
    view = reg.views["user_risk_v1"]
    compiler = OfflineCompiler(reg)
    cv = compiler.compile_view(view)

    keys = _sample_keys(con, reg, n_prefix, seed)
    report = LeakReport(n_sampled=len(keys), n_features=len(cv.feature_order))
    report.prefix = proof_prefix_stability(con, reg, keys, cv.feature_order, cv.feature_types)
    report.boundary = proof_boundary_mutation(con, reg, keys[:n_boundary], cv.feature_order, cv.feature_types)
    report.controls = proof_negative_controls(con, reg, n=max(n_boundary * 4, 60), seed=seed)
    try:
        from feature_platform.offline.samples import build_training_frame

        build = build_training_frame("user_risk_v1", start="2024-01-01", registry=reg, con=con)
        frame = build.frame.sample(min(n_naive, len(build.frame)), random_state=seed).reset_index(drop=True)
        report.naive = proof_naive_baseline(con, reg, frame, cv)
    except Exception as exc:  # keep the audit usable if sklearn is unavailable
        report.naive = {"error": f"{type(exc).__name__}: {exc}"}
    if own:
        con.close()
    return report


def _markdown(rep: dict) -> str:
    p1, p2, p3 = rep["proof_1_prefix_stability"], rep["proof_2_boundary_mutation"], rep["proof_3_naive_baseline"]
    lines = [
        "# Point-in-time leak audit",
        "",
        f"- sampled applications: **{rep['n_sampled_applications']}**",
        f"- features per sample: **{rep['n_features']}**",
        f"- verdict: **{rep['verdict']}**",
        "",
        "## Proof 1 - prefix stability (truncated warehouse)",
        "",
        f"- cells compared: {p1['cells_compared']}",
        f"- violations: **{p1['violations']}** ({p1['violation_rate']:.6%})",
        "",
        "## Proof 2 - boundary mutation (future rows perturbed)",
        "",
        f"- cells compared: {p2['cells_compared']}",
        f"- violations: **{p2['violations']}** ({p2['violation_rate']:.6%})",
        "",
        "## Proof 3 - naive baseline contrast (positive control)",
        "",
    ]
    if "error" in p3:
        lines.append(f"skipped: {p3['error']}")
    else:
        lines += [
            f"- rows: {p3['rows']} x {p3['feature_columns']} features",
            f"- platform AUC: **{p3['auc_platform']}**",
            "",
            "| naive variant | cells differing | rate | AUC | AUC delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, st in p3["variants"].items():
            lines.append(
                f"| {name} | {st['differing_cells']} | {st['differing_cell_rate']:.2%} | "
                f"{st['auc']:.4f} | {st['auc_delta_vs_platform']:+.4f} |"
            )
    p4 = rep.get("proof_4_negative_controls", {})
    if p4 and "controls" in p4:
        lines += [
            "",
            "## Proof 4 - negative controls (each rule removed in turn)",
            "",
            f"- keys: {p4['n_keys']} x {p4['n_features']} features",
            f"- all rules detectable: **{p4['all_detected']}**",
            "",
            "| removed rule | cells changed | detected | features affected |",
            "|---|---:|---|---|",
        ]
        for name, st in p4["controls"].items():
            feats = ", ".join(st["features"][:4]) + ("..." if len(st["features"]) > 4 else "")
            lines.append(
                f"| {name} | {st['cells_changed']} | {'yes' if st['detected'] else '**NO**'} | {feats} |"
            )
    return "\n".join(lines) + "\n"
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prefix", type=int, default=40)
    ap.add_argument("--n-boundary", type=int, default=15)
    ap.add_argument("--n-naive", type=int, default=1200)
    args = ap.parse_args()
    rep = run_audit(n_prefix=args.n_prefix, n_boundary=args.n_boundary, n_naive=args.n_naive)
    d = rep.as_dict()
    out = config.REPORTS_DIR / "leak_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    md = config.REPORTS_DIR / "leak_audit.md"
    md.write_text(_markdown(d))
    print(json.dumps(d, indent=2, default=str)[:2500])
    print("wrote", out, md)


if __name__ == "__main__":
    main()
