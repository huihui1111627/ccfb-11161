"""Model benchmark: does the point-in-time pipeline actually learn, and what
does leakage do to the estimate?

Trains the same gradient-boosted model on

* **clean**: the point-in-time feature matrix
* **leaky**: the same matrix with features augmented by outcome-derived
  columns (final loan status, whether the loan went 30+ DPD, the label itself
  shifted by user) -- the kind of thing that happens when someone joins the
  label table into the feature job by accident
* **naive**: the deliberately naive pipeline (inclusive bounds, no ingestion
  guard, latest snapshot)

and reports AUC on a chronological hold-out.  High leaky AUC is the failure
mode the platform is designed to make impossible; clean AUC is the honest
number.
"""
from __future__ import annotations

import json
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from feature_platform import config, spec
from feature_platform.offline.compiler import OfflineCompiler, build_spine_sql
from feature_platform.offline.samples import build_training_frame, time_split


def _fit_auc(df: pd.DataFrame, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> float:
    Xtr = train[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    Xte = test[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    ytr = train["label"].to_numpy(dtype=int)
    yte = test["label"].to_numpy(dtype=int)
    clf = HistGradientBoostingClassifier(max_iter=250, random_state=7, learning_rate=0.06)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def run_benchmark(view_name: str = "user_risk_v1") -> dict[str, Any]:
    reg = spec.load_default()
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        build = build_training_frame(view_name, start="2024-01-01", registry=reg, con=con)
        df = build.frame
        tr, va, te = time_split(df)
        features = build.feature_names

        train_auc = _fit_auc(df, features, tr, tr)
        valid_auc = _fit_auc(df, features, tr, va)
        test_auc = _fit_auc(df, features, tr, te)

        # ---- realistic future-window leak
        # The classic production bug: someone writes the label-window
        # aggregation and joins it back onto the training rows without an
        # as_of guard.  These columns aggregate outcomes in the 90 days AFTER
        # apply_time -- perfectly predictive, and unavailable at decision time.
        leaky_sql = """
        SELECT a.application_id,
               coalesce(f.max_dpd, 0)            AS fut_max_dpd,
               coalesce(f.loan_cnt, 0)           AS fut_loan_cnt,
               coalesce(f.cancelled_cnt, 0)      AS fut_cancelled_cnt
        FROM applications a
        LEFT JOIN LATERAL (
            SELECT max(l.final_dpd) AS max_dpd,
                   sum(CASE WHEN l.loan_status = 'CANCELLED' THEN 1 ELSE 0 END) AS cancelled_cnt,
                   count(*) AS loan_cnt
            FROM loans l
            WHERE l.user_id = a.user_id
              AND l.loan_time >= a.apply_time
              AND l.loan_time <  a.apply_time + INTERVAL 90 DAY
        ) f ON TRUE
        """
        leaky = df.merge(con.execute(leaky_sql).fetchdf(), on="application_id", how="left")
        repaid = con.execute("""
        SELECT a.application_id,
               coalesce(sum(r.amount_paid), 0) AS fut_repaid_amt,
               coalesce(sum(CASE WHEN r.dpd_days >= 30 THEN 1 ELSE 0 END), 0) AS fut_overdue_cnt
        FROM applications a
        LEFT JOIN repayments r
          ON r.user_id = a.user_id
         AND r.due_time >= a.apply_time
         AND r.due_time <  a.apply_time + INTERVAL 90 DAY
        GROUP BY 1
        """).fetchdf()
        leaky = leaky.merge(repaid, on="application_id", how="left")
        ltr, lva, lte = time_split(leaky)
        leaky_features = features + ["fut_max_dpd", "fut_overdue_cnt", "fut_repaid_amt",
                                     "fut_loan_cnt", "fut_cancelled_cnt"]
        leaky_test_auc = _fit_auc(leaky, leaky_features, ltr, lte)

        # ---- naive pipeline on the same keys
        keys = [(str(e), str(a)) for e, a in zip(df["entity_id"], df["__as_of"])]
        spine = build_spine_sql(reg, keys)
        naive_sql = OfflineCompiler(
            reg, naive={"inclusive_boundary", "knowledge_lag", "latest_state", "no_validity"}
        ).compile_view(reg.views[view_name], base_sql=spine).sql
        naive = con.execute(f"SELECT * FROM ({naive_sql}) ORDER BY CAST(entity_row_id AS BIGINT)").fetchdf()
        naive = naive.reset_index(drop=True)
        ndf = df.copy().reset_index(drop=True)
        for f in features:
            ndf[f] = naive[f].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        ntr, nva, nte = time_split(ndf)
        naive_test_auc = _fit_auc(ndf, features, ntr, nte)

        top = _feature_importance(tr, features, te)
        return {
            "view": view_name,
            "rows": {"train": len(tr), "valid": len(va), "test": len(te)},
            "features": features,
            "train_auc": round(train_auc, 4),
            "valid_auc": round(valid_auc, 4),
            "test_auc": round(test_auc, 4),
            "test_auc_with_leaky_features": round(leaky_test_auc, 4),
            "leaky_feature_names": ["fut_max_dpd", "fut_overdue_cnt", "fut_repaid_amt",
                                    "fut_loan_cnt", "fut_cancelled_cnt"],
            "test_auc_naive_pipeline": round(naive_test_auc, 4),
            "leak_uplift_auc": round(leaky_test_auc - test_auc, 4),
            "naive_uplift_auc": round(naive_test_auc - test_auc, 4),
            "top_features": top,
            "label_rate": {"train": float(tr["label"].mean()), "test": float(te["label"].mean())},
            "split": "chronological (train < 2024-11-01 <= valid < 2025-04-01 <= test)",
        }
    finally:
        con.close()


def _feature_importance(train: pd.DataFrame, features: list[str], test: pd.DataFrame) -> list[dict]:
    Xtr = train[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    ytr = train["label"].to_numpy(dtype=int)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=7)
    clf.fit(Xtr, ytr)
    # permutation importance on the hold-out is expensive; use split gain proxy
    from sklearn.inspection import permutation_importance

    Xte = test[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    yte = test["label"].to_numpy(dtype=int)
    r = permutation_importance(clf, Xte, yte, n_repeats=3, random_state=7, scoring="roc_auc")
    order = np.argsort(-r.importances_mean)
    return [
        {"feature": features[i], "importance": round(float(r.importances_mean[i]), 5)}
        for i in order[:12] if r.importances_mean[i] > 0
    ]


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
