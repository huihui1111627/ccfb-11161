#!/usr/bin/env python3
"""One-shot build of every artifact the platform and dashboard need.

    python scripts/build_all.py            # full build (~2 min)
    python scripts/build_all.py --quick    # smaller audit samples

Steps
-----
1. synthetic dataset      (data/feature_platform.duckdb)
2. frozen encodings       (data/online_index/encodings.json)
3. training samples       (data/training/{train,valid,test}.parquet)
4. offline/online parity  (reports/parity.json)
5. point-in-time audit    (reports/leak_audit.{json,md})
6. consistency monitoring (reports/consistency.json + monitor tables)
7. drift monitoring       (reports/drift.json + monitor tables)
8. model benchmark        (reports/model_benchmark.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_platform import spec  # noqa: E402


def stage(name: str, fn) -> None:
    t0 = time.time()
    print(f"\n=== {name} " + "=" * max(0, 64 - len(name)))
    fn()
    print(f"--- {name} done in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()

    if not args.skip_data:
        def gen():
            from feature_platform.data.generate import generate, write_to_duckdb

            tables = generate()
            write_to_duckdb(tables)
        stage("1/8 synthetic dataset", gen)

    def enc():
        from feature_platform.online.encodings import freeze

        freeze()
    stage("2/8 frozen encodings", enc)

    def samples():
        from feature_platform.offline.samples import build_training_frame, write_outputs

        build = build_training_frame(start="2024-01-01")
        paths = write_outputs(build)
        print(json.dumps({"rows": len(build.frame), "paths": paths}, indent=1))
    stage("3/8 training samples", samples)

    def parity():
        from feature_platform.audit.parity import run_parity

        out = {}
        for mode in ("exact", "skew"):
            rep = run_parity(n=600 if args.quick else 2000, mode=mode)
            out[mode] = rep.as_dict()
            print(f"[{mode}] rows={rep.n_compared} features={rep.n_features} "
                  f"row_mismatch={rep.mismatch_rate:.4%} cell_mismatch={rep.cell_mismatch_rate:.6%} "
                  f"skew_explained={rep.skew_explained} skew_unexplained={len(rep.skew_unexplained)}")
        path = ROOT / "reports" / "parity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    stage("4/8 offline/online parity", parity)

    def leak():
        from feature_platform.audit.leak import _markdown, run_audit

        rep = run_audit(
            n_prefix=20 if args.quick else 60,
            n_boundary=10 if args.quick else 25,
            n_naive=500 if args.quick else 2000,
        )
        d = rep.as_dict()
        out = ROOT / "reports" / "leak_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))
        (ROOT / "reports" / "leak_audit.md").write_text(_markdown(d))
        print(f"verdict={d['verdict']} prefix_viol={d['proof_1_prefix_stability']['violations']} "
              f"boundary_viol={d['proof_2_boundary_mutation']['violations']} "
              f"controls_detected={d['proof_4_negative_controls']['all_detected']}")
    stage("5/8 point-in-time leak audit", leak)

    def consistency():
        import duckdb
        from feature_platform import config
        from feature_platform.monitor.consistency import record_run, run_consistency

        con = duckdb.connect(str(config.DB_PATH), read_only=False)
        out = {}
        try:
            from feature_platform.online.engine import OnlineEngine, OnlineIndex

            reg = spec.load_default()
            index = OnlineIndex(reg, con=con)
            engine = OnlineEngine(reg, index)
            for mode in ("exact", "skew"):
                run = run_consistency(n=200 if args.quick else 800, mode=mode,
                                      registry=reg, con=con, engine=engine)
                record_run(con, run)
                out[mode] = run.as_dict()
                print(f"[{mode}] requests={run.n_requests} cells={run.cells} "
                      f"mismatches={len(run.mismatches)} cell_rate={run.mismatch_rate:.6%}")
        finally:
            con.close()
        path = ROOT / "reports" / "consistency.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str))
    stage("6/8 consistency monitoring", consistency)

    def drift():
        import duckdb
        from feature_platform import config
        from feature_platform.monitor.consistency import ensure_store
        from feature_platform.monitor.drift import persist, run_drift

        payload = run_drift(days=150)
        con = duckdb.connect(str(config.DB_PATH), read_only=False)
        try:
            ensure_store(con)
            con.execute("DELETE FROM monitor_drift")
            n = persist(con, payload)
        finally:
            con.close()
        path = ROOT / "reports" / "drift.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: v for k, v in payload.items() if k != "metrics"},
                                   indent=2, default=str))
        print(f"persisted {n} drift metrics across {payload['days_evaluated']} days")
    stage("7/8 drift monitoring", drift)

    def model():
        from feature_platform.audit.model_eval import run_benchmark

        res = run_benchmark()
        path = ROOT / "reports" / "model_benchmark.json"
        path.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res[k] for k in ("train_auc", "valid_auc", "test_auc",
                                             "test_auc_with_leaky_features")}, indent=1))
    stage("8/8 model benchmark", model)

    print("\nAll artifacts built. Start the dashboard with:")
    print("  uvicorn feature_platform.api.app:app --port 8000")


if __name__ == "__main__":
    main()
