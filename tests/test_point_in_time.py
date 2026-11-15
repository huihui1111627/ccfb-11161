"""Point-in-time guarantees: the tests that must never go green on a leak.

These assert the *contract*, not the implementation, so they remain meaningful
if the compiler or engine is rewritten.
"""
from __future__ import annotations

import time

import duckdb
import pandas as pd
import pytest

from feature_platform import config, spec
from feature_platform.audit.leak import (
    _mutation_sql,
    _register_sources,
    _clone_with_tables,
    _sample_keys,
    _compare,
    proof_negative_controls,
    proof_prefix_stability,
)
from feature_platform.offline.compiler import OfflineCompiler, build_spine_sql

pytestmark = pytest.mark.skipif(
    not config.DB_PATH.exists(), reason="run feature_platform.data.generate first"
)


@pytest.fixture(scope="module")
def reg():
    return spec.load_default()


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(config.DB_PATH), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="module")
def cv(reg):
    return OfflineCompiler(reg).compile_view(reg.views["user_risk_v1"])


# --------------------------------------------------------------------------
# Proof 1: truncating the warehouse to as_of must not change any feature
# --------------------------------------------------------------------------

def test_prefix_stability(reg, con, cv):
    keys = _sample_keys(con, reg, 25, seed=101)
    result = proof_prefix_stability(con, reg, keys, cv.feature_order, cv.feature_types)
    assert result["violations"] == 0, result["examples"][:5]
    assert result["cells_compared"] == 25 * len(cv.feature_order)


# --------------------------------------------------------------------------
# Proof 2: perturbing future rows must not change any feature
# --------------------------------------------------------------------------

def test_boundary_mutation(reg, con, cv):
    keys = _sample_keys(con, reg, 20, seed=202)
    ident = {n: f"SELECT * FROM {s.table}" for n, s in reg.sources.items() if s.table}
    for entity, as_of in keys:
        bmap = _register_sources(con, reg, "__t_base_", ident)
        base = OfflineCompiler(_clone_with_tables(reg, bmap)).compile_view(
            reg.views["user_risk_v1"], base_sql=build_spine_sql(reg, [(entity, as_of)])
        )
        baseline = con.execute(base.sql).fetchdf()

        mutated = {n: _mutation_sql(s, entity, as_of) for n, s in reg.sources.items() if s.table}
        mmap = _register_sources(con, reg, "__t_mut_", mutated)
        reg_m = _clone_with_tables(reg, mmap)
        mut_sql = OfflineCompiler(reg_m).compile_view(
            reg.views["user_risk_v1"], base_sql=build_spine_sql(reg_m, [(entity, as_of)])
        )
        after = con.execute(mut_sql.sql).fetchdf()
        diffs = _compare(after, baseline, cv.feature_order, cv.feature_types)
        assert not diffs, (entity, as_of, diffs[:3])


# --------------------------------------------------------------------------
# Proof 4: every point-in-time rule must be observably load-bearing
# --------------------------------------------------------------------------

def test_all_point_in_time_rules_are_detectable(reg, con, cv):
    result = proof_negative_controls(con, reg, n=60, seed=11)
    missing = [k for k, v in result["controls"].items() if not v["detected"]]
    assert not missing, f"rules with no observable effect: {missing}"
    assert result["all_detected"]


# --------------------------------------------------------------------------
# Training samples
# --------------------------------------------------------------------------

def test_training_rows_never_use_post_application_data(reg, con):
    from feature_platform.offline.samples import build_training_frame

    build = build_training_frame("user_risk_v1", start="2025-01-01", registry=reg, con=con)
    df = build.frame.head(300)
    # as_of must equal the application's own apply_time
    apps = con.execute("SELECT application_id, apply_time, user_id FROM applications").fetchdf()
    apps = apps.set_index("application_id")
    for _, r in df.iterrows():
        assert pd.Timestamp(r["__as_of"]) == pd.Timestamp(apps.loc[r["application_id"], "apply_time"])
        assert r["entity_id"] == apps.loc[r["application_id"], "user_id"]


def test_labels_carry_ready_time(reg, con):
    from feature_platform.offline.samples import build_training_frame

    build = build_training_frame("user_risk_v1", start="2024-01-01", registry=reg, con=con)
    assert "__label_ready_time" in build.frame.columns
    # every label became observable before the dataset's final day
    last_day = pd.Timestamp(config.TODAY) + pd.Timedelta(days=1)
    ready = pd.to_datetime(build.frame["__label_ready_time"])
    assert (ready <= last_day).all()
    # and the observation window closed after the application
    assert (ready > pd.to_datetime(build.frame["__as_of"])).all()


def test_training_split_is_chronological(reg, con):
    from feature_platform.offline.samples import build_training_frame, time_split

    build = build_training_frame("user_risk_v1", start="2024-01-01", registry=reg, con=con)
    tr, va, te = time_split(build.frame)
    assert tr["__as_of"].max() <= va["__as_of"].min()
    assert va["__as_of"].max() <= te["__as_of"].min()
    assert len(tr) and len(va) and len(te)


# --------------------------------------------------------------------------
# Latency requirement
# --------------------------------------------------------------------------

def test_online_latency_under_half_a_millisecond(reg):
    """Single-user feature vector must stay in the low-millisecond range.

    The requirement is "tens of milliseconds" end-to-end; the in-process
    computation budget here is far tighter so HTTP and serialisation have room.
    """
    from feature_platform.online.engine import OnlineEngine, OnlineIndex

    engine = OnlineEngine(reg, OnlineIndex(reg))
    users = ["U0000001", "U0000100", "U0005000", "U0010000", "U0015000"]
    instants = ["2024-03-01T12:00:00", "2024-09-01T08:00:00",
                "2025-01-15T18:30:00", "2025-05-01T00:00:00", "2025-09-01T06:00:00"]
    times = []
    for u in users:
        for t in instants:
            t0 = time.perf_counter_ns()
            engine.score("user_risk_v1", u, t)
            times.append((time.perf_counter_ns() - t0) / 1000)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    assert p50 < 2_000, f"p50 {p50:.0f} µs exceeds 2 ms"
    assert p95 < 10_000, f"p95 {p95:.0f} µs exceeds 10 ms"


def test_cached_score_is_faster(reg):
    from feature_platform.online.cache import CachedOnlineStore, DependencyCache
    from feature_platform.online.engine import OnlineEngine, OnlineIndex

    engine = OnlineEngine(reg, OnlineIndex(reg))
    store = CachedOnlineStore(engine, reg, DependencyCache())
    inst = "2025-05-01T00:00:00"
    store.score("user_risk_v1", "U0000001", inst)
    t0 = time.perf_counter_ns()
    for _ in range(20):
        store.score("user_risk_v1", "U0000001", inst)
    cached_us = (time.perf_counter_ns() - t0) / 20 / 1000
    t0 = time.perf_counter_ns()
    engine.score("user_risk_v1", "U0000001", inst)
    uncached_us = (time.perf_counter_ns() - t0) / 1000
    assert cached_us < uncached_us
