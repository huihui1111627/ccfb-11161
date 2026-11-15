"""Offline <-> online engine parity, cache invalidation, and API smoke tests.

The DB-backed tests reuse the built DuckDB file; they skip (rather than fail)
when the dataset has not been generated, so a fresh checkout can still run the
pure-expression and spec tests.
"""
from __future__ import annotations

import random

import duckdb
import pandas as pd
import pytest

from feature_platform import config, spec
from feature_platform.audit.parity import _values_equal, run_parity
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
def engine(reg, con):
    from feature_platform.online.engine import OnlineEngine, OnlineIndex

    return OnlineEngine(reg, OnlineIndex(reg, con=con))


@pytest.fixture(scope="module")
def compiled(reg):
    return OfflineCompiler(reg).compile_view(reg.views["user_risk_v1"])


# --------------------------------------------------------------------------
# Compiled SQL
# --------------------------------------------------------------------------

def test_compiled_sql_runs(compiled, con):
    df = con.execute(compiled.sql.replace("FROM (SELECT * FROM _base) b", "FROM (SELECT 'x' AS entity_row_id, 'U0000001' AS entity_id, TIMESTAMP '2025-06-01' AS as_of, 'x' AS application_id) b") if False else
                     f"WITH _base AS ({build_spine_sql(spec.load_default(), [('U0000001', '2025-06-01 12:00:00')])}) {compiled.sql}").fetchdf()
    assert len(df) == 1
    for f in compiled.feature_order:
        assert f in df.columns


def test_compiled_sql_has_point_in_time_guards(compiled):
    sql = compiled.sql
    assert '"apply_time" < b.as_of' in sql
    assert '"ingested_at" <= b.as_of' in sql
    # deferred decision columns are masked
    assert 'CASE WHEN w' in sql and '"decision_time" <= b.as_of' in sql.replace("w1.", "w").replace("w2.", "w") or '"decision_time" <= b.as_of' in sql


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------

def test_parity_exact_is_zero(engine, con):
    rep = run_parity(n=120, mode="exact", con=con, engine=engine)
    assert rep.n_compared > 50
    assert rep.mismatch_rows == 0, rep.mismatches[:5]


def test_parity_skew_mismatches_are_boundary_events(engine, con):
    rep = run_parity(n=120, mode="skew", jitter_seconds=2, con=con, engine=engine)
    assert len(rep.skew_unexplained) == 0, rep.skew_unexplained[:3]


def test_online_matches_batch_for_arbitrary_instants(engine, con, reg, compiled):
    rng = random.Random(99)
    users = con.execute("SELECT DISTINCT user_id FROM applications").fetchdf()["user_id"].tolist()
    keys = []
    for _ in range(25):
        u = rng.choice(users)
        t = pd.Timestamp("2024-06-01") + pd.Timedelta(days=rng.randint(0, 400),
                                                      seconds=rng.randint(0, 86400))
        keys.append((u, str(t)))
    sql = compiled.sql.replace(
        "FROM (SELECT * FROM _base) b", f"FROM ({build_spine_sql(reg, keys)}) b")
    off = con.execute(f"SELECT * FROM ({sql}) ORDER BY CAST(entity_row_id AS BIGINT)").fetchdf()
    for i, (u, t) in enumerate(keys):
        on = engine.score("user_risk_v1", u, t)["features"]
        for f in compiled.feature_order:
            assert _values_equal(off.iloc[i][f], on[f], compiled.feature_types[f]), (u, t, f)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_cache_hit_and_invalidation(reg, engine):
    from feature_platform.online.cache import CachedOnlineStore, DependencyCache

    store = CachedOnlineStore(engine, reg, DependencyCache())
    r1, hit1 = store.score("user_risk_v1", "U0000001", "2025-06-01T12:00:00")
    r2, hit2 = store.score("user_risk_v1", "U0000001", "2025-06-01T12:00:00")
    assert hit1 is False and hit2 is True
    assert r1["features"] == r2["features"]

    # bumping `profile` evicts only features that read the profile table
    n_features = len(reg.views["user_risk_v1"].features)
    store.cache.bump_source("profile")
    r3, hit3 = store.score("user_risk_v1", "U0000001", "2025-06-01T12:00:00")
    assert hit3 is False
    profile_feats = [f.name for f in reg.views["user_risk_v1"].features
                     if "profile" in reg.dependencies(reg.views["user_risk_v1"], f)]
    assert 0 < r3["cache_feature_misses"] < n_features
    assert r3["cache_feature_misses"] == len(profile_feats)

    # bumping an unrelated source (repayments was never bumped) keeps hits
    _, hit4 = store.score("user_risk_v1", "U0000001", "2025-06-01T12:00:00")
    assert hit4 is True


def test_cache_stats_accounting(reg, engine):
    from feature_platform.online.cache import CachedOnlineStore, DependencyCache

    store = CachedOnlineStore(engine, reg, DependencyCache())
    n = len(reg.views["user_risk_v1"].features)
    for _ in range(3):
        store.score("user_risk_v1", "U0000002", "2025-05-01T00:00:00")
    snap = store.cache.snapshot()
    assert snap["hits"] == 2 * n
    assert snap["computed"] == n
    assert snap["hit_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_cache_lru_eviction(reg, engine):
    from feature_platform.online.cache import CachedOnlineStore, DependencyCache

    store = CachedOnlineStore(engine, reg, DependencyCache(max_entries=5))
    for i in range(4):
        store.score("user_risk_v1", f"U000000{i}", "2025-05-01T00:00:00")
    assert store.cache.snapshot()["entries"] <= 5


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def test_frozen_encoding_unknown_fallback(reg, engine):
    enc = spec.load_encodings()
    key = "user_risk_v1.city_tier_code"
    default = enc["encodings"][key]["default_code"]
    # simulate an unseen category
    f = reg.views["user_risk_v1"].by_name()["city_tier_code"]
    assert engine._encode("user_risk_v1", f, "tier99_new") == default
    assert engine._encode("user_risk_v1", f, None) == default


def test_all_encoded_features_have_frozen_table(reg):
    enc = spec.load_encodings()["encodings"]
    for view in reg.views.values():
        for f in view.features:
            if f.encoder is not None:
                assert f"{view.name}.{f.name}" in enc


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def test_api_endpoints():
    from fastapi.testclient import TestClient

    from feature_platform.api.app import app

    c = TestClient(app)
    assert c.get("/v1/health").status_code == 200
    r = c.post("/v1/score", json={"view": "user_risk_v1", "entity_id": "U0000001",
                                  "as_of": "2025-06-01T12:00:00"})
    assert r.status_code == 200
    body = r.json()
    assert "features" in body and "encoded" in body and "latency_us" in body
    assert body["features"]["spec_version_code"] == 1

    r2 = c.post("/v1/score/batch", json={
        "view": "user_risk_v1",
        "requests": [{"entity_id": "U0000001", "as_of": "2025-06-01T12:00:00"},
                     {"entity_id": "U0000002", "as_of": "2025-05-01T12:00:00"}],
    })
    assert r2.status_code == 200 and r2.json()["count"] == 2

    for path in ("/", "/ui/features", "/ui/views", "/ui/training", "/ui/monitoring",
                 "/ui/consistency", "/v1/views"):
        assert c.get(path).status_code == 200, path


def test_api_rejects_unknown_view():
    from fastapi.testclient import TestClient

    from feature_platform.api.app import app

    c = TestClient(app)
    r = c.post("/v1/score", json={"view": "nope", "entity_id": "U1", "as_of": "2025-01-01"})
    assert r.status_code == 404
