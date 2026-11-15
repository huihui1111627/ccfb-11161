"""HTTP API for the feature platform (FastAPI).

Routes
------
UI
  GET  /                          dashboard
  GET  /ui/features               feature definitions
  GET  /ui/views                  feature views
  GET  /ui/training               training sample builder
  GET  /ui/monitoring             online service monitoring
  GET  /ui/consistency            offline/online consistency
  GET  /ui/audit                  leak audit & encoding registry

Serving
  POST /v1/score                  single-entity feature vector at an instant
  POST /v1/score/batch            batch of (entity, as_of)
  GET  /v1/views                  view catalogue
  GET  /v1/views/{view}/features  compiled feature metadata + SQL
  GET  /v1/views/{view}/encodings frozen encoding tables
  GET  /v1/health                 liveness + latency + cache stats

Ops
  GET  /v1/monitor/latency        serving latency percentiles
  GET  /v1/monitor/consistency    recent consistency runs
  GET  /v1/monitor/drift          drift metrics
  POST /v1/monitor/consistency/run   trigger a consistency check
  GET  /v1/audit/leak             last leak-audit report
  POST /v1/admin/reload           reload spec + index (explicit, audited)
  POST /v1/admin/invalidate       bump a source version (cache invalidation demo)
"""
from __future__ import annotations

import json
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from feature_platform import config
from feature_platform.api.state import get_state
from feature_platform.offline.compiler import OfflineCompiler

app = FastAPI(title="Feature Platform", version="1.0")
templates = Jinja2Templates(directory=str(config.WEB_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(config.WEB_DIR / "static")), name="static")


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    view: str = "user_risk_v1"
    entity_id: str = Field(..., description="user_id")
    as_of: str = Field(..., description="ISO-8601 instant, e.g. 2025-06-01T12:00:00")
    use_cache: bool = True


class BatchScoreRequest(BaseModel):
    view: str = "user_risk_v1"
    requests: list[dict[str, str]] = Field(..., description='[{"entity_id": "...", "as_of": "..."}]')
    use_cache: bool = True


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

@app.post("/v1/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    st = get_state()
    if req.view not in st.registry.views:  # type: ignore[union-attr]
        raise HTTPException(404, f"unknown view {req.view}")
    try:
        return st.score(req.view, req.entity_id, req.as_of, req.use_cache)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/score/batch")
def score_batch(req: BatchScoreRequest) -> dict[str, Any]:
    st = get_state()
    if req.view not in st.registry.views:  # type: ignore[union-attr]
        raise HTTPException(404, f"unknown view {req.view}")
    results = []
    for item in req.requests:
        try:
            results.append(st.score(req.view, item["entity_id"], item["as_of"], req.use_cache))
        except Exception as exc:  # per-item failure must not kill the batch
            results.append({"entity_id": item.get("entity_id"), "error": str(exc)})
    return {"count": len(results), "results": results}


@app.get("/v1/views")
def views() -> dict[str, Any]:
    st = get_state()
    out = []
    for name, v in st.registry.views.items():  # type: ignore[union-attr]
        out.append({
            "name": name,
            "description": v.description,
            "owner": v.owner,
            "entity": v.entity,
            "n_features": len(v.features),
            "online_ttl_seconds": v.online_ttl_seconds,
            "dependencies": sorted(st.registry.view_dependencies(v)),  # type: ignore[union-attr]
        })
    return {"views": out, "spec_fingerprint": st.registry.fingerprint()}  # type: ignore[union-attr]


@app.get("/v1/views/{view_name}/features")
def view_features(view_name: str) -> dict[str, Any]:
    st = get_state()
    reg = st.registry
    if view_name not in reg.views:  # type: ignore[union-attr]
        raise HTTPException(404, f"unknown view {view_name}")
    view = reg.views[view_name]  # type: ignore[union-attr]
    cv = OfflineCompiler(reg).compile_view(view)  # type: ignore[arg-type]
    feats = []
    for f in view.ordered():
        feats.append({
            "name": f.name,
            "kind": f.kind,
            "type": f.type,
            "source": f.source,
            "window": f.window,
            "agg": f.agg,
            "column": f.column,
            "where": f.where,
            "attr": f.attr,
            "expr": f.expr,
            "fill": f.fill,
            "tags": f.tags,
            "description": f.description,
            "encoded": f.encoder is not None,
            "dependencies": sorted(reg.dependencies(view, f)),  # type: ignore[arg-type]
        })
    return {
        "view": view_name,
        "features": feats,
        "feature_order": cv.feature_order,
        "compiled_sql": cv.sql,
        "spec_fingerprint": reg.fingerprint(),  # type: ignore[union-attr]
    }


@app.get("/v1/views/{view_name}/encodings")
def view_encodings(view_name: str) -> dict[str, Any]:
    st = get_state()
    enc = {k: v for k, v in st.encodings.get("encodings", {}).items() if k.startswith(view_name + ".")}
    return {
        "view": view_name,
        "spec_fingerprint": st.encodings.get("spec_fingerprint"),
        "frozen_at": st.encodings.get("frozen_at"),
        "policy": st.encodings.get("policy"),
        "encodings": enc,
    }


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return get_state().health()


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------

@app.get("/v1/monitor/latency")
def monitor_latency() -> dict[str, Any]:
    st = get_state()
    return {"latency": st.latency.summary(), "cache": st.store.cache.snapshot() if st.store else {}}


@app.get("/v1/monitor/consistency")
def monitor_consistency(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT CAST(run_ts AS DATE) AS day, view, mode,
                   count(*) AS runs, sum(cells) AS cells, sum(mismatches) AS mismatches,
                   avg(cell_mismatch_rate) AS avg_cell_mismatch_rate,
                   max(cell_mismatch_rate) AS max_cell_mismatch_rate,
                   avg(duration_ms) AS avg_duration_ms
            FROM monitor_consistency
            GROUP BY 1,2,3 ORDER BY 1 DESC, 2
        """).fetchdf()
        runs = con.execute(
            "SELECT run_ts, view, mode, n_requests, cells, mismatches, "
            "cell_mismatch_rate, request_mismatch_rate, duration_ms "
            "FROM monitor_consistency ORDER BY run_ts DESC LIMIT 50"
        ).fetchdf()
    finally:
        con.close()
    return {
        "daily": json.loads(df.to_json(orient="records", date_format="iso")),
        "recent_runs": json.loads(runs.to_json(orient="records", date_format="iso")),
    }


@app.post("/v1/monitor/consistency/run")
def trigger_consistency(n: int = Query(100, ge=10, le=2000),
                        mode: str = Query("exact", pattern="^(exact|skew)$")) -> dict[str, Any]:
    from feature_platform.monitor.consistency import record_run, run_consistency

    run = run_consistency(n=n, mode=mode)
    con = duckdb.connect(str(config.DB_PATH), read_only=False)
    try:
        record_run(con, run)
    finally:
        con.close()
    return run.as_dict()


@app.get("/v1/monitor/drift")
def monitor_drift(days: int = Query(30, ge=1, le=365), feature: str | None = None) -> dict[str, Any]:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        where = f"AND feature = '{feature}'" if feature else ""
        # The monitor stores a synthetic-history dataset, so "recent" is
        # relative to the newest monitored day, not to wall-clock today.
        anchor = con.execute("SELECT max(run_date) FROM monitor_drift").fetchone()[0]
        if anchor is None:
            return {"series": [], "breached_features": [], "anchor_date": None}
        df = con.execute(f"""
            SELECT run_date, feature, kind, metric, value, baseline, threshold, breached
            FROM monitor_drift
            WHERE run_date >= DATE '{anchor}' - INTERVAL {int(days)} DAY {where}
            ORDER BY run_date, feature, metric
        """).fetchdf()
        summary = con.execute(f"""
            SELECT feature, metric, count(*) AS days,
                   sum(CASE WHEN breached THEN 1 ELSE 0 END) AS breaches,
                   avg(value) AS avg_value, max(value) AS max_value
            FROM monitor_drift
            WHERE run_date >= DATE '{anchor}' - INTERVAL {int(days)} DAY {where}
            GROUP BY 1,2 HAVING sum(CASE WHEN breached THEN 1 ELSE 0 END) > 0
            ORDER BY breaches DESC
        """).fetchdf()
    finally:
        con.close()
    return {
        "anchor_date": str(anchor),
        "series": json.loads(df.to_json(orient="records", date_format="iso")),
        "breached_features": json.loads(summary.to_json(orient="records", date_format="iso")),
    }


@app.get("/v1/audit/leak")
def audit_leak() -> dict[str, Any]:
    path = config.REPORTS_DIR / "leak_audit.json"
    if not path.exists():
        raise HTTPException(404, "no leak audit report; run feature_platform.audit.leak")
    return json.loads(path.read_text())


@app.get("/v1/audit/parity")
def audit_parity() -> dict[str, Any]:
    path = config.REPORTS_DIR / "parity.json"
    if not path.exists():
        raise HTTPException(404, "no parity report; run feature_platform.audit.parity")
    return json.loads(path.read_text())


@app.post("/v1/admin/reload")
def admin_reload() -> dict[str, Any]:
    st = get_state()
    st.reload()
    return {"reloaded": True, "spec_fingerprint": st.registry.fingerprint(),  # type: ignore[union-attr]
            "entities_indexed": len(st.index.entities())}  # type: ignore[union-attr]


@app.post("/v1/admin/invalidate")
def admin_invalidate(source: str = Query(..., description="source name, e.g. loans")) -> dict[str, Any]:
    st = get_state()
    if st.store is None or source not in st.registry.sources:  # type: ignore[union-attr]
        raise HTTPException(404, f"unknown source {source}")
    before = st.store.cache.snapshot()
    version = st.store.cache.bump_source(source)
    after = st.store.cache.snapshot()
    return {
        "source": source,
        "new_version": version,
        "entries_invalidated": before["entries"] - after["entries"],
        "cache": after,
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def _page(request: Request, template: str, active: str, **ctx) -> HTMLResponse:
    st = get_state()
    return templates.TemplateResponse(request, template, {
        "active": active,
        "spec_fingerprint": st.registry.fingerprint() if st.registry else "",
        "n_views": len(st.registry.views) if st.registry else 0,
        **ctx,
    })


@app.get("/", response_class=HTMLResponse)
def page_dashboard(request: Request):
    st = get_state()
    reg = st.registry
    total_features = sum(len(v.features) for v in reg.views.values())  # type: ignore[union-attr]
    leak = {}
    p = config.REPORTS_DIR / "leak_audit.json"
    if p.exists():
        leak = json.loads(p.read_text())
    parity = {}
    p2 = config.REPORTS_DIR / "parity.json"
    if p2.exists():
        parity = json.loads(p2.read_text())
    return _page(request, "dashboard.html", "dashboard",
                 views=list(reg.views.values()),  # type: ignore[union-attr]
                 total_features=total_features,
                 health=st.health(), leak=leak, parity=parity)


@app.get("/ui/features", response_class=HTMLResponse)
def page_features(request: Request):
    st = get_state()
    reg = st.registry
    cv = OfflineCompiler(reg).compile_view(reg.views["user_risk_v1"])  # type: ignore[arg-type]
    return _page(request, "features.html", "features",
                 views=list(reg.views.values()),  # type: ignore[union-attr]
                 sources=reg.sources,  # type: ignore[union-attr]
                 compiled_sql=cv.sql,
                 feature_order=cv.feature_order)


@app.get("/ui/views", response_class=HTMLResponse)
def page_views(request: Request):
    st = get_state()
    reg = st.registry
    details = []
    for name, v in reg.views.items():  # type: ignore[union-attr]
        details.append({
            "view": v,
            "deps": sorted(reg.view_dependencies(v)),  # type: ignore[union-attr]
            "sql": OfflineCompiler(reg).compile_view(v).sql,  # type: ignore[arg-type]
        })
    return _page(request, "views.html", "views", details=details)


@app.get("/ui/training", response_class=HTMLResponse)
def page_training(request: Request):
    meta_path = config.DATA_DIR / "training" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    leak_path = config.REPORTS_DIR / "leak_audit.json"
    leak = json.loads(leak_path.read_text()) if leak_path.exists() else {}
    sample_rows = []
    tp = config.DATA_DIR / "training" / "test.parquet"
    if tp.exists():
        import pandas as pd

        df = pd.read_parquet(tp)
        cols = [c for c in ("application_id", "entity_id", "__as_of", "label", "__label_ready_time") if c in df.columns]
        sample_rows = df[cols].head(40).to_dict(orient="records")
    return _page(request, "training.html", "training", meta=meta, leak=leak,
                 sample_rows=sample_rows)


@app.get("/ui/monitoring", response_class=HTMLResponse)
def page_monitoring(request: Request):
    st = get_state()
    return _page(request, "monitoring.html", "monitoring", health=st.health(),
                 encodings=st.encodings)


@app.get("/ui/consistency", response_class=HTMLResponse)
def page_consistency(request: Request):
    st = get_state()
    parity = {}
    p = config.REPORTS_DIR / "parity.json"
    if p.exists():
        parity = json.loads(p.read_text())
    return _page(request, "consistency.html", "consistency", parity=parity,
                 health=st.health())
