"""Frozen category encodings (requirement 7).

The dictionary is built **once** from the training window and then frozen.
Serving never rebuilds it: an unseen category maps to an explicit
``__UNKNOWN__`` bucket whose code is also frozen.  This makes encoding
reproducible across offline/online and makes "new category" a monitored,
explicit event rather than a silent shape change or a KeyError at 3am.

The frozen artifact records:

* the ordered dictionary and the code for each category
* the default/unknown code
* a hash so a change to the table is detectable
* the training window it was derived from

``discover_categories`` is a *build-time* helper only; the serving path reads
``encodings.json`` exclusively.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from feature_platform import config, spec


def discover_categories(
    view_name: str = "user_risk_v1",
    start: str = "2024-01-01",
    end: str | None = "2025-01-01",
    max_categories: int = 64,
    registry: spec.Registry | None = None,
    con=None,
) -> dict[str, list[str]]:
    """Observe the raw category values of every encoded feature in a window.

    Only categories that occur in the training window are candidates; the
    spec's hand-curated dictionary stays authoritative and is merged with what
    is observed (so an expected category is never dropped just because it was
    rare).  Frequencies are returned for the dashboard.
    """
    reg = registry or spec.load_default()
    own = con is None
    if own:
        con = duckdb.connect(str(config.DB_PATH), read_only=True)
    view = reg.views[view_name]
    out: dict[str, Any] = {}
    for f in view.features:
        if f.encoder is None:
            continue
        if f.kind == "window" and f.column:
            src = reg.sources[f.source]  # type: ignore[index]
            where = []
            if start:
                where.append(f'"{src.event_time}" >= TIMESTAMP \'{start}\'')
            if end:
                where.append(f'"{src.event_time}" < TIMESTAMP \'{end}\'')
            where_sql = ("WHERE " + " AND ".join(where)) if where else ""
            rows = con.execute(
                f'SELECT "{f.column}" AS v, count(*) c FROM {src.relation_sql} {where_sql} '
                f'GROUP BY 1 ORDER BY 2 DESC LIMIT {max_categories}'
            ).fetchdf()
            observed = [str(v) for v in rows["v"] if v is not None and str(v) != "None"]
            freq = {str(r.v): int(r.c) for r in rows.itertuples() if r.v is not None}
        elif f.kind in ("static_attr", "asof_attr"):
            src = reg.sources[f.source]  # type: ignore[index]
            rows = con.execute(
                f'SELECT "{f.attr}" AS v, count(*) c FROM {src.relation_sql} '
                f"GROUP BY 1 ORDER BY 2 DESC LIMIT {max_categories}"
            ).fetchdf()
            observed = [str(v) for v in rows["v"] if v is not None and str(v) != "None"]
            freq = {str(r.v): int(r.c) for r in rows.itertuples() if r.v is not None}
        else:
            continue
        merged = list(dict.fromkeys(list(f.encoder.dictionary) + observed))
        out[f.name] = {"observed": freq, "merged_dictionary": merged}
    if own:
        con.close()
    return out


def freeze(
    view_name: str = "user_risk_v1",
    start: str = "2024-01-01",
    end: str | None = "2025-01-01",
    registry: spec.Registry | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    reg = registry or spec.load_default()
    path = out_path or (config.ONLINE_INDEX_DIR / "encodings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = spec.build_encodings(reg)
    payload = {
        "spec_version": reg.version,
        "spec_fingerprint": reg.fingerprint(),
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "training_window": {"start": start, "end": end},
        "policy": {
            "unknown": "map to explicit default bucket",
            "rebuild": "never at serving time; rebuild only via this CLI and a model release",
        },
        "encodings": payload,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="user_risk_v1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--observe", action="store_true", help="also print observed frequencies")
    args = ap.parse_args()
    payload = freeze(args.view, args.start, args.end)
    print(json.dumps({"frozen": len(payload["encodings"]),
                      "path": str(config.ONLINE_INDEX_DIR / "encodings.json")}, indent=1))
    for k, v in payload["encodings"].items():
        print(f"  {k:34s} codes={len(v['dictionary'])} default={v['default']}({v['default_code']}) "
              f"hash={v['dictionary_hash_sha256_16']}")
    if args.observe:
        obs = discover_categories(args.view, args.start, args.end)
        print(json.dumps(obs, indent=1))


if __name__ == "__main__":
    main()
