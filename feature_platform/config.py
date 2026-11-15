"""Global configuration: paths, time semantics, deterministic seed."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FP_DATA_DIR", ROOT / "data"))
REPORTS_DIR = Path(os.environ.get("FP_REPORTS_DIR", ROOT / "reports"))
WEB_DIR = ROOT / "feature_platform" / "web"

DB_PATH = DATA_DIR / "feature_platform.duckdb"
ONLINE_INDEX_DIR = DATA_DIR / "online_index"
SPEC_PATH = ROOT / "feature_platform" / "features" / "specs.yaml"

# All timestamps in the platform are UTC-naive: "a naive timestamp is a
# wall-clock reading in the business timezone".  Mixing tz-aware and naive
# values is one of the classic sources of point-in-time leaks, so the whole
# codebase normalises to UTC-naive at the boundary.
SEED = 20241116
TODAY = "2025-11-15"          # last day covered by the synthetic dataset
TRAIN_START = "2023-11-16"    # two years of history
DATA_START = "2023-06-01"     # extra warm-up so first-day windows are populated

# Event-time clock skew tolerance: a record may be *recorded* slightly after
# its business timestamp, but for feature correctness the only thing that
# matters is the business timestamp (`apply_time`, `loan_time`, ...).
INGEST_LAG_MAX_HOURS = 48

# Label definition: a loan is "bad" if it ever reaches 30+ DPD within 90 days.
LABEL_DPD_THRESHOLD = 30
LABEL_WINDOW_DAYS = 90
