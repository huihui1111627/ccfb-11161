"""Spec loading and validation."""
from __future__ import annotations

import copy
import pytest

from feature_platform import spec


def test_default_spec_loads():
    reg = spec.load_default()
    assert reg.views
    assert reg.entity == "user"
    assert reg.fingerprint()


def test_view_ordering_is_topological():
    reg = spec.load_default()
    order = [f.name for f in reg.views["user_risk_v1"].ordered()]
    assert order.index("app_cnt_30d") < order.index("loan_to_app_ratio_30d")
    assert order.index("app_amt_sum_30d") < order.index("amt_per_app_30d")


def test_derived_cycle_detected():
    payload = {
        "sources": {"applications": {"kind": "event", "table": "applications",
                                     "event_time": "apply_time", "knowledge_time": "ingested_at",
                                     "columns": ["user_id", "apply_time", "ingested_at"]}},
        "feature_views": [{
            "name": "v",
            "features": [
                {"name": "a", "kind": "derived", "expr": "b + 1"},
                {"name": "b", "kind": "derived", "expr": "a + 1"},
            ],
        }],
    }
    with pytest.raises(ValueError, match="cycle"):
        spec.Registry.from_dict(payload)


def test_unknown_source_rejected():
    payload = {
        "sources": {},
        "feature_views": [{"name": "v", "features": [
            {"name": "x", "kind": "window", "source": "nope", "window": "7d", "agg": "count"}
        ]}],
    }
    with pytest.raises(ValueError, match="unknown source"):
        spec.Registry.from_dict(payload)


def test_forbidden_outcome_column_rejected():
    reg = spec.load_default()
    raw = copy.deepcopy(reg.raw)
    raw["feature_views"][0]["features"].append({
        "name": "leak", "kind": "window", "source": "loans", "window": "30d",
        "agg": "max", "column": "final_dpd",
    })
    with pytest.raises(ValueError, match="outcome-only"):
        spec.Registry.from_dict(raw)


def test_deferred_column_declaration_parsed():
    payload = {
        "sources": {"applications": {
            "kind": "event", "table": "applications",
            "event_time": "apply_time", "knowledge_time": "ingested_at",
            "columns": ["user_id", "apply_time", "ingested_at", "apply_status", "decision_time"],
            "deferred": {"apply_status": "decision_time"},
        }},
        "feature_views": [{"name": "v", "features": [
            {"name": "ok", "kind": "window", "source": "applications", "window": "7d",
             "agg": "count", "column": "apply_status"},
        ]}],
    }
    reg = spec.Registry.from_dict(payload)
    d = reg.sources["applications"].deferred["apply_status"]
    assert d.known_at == "decision_time"
    assert d.cleared_at is None


def test_deferred_cleared_at_parsed():
    payload = {
        "sources": {"repayments": {
            "kind": "event", "table": "repayments",
            "event_time": "due_time", "knowledge_time": "schedule_known_at",
            "columns": ["user_id", "due_time", "schedule_known_at", "amount_paid",
                        "repay_time", "reversal_time"],
            "deferred": {"amount_paid": {"known_at": "repay_time", "cleared_at": "reversal_time"}},
        }},
        "feature_views": [{"name": "v", "features": [
            {"name": "paid", "kind": "window", "source": "repayments", "window": "30d",
             "agg": "sum", "column": "amount_paid"},
        ]}],
    }
    reg = spec.Registry.from_dict(payload)
    d = reg.sources["repayments"].deferred["amount_paid"]
    assert d.known_at == "repay_time"
    assert d.cleared_at == "reversal_time"


def test_encoder_dictionary_frozen():
    reg = spec.load_default()
    enc = spec.build_encodings(reg)
    key = "user_risk_v1.city_tier_code"
    assert key in enc
    mapping = enc[key]["mapping"]
    assert mapping["tier1"] == 0
    assert enc[key]["default_code"] == mapping["__UNKNOWN__"]


def test_encoder_unknown_maps_to_default():
    enc = spec.EncoderSpec(dictionary=["a", "b"], default="__UNKNOWN__")
    assert enc.code_for("a") == 0
    assert enc.code_for("zzz") == enc.mapping()["__UNKNOWN__"]
    assert enc.code_for(None) == enc.mapping()["__UNKNOWN__"]


def test_window_parsing():
    assert spec.parse_window("7d").seconds == 7 * 86400
    assert spec.parse_window("30m").seconds == 1800
    assert spec.parse_window("2w").sql_interval == "INTERVAL 2 WEEK"
    with pytest.raises(ValueError):
        spec.parse_window("7 days")


def test_duplicate_feature_names_rejected():
    reg = spec.load_default()
    raw = copy.deepcopy(reg.raw)
    raw["feature_views"][0]["features"].append(dict(raw["feature_views"][0]["features"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        spec.Registry.from_dict(raw)


def test_all_spec_features_have_unique_dependencies():
    reg = spec.load_default()
    for view in reg.views.values():
        names = {f.name for f in view.features}
        assert len(names) == len(view.features)
