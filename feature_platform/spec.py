"""Declarative feature specifications: models, validation, compilation inputs.

A feature is defined once in YAML.  The same object is compiled to

* DuckDB SQL by :mod:`feature_platform.offline.compiler`
* a Python execution plan by :mod:`feature_platform.online.engine`

Point-in-time semantics
-----------------------
Every source declares

``event_time``      the business timestamp of the row (when the thing happened)
``knowledge_time``  when the warehouse learned about the row (ingest/commit)
``valid_until``     optional: when the row stopped being true (cancellation,
                    payment reversal, ...)

A row may contribute to a feature evaluated at ``as_of`` iff::

    event_time      <  as_of          (strictly past)
    knowledge_time  <= as_of          (already ingested)
    valid_until     IS NULL OR > as_of  (had not been retracted yet)

This triple is what makes backfills (late-arriving feeds), restatements
(append-only duplicate rows), and reversals safe.  The leak audit in
:mod:`feature_platform.audit.leak` proves that every training row obeys it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from feature_platform import config, expr

# --------------------------------------------------------------------------
# Small time helpers (shared by offline and online; keep the units identical)
# --------------------------------------------------------------------------

_WINDOW_RE = re.compile(r"^\s*(\d+)\s*(s|m|h|d|w)\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_UNIT_SQL = {"s": "SECOND", "m": "MINUTE", "h": "HOUR", "d": "DAY", "w": "WEEK"}


@dataclass(frozen=True)
class Window:
    raw: str
    count: int
    unit: str

    @property
    def seconds(self) -> int:
        return self.count * _UNIT_SECONDS[self.unit]

    @property
    def sql_interval(self) -> str:
        return f"INTERVAL {self.count} {_UNIT_SQL[self.unit]}"


def parse_window(text: str) -> Window:
    m = _WINDOW_RE.match(text or "")
    if not m:
        raise ValueError(f"bad window {text!r}; expected e.g. '7d', '24h', '30m'")
    return Window(text, int(m.group(1)), m.group(2).lower())


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

FEATURE_TYPES = {"int", "float", "string", "bool"}
AGGREGATIONS = {"count", "sum", "avg", "min", "max", "count_distinct", "last", "first"}
SOURCE_KINDS = {"event", "scd", "static"}
FEATURE_KINDS = {"window", "asof_attr", "static_attr", "derived", "constant"}


@dataclass
class DeferredSpec:
    """A column whose value only becomes known later, and may later be revoked.

    ``known_at``    -- column holding the timestamp at which the value became
                       observable (e.g. ``repay_time`` for ``amount_paid``).
    ``cleared_at``  -- optional column holding the timestamp at which the value
                       stopped being true (e.g. ``reversal_time``); NULL means
                       it never stopped being true.
    """
    known_at: str
    cleared_at: str | None = None

    @classmethod
    def parse(cls, raw: Any) -> "DeferredSpec":
        if isinstance(raw, str):
            return cls(known_at=raw)
        if isinstance(raw, dict):
            return cls(known_at=raw["known_at"], cleared_at=raw.get("cleared_at"))
        raise ValueError(f"bad deferred spec {raw!r}")


@dataclass
class SourceSpec:
    name: str
    kind: str                       # event | scd | static
    entity_key: str
    event_time: str
    knowledge_time: str
    table: str | None = None
    sql: str | None = None
    valid_until: str | None = None
    dedupe_key: str | None = None
    dedupe_order: str | None = None
    # column -> the column that records when that column's value became known.
    # Used by the static analyser to flag unguarded use of future knowledge.
    deferred: dict[str, DeferredSpec] = field(default_factory=dict)
    forbidden: set[str] = field(default_factory=set)
    description: str = ""
    columns: list[str] = field(default_factory=list)

    @property
    def relation_sql(self) -> str:
        if self.sql:
            return f"({self.sql})"
        if self.table:
            return self.table
        raise ValueError(f"source {self.name!r} needs table or sql")

    def validate(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"source {self.name}: bad kind {self.kind!r}")
        if not (self.table or self.sql):
            raise ValueError(f"source {self.name}: table or sql required")


@dataclass
class EncoderSpec:
    """Frozen category dictionary.

    ``dictionary`` is the complete set of categories the model was trained on.
    Anything else maps to ``default`` (an explicit UNKNOWN bucket), which is
    what keeps online scoring from ever KeyError-ing on a new category.
    """
    dictionary: list[str]
    default: str = "__UNKNOWN__"

    def mapping(self) -> dict[str, int]:
        m = {str(c): i for i, c in enumerate(self.dictionary)}
        if self.default not in m:
            m[self.default] = len(m)
        return m

    def code_for(self, value: Any) -> int:
        if value is None:
            return self.mapping()[self.default]
        return self.mapping().get(str(value), self.mapping()[self.default])


@dataclass
class FeatureSpec:
    name: str
    type: str = "float"
    kind: str = "window"
    description: str = ""
    # window
    source: str | None = None
    window: str | None = None
    agg: str | None = None
    column: str | None = None
    where: str | None = None
    # asof_attr / static_attr
    attr: str | None = None
    # derived / constant
    expr: str | None = None
    value: Any = None
    fill: Any = None
    transform: str | None = None      # e.g. days_since (timestamp -> age in days at as_of)
    encoder: EncoderSpec | None = None
    tags: list[str] = field(default_factory=list)
    _window: Window | None = field(default=None, repr=False)
    _expr_node: expr.Node | None = field(default=None, repr=False)
    _where_node: expr.Node | None = field(default=None, repr=False)

    @property
    def window_obj(self) -> Window:
        if self._window is None:
            self._window = parse_window(self.window or "")
        return self._window

    @property
    def expr_node(self) -> expr.Node:
        if self._expr_node is None:
            self._expr_node = expr.parse(self.expr or "")
        return self._expr_node

    @property
    def where_node(self) -> expr.Node | None:
        if self._where_node is None and self.where:
            self._where_node = expr.parse(self.where)
        return self._where_node

    def validate(self, sources: dict[str, SourceSpec]) -> None:
        if self.type not in FEATURE_TYPES:
            raise ValueError(f"feature {self.name}: bad type {self.type!r}")
        if self.kind not in FEATURE_KINDS:
            raise ValueError(f"feature {self.name}: bad kind {self.kind!r}")
        if self.kind == "window":
            if self.source not in sources:
                raise ValueError(f"feature {self.name}: unknown source {self.source!r}")
            src = sources[self.source]
            self.window_obj  # validate
            if self.agg not in AGGREGATIONS:
                raise ValueError(f"feature {self.name}: bad agg {self.agg!r}")
            if self.agg not in ("count",) and not self.column:
                raise ValueError(f"feature {self.name}: agg {self.agg} needs column")
            if self.column:
                # Aggregating a deferred column is well-defined: the compiler and
                # the online engine mask it to NULL until it is known (and again
                # after it is revoked), so the aggregate can only see facts.
                _assert_allowed(self.name, src, self.column)
            if self.where:
                # where-clauses are evaluated over the *masked* source (deferred
                # values are NULL until known), so they may reference them.
                _check_expression_columns(self.name, self.where_node, src, guarded=False)
        elif self.kind == "asof_attr":
            if self.source not in sources:
                raise ValueError(f"feature {self.name}: unknown source {self.source!r}")
            src = sources[self.source]
            if src.kind != "scd":
                raise ValueError(f"feature {self.name}: asof_attr requires an scd source, {src.name} is {src.kind}")
            if not self.attr:
                raise ValueError(f"feature {self.name}: asof_attr needs attr")
            _assert_allowed(self.name, src, self.attr)
        elif self.kind == "static_attr":
            if self.source not in sources:
                raise ValueError(f"feature {self.name}: unknown source {self.source!r}")
            src = sources[self.source]
            if src.kind != "static":
                raise ValueError(f"feature {self.name}: static_attr requires a static source")
            if not self.attr:
                raise ValueError(f"feature {self.name}: static_attr needs attr")
            _assert_allowed(self.name, src, self.attr)
        elif self.kind == "derived":
            if not self.expr:
                raise ValueError(f"feature {self.name}: derived needs expr")
            self.expr_node  # parse now
        elif self.kind == "constant":
            if self.value is None:
                raise ValueError(f"feature {self.name}: constant needs value")
        if self.transform is not None:
            if self.transform != "days_since":
                raise ValueError(f"feature {self.name}: unknown transform {self.transform!r}")
            if self.kind not in ("static_attr", "asof_attr", "window"):
                raise ValueError(f"feature {self.name}: days_since transform requires a timestamp-valued feature")
        if self.encoder is not None:
            if not self.encoder.dictionary:
                raise ValueError(f"feature {self.name}: encoder dictionary is empty")
            if self.encoder.default in self.encoder.dictionary:
                # de-duplicate: default bucket gets the last index
                pass
            if self.type not in ("int", "float"):
                raise ValueError(f"feature {self.name}: encoded features must be int/float")


def _assert_allowed(feature: str, src: SourceSpec, column: str) -> None:
    if column in src.forbidden:
        raise ValueError(
            f"feature {feature}: column {column!r} is outcome-only in source {src.name!r} "
            f"(forbidden: {sorted(src.forbidden)})"
        )
    if src.columns and column not in src.columns:
        raise ValueError(f"feature {feature}: column {column!r} not in source {src.name!r}")


def _check_expression_columns(feature: str, node: expr.Node, src: SourceSpec, guarded: bool) -> None:
    cols = expr.referenced_columns(node)
    missing = cols - set(src.columns)
    if missing:
        raise ValueError(f"feature {feature}: unknown columns {sorted(missing)} in source {src.name!r}")
    bad = cols & src.forbidden
    if bad:
        raise ValueError(f"feature {feature}: forbidden outcome columns {sorted(bad)}")
    # Deferred columns are allowed only if their knowledge column is also
    # referenced in the filter -- a cheap static guard.  The empirical audit
    # is the authoritative proof.
    if guarded:
        for col, dspec in src.deferred.items():
            known_at = dspec.known_at
            if col in cols and known_at not in cols and known_at not in expr.referenced_context(node):
                raise ValueError(
                    f"feature {feature}: uses deferred column {col!r} without guarding on "
                    f"{known_at!r} (that value is future knowledge at apply time)"
                )


@dataclass
class ViewSpec:
    name: str
    description: str = ""
    owner: str = "risk"
    entity: str = "user"
    online_ttl_seconds: int = 300
    features: list[FeatureSpec] = field(default_factory=list)

    def by_name(self) -> dict[str, FeatureSpec]:
        return {f.name: f for f in self.features}

    def ordered(self) -> list[FeatureSpec]:
        """Topologically sort derived features after their dependencies."""
        by = self.by_name()
        seen: dict[str, int] = {}
        out: list[FeatureSpec] = []

        def visit(f: FeatureSpec, stack: tuple[str, ...]) -> None:
            state = seen.get(f.name, 0)
            if state == 2:
                return
            if state == 1:
                raise ValueError(f"cycle in derived features: {' -> '.join(stack + (f.name,))}")
            seen[f.name] = 1
            if f.kind == "derived":
                for dep in expr.referenced_columns(f.expr_node):
                    if dep in by:
                        visit(by[dep], stack + (f.name,))
                    elif dep in _CONTEXT_NAMES:
                        continue
                    else:
                        raise ValueError(f"derived feature {f.name!r} references unknown feature {dep!r}")
            seen[f.name] = 2
            out.append(f)

        for f in self.features:
            visit(f, ())
        return out

    def validate(self, sources: dict[str, SourceSpec]) -> None:
        if not self.features:
            raise ValueError(f"view {self.name}: no features")
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"view {self.name}: duplicate features {dup}")
        for f in self.features:
            f.validate(sources)
        self.ordered()  # cycle + reference check
        # derived features may not shadow raw context names
        for f in self.features:
            if f.name in _CONTEXT_NAMES:
                raise ValueError(f"feature name {f.name!r} is reserved")


_CONTEXT_NAMES = {"as_of", "entity_id", "now"}


@dataclass
class Registry:
    sources: dict[str, SourceSpec]
    views: dict[str, ViewSpec]
    entity: str = "user"
    entity_key: str = "user_id"
    base_table: str = "applications"
    base_id: str = "application_id"
    base_time: str = "apply_time"
    base_entity: str = "user_id"
    label_table: str = "labels"
    version: str = "v1"
    raw: dict[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Registry":
        payload = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Registry":
        if not isinstance(payload, dict):
            raise ValueError("feature spec must be a mapping")
        sources: dict[str, SourceSpec] = {}
        for name, raw in (payload.get("sources") or {}).items():
            src = SourceSpec(
                name=name,
                kind=raw.get("kind", "event"),
                entity_key=raw.get("entity_key", "user_id"),
                event_time=raw.get("event_time", "event_time"),
                knowledge_time=raw.get("knowledge_time", "ingested_at"),
                table=raw.get("table"),
                sql=raw.get("sql"),
                valid_until=raw.get("valid_until"),
                dedupe_key=raw.get("dedupe_key"),
                dedupe_order=raw.get("dedupe_order"),
                deferred={k: DeferredSpec.parse(v) for k, v in (raw.get("deferred") or {}).items()},
                forbidden=set(raw.get("forbidden") or []),
                description=raw.get("description", ""),
                columns=list(raw.get("columns") or []),
            )
            src.validate()
            sources[name] = src

        views: dict[str, ViewSpec] = {}
        for raw in payload.get("feature_views") or []:
            feats = []
            for fr in raw.get("features") or []:
                enc = fr.get("encoder")
                feats.append(FeatureSpec(
                    name=fr["name"],
                    type=fr.get("type", "float"),
                    kind=fr.get("kind", "window"),
                    description=fr.get("description", ""),
                    source=fr.get("source"),
                    window=fr.get("window"),
                    agg=fr.get("agg"),
                    column=fr.get("column"),
                    where=fr.get("where"),
                    attr=fr.get("attr"),
                    expr=fr.get("expr"),
                    value=fr.get("value"),
                    fill=fr.get("fill"),
                    transform=fr.get("transform"),
                    encoder=EncoderSpec(dictionary=list(enc["dictionary"]), default=enc.get("default", "__UNKNOWN__")) if enc else None,
                    tags=list(fr.get("tags") or []),
                ))
            view = ViewSpec(
                name=raw["name"],
                description=raw.get("description", ""),
                owner=raw.get("owner", "risk"),
                entity=raw.get("entity", "user"),
                online_ttl_seconds=int(raw.get("online_ttl_seconds", 300)),
                features=feats,
            )
            view.validate(sources)
            views[view.name] = view

        base = payload.get("base") or {}
        reg = cls(
            sources=sources,
            views=views,
            entity=payload.get("entity", "user"),
            entity_key=payload.get("entity_key", "user_id"),
            base_table=base.get("table", "applications"),
            base_id=base.get("id", "application_id"),
            base_time=base.get("time", "apply_time"),
            base_entity=base.get("entity", "user_id"),
            label_table=payload.get("label_table", "labels"),
            version=str(payload.get("version", "v1")),
            raw=payload,
        )
        reg._validate_registry()
        return reg

    def _validate_registry(self) -> None:
        if not self.views:
            raise ValueError("no feature_views defined")
        for view in self.views.values():
            for f in view.features:
                if f.encoder is not None and f.kind not in ("static_attr", "asof_attr", "window", "derived"):
                    raise ValueError(f"{view.name}.{f.name}: encoder only valid on value features")

    # -- helpers ---------------------------------------------------------
    def all_features(self) -> Iterable[tuple[str, FeatureSpec]]:
        for view in self.views.values():
            for f in view.features:
                yield view.name, f

    def dependencies(self, view: ViewSpec, feature: FeatureSpec) -> set[str]:
        """Physical sources a feature depends on (for cache invalidation)."""
        if feature.kind in ("window", "asof_attr", "static_attr"):
            return {feature.source}  # type: ignore[arg-type]
        if feature.kind == "derived":
            deps: set[str] = set()
            by = view.by_name()
            for ref in expr.referenced_columns(feature.expr_node):
                if ref in by:
                    deps |= self.dependencies(view, by[ref])
                elif ref in self.sources:
                    deps.add(ref)
            return deps
        return set()

    def view_dependencies(self, view: ViewSpec) -> set[str]:
        deps: set[str] = set()
        for f in view.features:
            deps |= self.dependencies(view, f)
        return deps

    def fingerprint(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_default() -> Registry:
    return Registry.from_yaml(config.SPEC_PATH)


# --------------------------------------------------------------------------
# Frozen encodings
# --------------------------------------------------------------------------

def build_encodings(reg: Registry) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view in reg.views.values():
        for f in view.features:
            if f.encoder is None:
                continue
            m = f.encoder.mapping()
            out[f"{view.name}.{f.name}"] = {
                "feature": f.name,
                "dictionary": list(m.keys()),
                "mapping": m,
                "default": f.encoder.default,
                "default_code": m[f.encoder.default],
                "unknown_policy": "map_to_default",
                "dictionary_hash_sha256_16": hashlib.sha256(
                    json.dumps(list(m.keys()), ensure_ascii=False).encode()
                ).hexdigest()[:16],
            }
    return out


def write_encodings(reg: Registry, path: Path | None = None) -> Path:
    path = path or (config.ONLINE_INDEX_DIR / "encodings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": reg.version,
        "spec_fingerprint": reg.fingerprint(),
        "generated_from": str(config.SPEC_PATH),
        "encodings": build_encodings(reg),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def load_encodings(path: Path | None = None) -> dict[str, Any]:
    path = path or (config.ONLINE_INDEX_DIR / "encodings.json")
    return json.loads(path.read_text())
