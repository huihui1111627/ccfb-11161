"""Compile declarative feature specs to point-in-time DuckDB SQL.

Structure of a compiled view
----------------------------
::

    WITH _base AS ( ... base spine with as_of ... ),
         _src_<name> AS ( ... source with knowledge guards + knowledge masking ... )
    SELECT
        <entity columns>,
        <derived features referencing materialised features and $as_of>
    FROM (
        SELECT
            <base columns>,
            <fill / encode / cast of each window|asof|static|constant feature>
        FROM (
            SELECT
                <base columns>,
                <raw aggregate per feature>          -- correlated LATERAL
            FROM _base b
            LEFT JOIN LATERAL (...) ON TRUE          -- one per window feature
            LEFT JOIN LATERAL (...) ON TRUE          -- one per asof_attr
        ) r
    ) f

Knowledge guards (identical for every source):

* ``knowledge_time <= as_of``     -- late-arriving feed rows are invisible
* ``event_time     <  as_of``     -- strict past only
* ``valid_until    >  as_of``     -- cancellations/reversals applied
* ``dedupe_key``                  -- append-only restatements resolved
                                     *as of* as_of, not against final state
* ``deferred`` columns             -- values that only become known later are
                                     masked to NULL at as_of (T2 / T5)

The online engine implements exactly the same five rules, which is why the
offline and online values agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from feature_platform import expr, spec


@dataclass
class CompiledView:
    name: str
    sql: str
    feature_order: list[str]
    feature_types: dict[str, str]
    aliases: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# SQL fragments
# --------------------------------------------------------------------------

def _lit(v) -> str:
    return expr.Literal(v).to_sql(expr.SqlDialect())


def _mask_deferred(src: spec.SourceSpec, alias: str) -> str:
    """Projection for a source that hides values not yet known at as_of."""
    if not src.deferred:
        return f"SELECT {alias}.* FROM ({src.relation_sql}) {alias}"
    parts = []
    for col in src.columns or ["*"]:
        if col == "*":
            parts.append(f"{alias}.*")
        elif col in src.deferred:
            dspec = src.deferred[col]
            cond = f'{alias}."{dspec.known_at}" <= b.as_of'
            if dspec.cleared_at:
                cond += (
                    f' AND ({alias}."{dspec.cleared_at}" IS NULL '
                    f'OR {alias}."{dspec.cleared_at}" > b.as_of)'
                )
            parts.append(f'CASE WHEN {cond} THEN {alias}."{col}" END AS "{col}"')
        else:
            parts.append(f'{alias}."{col}"')
    return f"SELECT {', '.join(parts)} FROM ({src.relation_sql}) {alias}"


def _source_cte(src: spec.SourceSpec) -> str:
    """Named CTE for one source.

    Kept as a CTE (not inlined per feature) so DuckDB can push the entity and
    knowledge filters into a single scan per feature family, and so the audit
    can inspect exactly the same SQL the model sees.
    """
    masked = _mask_deferred(src, "s")
    return f"""
    SELECT s.*
    FROM ({masked}) s
    """


def _cast_sql(sql: str, ftype: str) -> str:
    if ftype == "int":
        return f"CAST({sql} AS BIGINT)"
    if ftype == "float":
        return f"CAST({sql} AS DOUBLE)"
    if ftype == "bool":
        return f"CAST({sql} AS BOOLEAN)"
    return f"CAST({sql} AS VARCHAR)"


def _encode_sql(raw: str, enc: spec.EncoderSpec) -> str:
    mapping = enc.mapping()
    cases = " ".join(
        f"WHEN {raw} = {_lit(cat)} THEN {code}" for cat, code in mapping.items() if cat != enc.default
    )
    default = mapping[enc.default]
    return f"CASE {cases} ELSE {default} END"


NAIVE_MODES = {"knowledge_lag", "inclusive_boundary", "no_dedupe", "no_validity",
               "latest_state", "no_deferred", "local_time_shift"}


class OfflineCompiler:
    """Compile a view to SQL.

    ``naive`` accepts any subset of:

    ``knowledge_lag``      ignore knowledge_time <= as_of (late feeds leak in)
    ``inclusive_boundary`` use ``event_time <= as_of`` instead of ``<``
    ``no_dedupe``          keep every restated loan row
    ``no_validity``        ignore cancellations / payment reversals
    ``latest_state``       asof_attr takes the latest snapshot overall
    ``no_deferred``        expose deferred columns regardless of when known
    ``local_time_shift``   parse the +08:00 local timestamp as if it were UTC,
                           shifting every application event by 8 hours (T1)

    These are *deliberately wrong* modes used only by the leak audit's
    positive-control baseline.
    """

    def __init__(self, registry: spec.Registry, naive: set[str] | None = None):
        self.reg = registry
        self.naive = naive or set()
        unknown = self.naive - NAIVE_MODES
        if unknown:
            raise ValueError(f"unknown naive modes {sorted(unknown)}")

    # ------------------------------------------------------------------
    def compile_view(self, view: spec.ViewSpec, base_sql: str | None = None) -> CompiledView:
        ordered = view.ordered()
        raw_names: dict[str, str] = {}
        joins: list[str] = []
        lateral = 0

        for f in ordered:
            if f.kind == "window":
                lateral += 1
                alias = f"w{lateral}"
                raw_names[f.name] = f"{alias}_v.v"
                joins.append(self._window_lateral(f, alias))
            elif f.kind == "asof_attr":
                lateral += 1
                alias = f"sc{lateral}"
                raw_names[f.name] = f"{alias}_v.v"
                joins.append(self._asof_lateral(f, alias))

        # -------- innermost: raw aggregates
        base_cols = (
            'b.entity_row_id, b.entity_id, b.as_of, '
            f'b."{self.reg.base_id}" AS {self.reg.base_id}'
        )
        raw_select = [base_cols]
        for f in ordered:
            if f.kind in ("window", "asof_attr"):
                raw_select.append(f'{raw_names[f.name]} AS "_raw_{f.name}"')
            elif f.kind == "static_attr":
                raw_select.append(f'{self._static_raw(f)} AS "_raw_{f.name}"')
            elif f.kind == "constant":
                raw_select.append(f'{_lit(f.value)} AS "_raw_{f.name}"')
        raw_sql = (
            f"SELECT {', '.join(raw_select)} FROM (SELECT * FROM _base) b\n"
            + "\n".join(joins)
        )

        # -------- middle: fill / transform / encode / cast
        mid_base = 'r.entity_row_id, r.entity_id, r.as_of, r."%s" AS %s' % (self.reg.base_id, self.reg.base_id)
        mid_select = [mid_base]
        for f in ordered:
            if f.kind == "derived":
                continue
            raw_ref = f'"_raw_{f.name}"'
            mid_select.append(f'{self._finalise(f, raw_ref)} AS "{f.name}"')
        mid_sql = f"SELECT {', '.join(mid_select)} FROM ({raw_sql}) r"

        # -------- outer: derived features
        out_base = 'f.entity_row_id, f.entity_id, f.as_of, f."%s" AS %s' % (self.reg.base_id, self.reg.base_id)
        out_select = [out_base]
        for f in ordered:
            if f.kind == "derived":
                d = expr.SqlDialect(context_vars={"as_of": "f.as_of"})
                expr_sql = f.expr_node.to_sql(d)
                if f.fill is not None:
                    expr_sql = f"COALESCE({expr_sql}, {_lit(f.fill)})"
                out_select.append(f'{_cast_sql(expr_sql, f.type)} AS "{f.name}"')
            else:
                out_select.append(f'f."{f.name}"')
        sql = f"SELECT {', '.join(out_select)} FROM ({mid_sql}) f"
        if base_sql:
            sql = sql.replace("FROM (SELECT * FROM _base) b", f"FROM ({base_sql}) b")
        return CompiledView(
            name=view.name,
            sql=sql,
            feature_order=[f.name for f in ordered],
            feature_types={f.name: f.type for f in ordered},
            aliases=raw_names,
        )

    # ------------------------------------------------------------------
    def _window_lateral(self, f: spec.FeatureSpec, alias: str) -> str:
        src = self.reg.sources[f.source]  # type: ignore[index]
        w = f.window_obj
        inner = self._deduped(f, src, alias)

        where_user = ""
        if f.where:
            d = expr.SqlDialect(column_prefix=f"{alias}.")
            where_user = f" AND ({f.where_node.to_sql(d)})"  # type: ignore[union-attr]

        valid = ""
        if src.valid_until and "no_validity" not in self.naive:
            valid = (
                f' AND ({alias}."{src.valid_until}" IS NULL '
                f'OR {alias}."{src.valid_until}" > b.as_of)'
            )
        upper = "<=" if "inclusive_boundary" in self.naive else "<"
        ev = self._event_time_sql(src, alias)
        window_pred = (
            f'{ev} {upper} b.as_of '
            f'AND {ev} >= b.as_of - {w.sql_interval}'
        )

        agg = self._agg_sql(f, src, alias)
        return f"""
        LEFT JOIN LATERAL (
            SELECT {agg} AS v
            FROM ({inner}) {alias}
            WHERE {window_pred}{where_user}{valid}
        ) {alias}_v ON TRUE
        """

    def _event_time_sql(self, src: spec.SourceSpec, alias: str) -> str:
        """Physical SQL for the source event time (naive modes may distort it)."""
        col = f'{alias}."{src.event_time}"'
        if "local_time_shift" in self.naive and src.name == "applications":
            # apply_time_local is UTC+08:00; naive pipelines sometimes CAST the
            # local string to TIMESTAMP and treat it as UTC.  The partition
            # column AND the watermark move together, which is exactly why this
            # bug is invisible in row counts but wrong at the boundary.
            return f'({col} - INTERVAL 8 HOUR)'
        return col

    def _deduped(self, f: spec.FeatureSpec, src: spec.SourceSpec, alias: str) -> str:
        if "no_deferred" in self.naive:
            masked = f"SELECT {alias}.* FROM ({src.relation_sql}) {alias}"
        else:
            masked = _mask_deferred(src, alias)
        knowledge = "" if "knowledge_lag" in self.naive else (
            f' AND {alias}."{src.knowledge_time}" <= b.as_of'
        )
        ent_filter = f'{alias}."{src.entity_key}" = b.entity_id'
        body = f"""
            SELECT {alias}.*
            FROM ({masked}) {alias}
            WHERE {ent_filter}{knowledge}
        """
        if src.dedupe_key and src.dedupe_order and "no_dedupe" not in self.naive:
            return f"""
            SELECT * EXCLUDE (_rn) FROM (
                SELECT {alias}.*,
                       row_number() OVER (
                           PARTITION BY {alias}."{src.dedupe_key}"
                           ORDER BY {alias}."{src.dedupe_order}" DESC
                       ) AS _rn
                FROM ({body}) {alias}
            ) {alias}
            WHERE {alias}._rn = 1
            """
        return body

    def _agg_sql(self, f: spec.FeatureSpec, src: spec.SourceSpec, alias: str) -> str:
        col = f'{alias}."{f.column}"' if f.column else None
        if f.agg == "count":
            return "count(*)"
        if f.agg == "count_distinct":
            return f"count(DISTINCT {col})"
        if f.agg in ("sum", "avg", "min", "max"):
            return f"{f.agg}({col})"
        if f.agg in ("last", "first"):
            fn = "arg_max" if f.agg == "last" else "arg_min"
            return f'{fn}({col}, {alias}."{src.event_time}")'
        raise ValueError(f"{f.name}: unsupported agg {f.agg!r}")

    def _asof_lateral(self, f: spec.FeatureSpec, alias: str) -> str:
        src = self.reg.sources[f.source]  # type: ignore[index]
        if "latest_state" in self.naive:
            # naive: "just take the latest profile row we have"
            return f"""
            LEFT JOIN LATERAL (
                SELECT {alias}."{f.attr}" AS v
                FROM {src.relation_sql} {alias}
                WHERE {alias}."{src.entity_key}" = b.entity_id
                ORDER BY {alias}."{src.event_time}" DESC
                LIMIT 1
            ) {alias}_v ON TRUE
            """
        if "no_deferred" in self.naive:
            masked = f"SELECT {alias}.* FROM ({src.relation_sql}) {alias}"
        else:
            masked = _mask_deferred(src, alias)
        knowledge = "" if "knowledge_lag" in self.naive else (
            f' AND {alias}."{src.knowledge_time}" <= b.as_of'
        )
        time_cond = "" if "latest_state" in self.naive else (
            f' AND {alias}."{src.event_time}" < b.as_of'
        )
        return f"""
        LEFT JOIN LATERAL (
            SELECT {alias}."{f.attr}" AS v
            FROM ({masked}) {alias}
            WHERE {alias}."{src.entity_key}" = b.entity_id{knowledge}{time_cond}
            ORDER BY {alias}."{src.event_time}" DESC
            LIMIT 1
        ) {alias}_v ON TRUE
        """

    def _static_raw(self, f: spec.FeatureSpec) -> str:
        src = self.reg.sources[f.source]  # type: ignore[index]
        col = f's."{f.attr}"'
        # Static dimensions are also point-in-time: a user who had not
        # registered yet at as_of must have no attributes, not future ones.
        conds = [f's."{src.entity_key}" = b.entity_id']
        if "knowledge_lag" not in self.naive:
            conds.append(f's."{src.knowledge_time}" <= b.as_of')
        time_cmp = "<=" if "inclusive_boundary" in self.naive else "<"
        if "latest_state" not in self.naive:
            conds.append(f's."{src.event_time}" {time_cmp} b.as_of')
        return f"(SELECT {col} FROM {src.relation_sql} s WHERE {' AND '.join(conds)} LIMIT 1)"

    def _finalise(self, f: spec.FeatureSpec, raw: str) -> str:
        sql = raw
        # days_since: whole calendar days between the timestamp and as_of
        if f.transform == "days_since":
            sql = f"date_diff('day', CAST({sql} AS DATE), CAST(r.as_of AS DATE))"
        if f.encoder is not None:
            sql = _encode_sql(sql, f.encoder)
        elif f.fill is not None and f.kind in ("window", "asof_attr", "static_attr"):
            fill_sql = "NULL" if f.fill is None else _lit(f.fill)
            sql = f"COALESCE({sql}, {fill_sql})"
        elif f.kind == "window" and f.fill is None:
            sql = sql  # NULL is a legitimate value (no history)
        return _cast_sql(sql, f.type)

    # ------------------------------------------------------------------
    def compile_all(self) -> dict[str, CompiledView]:
        return {name: self.compile_view(v) for name, v in self.reg.views.items()}

    def training_sql(self, view_name: str, label_table: str | None = None) -> str:
        """Point-in-time training sample: one row per application, as_of = apply_time."""
        cv = self.compile_view(self.reg.views[view_name])
        labels = label_table or self.reg.label_table
        return f"""
        SELECT
            f.{self.reg.base_id},
            l.label,
            l.label_ready_time,
            {', '.join(f'f."{n}"' for n in cv.feature_order)}
        FROM ({cv.sql}) f
        LEFT JOIN {labels} l ON l.{self.reg.base_id} = f.{self.reg.base_id}
        """


def build_spine_sql(reg: spec.Registry, keys: list[tuple[str, Any]], base_id: str | None = None) -> str:
    """Build a base spine from explicit (entity_id, as_of) pairs.

    Used by the consistency monitor and the leak audit to ask the *batch* engine
    for a feature vector at an arbitrary instant, not just at application time.
    """
    if not keys:
        raise ValueError("no keys")
    rows = []
    for i, (entity, as_of) in enumerate(keys):
        ent = str(entity).replace("'", "''")
        ts = expr.Literal(as_of).to_sql(expr.SqlDialect())
        if not (ts.startswith("'") or ts.startswith("TIMESTAMP")):
            raise ValueError(f"as_of must be a timestamp literal, got {as_of!r}")
        rows.append(f"({i}, '{ent}', CAST({ts} AS TIMESTAMP))")
    values = ", ".join(rows)
    bid = base_id or reg.base_id
    return f"""
    SELECT CAST(seq AS VARCHAR) AS entity_row_id,
           entity_id,
           as_of,
           CAST(seq AS VARCHAR) AS "{bid}"
    FROM (VALUES {values}) AS v(seq, entity_id, as_of)
    ORDER BY seq
    """


def build_base_cte(reg: spec.Registry) -> str:
    return f"""
    SELECT "{reg.base_id}" AS entity_row_id,
           "{reg.base_entity}" AS entity_id,
           "{reg.base_time}" AS as_of,
           "{reg.base_id}" AS "{reg.base_id}"
    FROM {reg.base_table}
    """
