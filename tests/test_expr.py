"""Expression engine: parser, SQL emission, NumPy evaluation, SQL parity."""
from __future__ import annotations

import random

import duckdb
import numpy as np
import pandas as pd
import pytest

from feature_platform import expr


def ev(source, columns, n=1, ctx=None):
    node = expr.parse(source)
    return list(node.evaluate(expr.Frame(columns, n, ctx)))


# --------------------------------------------------------------------------
# Parsing and SQL emission
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    "a > 1",
    "a > 1 and b < 2",
    "a >= 1 or (b = 'x' and not c)",
    "(a + b) * 2 / 3",
    "a % 2 = 0",
    "a between 1 and 10",
    "a not between 1 and 10",
    "channel in ('a', 'b')",
    "channel not in ('a', 'b')",
    "x is null",
    "x is not null",
    "case when a > 1 then 1 when a > 0 then 0 else -1 end",
    "coalesce(a, 0) + abs(b)",
    "round(a, 2) > 1.5",
    "date_diff('day', a, b) > 7",
    "$as_of is not null",
    "-a + +b",
    "!a and b",
])
def test_parser_accepts(source):
    assert expr.parse(source) is not None


@pytest.mark.parametrize("source", [
    "a >",
    "a and",
    "case when a then 1",
    "foo(1,)",
    "(a",
    "a + + ",
    "",
    "   ",
])
def test_parser_rejects(source):
    with pytest.raises(expr.ExpressionError):
        expr.parse(source)


def test_sql_division_guards_zero():
    sql = expr.parse("a / b").to_sql(expr.SqlDialect())
    assert "NULLIF" in sql


def test_sql_quotes_strings():
    sql = expr.parse("name = 'o''brien'").to_sql(expr.SqlDialect())
    assert "o''brien" in sql


def test_parser_rejects_trailing_tokens():
    with pytest.raises(expr.ExpressionError):
        expr.parse("a b")


# --------------------------------------------------------------------------
# Semantics: SQL three-valued logic
# --------------------------------------------------------------------------

def test_null_propagates_through_comparison():
    out = ev("a > 1", {"a": np.array([2, None, 0], dtype=object)}, 3)
    assert out == [True, None, False]


def test_null_propagates_through_arithmetic():
    out = ev("a + 1", {"a": np.array([1, None], dtype=object)}, 2)
    assert out[0] == 2 and out[1] is None


def test_division_by_zero_is_null():
    out = ev("a / b", {"a": np.array([1, 1], dtype=object), "b": np.array([2, 0], dtype=object)}, 2)
    assert out[0] == 0.5 and out[1] is None


def test_kleene_and():
    a = np.array([True, True, False, False, None, None, None], dtype=object)
    b = np.array([True, None, True, None, True, False, None], dtype=object)
    out = ev("a and b", {"a": a, "b": b}, 7)
    assert out == [True, None, False, False, None, False, None]


def test_kleene_or():
    a = np.array([True, True, False, False, None, None], dtype=object)
    b = np.array([True, None, True, None, True, None], dtype=object)
    out = ev("a or b", {"a": a, "b": b}, 6)
    assert out == [True, True, True, None, True, None]


def test_is_null_handles_nan_and_nat():
    a = np.array([1.0, float("nan"), None, pd.NaT], dtype=object)
    out = ev("a is null", {"a": a}, 4)
    assert out == [False, True, True, True]


def test_case_null_condition_falls_through():
    out = ev("case when a > 1 then 'big' else 'small' end",
             {"a": np.array([2, None, 0], dtype=object)}, 3)
    assert out == ["big", "small", "small"]


def test_between_inclusive():
    out = ev("a between 1 and 3", {"a": np.array([0, 1, 3, 4], dtype=object)}, 4)
    assert out == [False, True, True, False]


def test_in_list_with_null_option():
    out = ev("a in (1, null)", {"a": np.array([1, 2, None], dtype=object)}, 3)
    assert out == [True, None, None]


def test_coalesce():
    out = ev("coalesce(a, b, 0)",
             {"a": np.array([None, 1], dtype=object), "b": np.array([5, None], dtype=object)}, 2)
    assert out == [5, 1]


# --------------------------------------------------------------------------
# Cross-engine parity: the same expression evaluated by DuckDB and by NumPy
# --------------------------------------------------------------------------

PARITY_EXPRESSIONS = [
    "a + b",
    "a - b",
    "a * b",
    "a / b",
    "a > b",
    "a >= b",
    "a = b",
    "a <> b",
    "a and b_flag",
    "a or b_flag",
    "not b_flag",
    "b_flag and not b_flag",
    "coalesce(a, 0) > 1",
    "case when a > b then 1 else 0 end",
    "a between b and b + 5",
    "a is null",
    "a is not null",
    "abs(a - b)",
    "greatest(a, b)",
    "least(a, b)",
    "nullif(a, b)",
    "a % 3",
]


def _random_frame(rng, n=200):
    def maybe_null(v, p=0.15):
        return None if rng.random() < p else v

    a = [maybe_null(rng.randint(-5, 5) * 1.0) for _ in range(n)]
    b = [maybe_null(rng.randint(-5, 5) * 1.0) for _ in range(n)]
    flag = [maybe_null(rng.random() < 0.5, 0.1) for _ in range(n)]
    return {"a": np.array(a, dtype=object), "b": np.array(b, dtype=object),
            "b_flag": np.array(flag, dtype=object)}


@pytest.mark.parametrize("source", PARITY_EXPRESSIONS)
def test_sql_python_parity(source):
    rng = random.Random(1234)
    cols = _random_frame(rng)
    n = len(cols["a"])

    con = duckdb.connect()
    con.execute("CREATE TABLE t (a DOUBLE, b DOUBLE, b_flag BOOLEAN)")
    df = pd.DataFrame({"a": [None if v is None else float(v) for v in cols["a"]],
                       "b": [None if v is None else float(v) for v in cols["b"]],
                       "b_flag": [None if v is None else bool(v) for v in cols["b_flag"]]})
    con.register("tmp", df)
    con.execute("INSERT INTO t SELECT * FROM tmp")
    node = expr.parse(source)
    sql_rows = con.execute(f"SELECT {node.to_sql(expr.SqlDialect())} AS v, rowid FROM t ORDER BY rowid").fetchall()
    con.close()

    py_vals = list(node.evaluate(expr.Frame(cols, n)))
    for (sql_val, rowid), py_val in zip(sql_rows, py_vals):
        assert _same(sql_val, py_val), f"{source} row {rowid}: sql={sql_val!r} py={py_val!r}"


def _same(a, b, tol=1e-12):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return str(a) == str(b)
