"""A small, null-safe expression language with dual compilation targets.

One expression (parsed from the YAML feature specs) is compiled into

* ``to_sql()``   -> DuckDB SQL, used by the offline / batch engine
* ``evaluate()`` -> NumPy-backed vectorised evaluation, used by the online
                    engine (with a one-row frame) and by unit tests

Both targets implement SQL three-valued logic (NULL propagates through
arithmetic and comparisons; ``AND``/``OR`` are Kleene logic), so the same
definition produces the same value in both worlds.  The parity test-suite
feeds random expressions through both and asserts exact agreement.

Grammar
-------
::

    expr    := or_expr
    or_expr := and_expr (("or" | "||") and_expr)*
    and_expr:= not_expr (("and" | "&&") not_expr)*
    not_expr:= ("not" | "!") not_expr | cmp
    cmp     := add (("=" | "==" | "!=" | "<>" | "<" | "<=" | ">" | ">=") add
                    | "is" ["not"] "null"
                    | ["not"] "in" "(" expr ("," expr)* ")"
                    | ["not"] "between" add "and" add)*
    add     := mul (("+" | "-") mul)*
    mul     := unary (("*" | "/" | "%") unary)*
    unary   := ("-" | "+") unary | primary
    primary := NUMBER | STRING | TRUE | FALSE | NULL
             | IDENT ("(" args ")")?        # column ref or function call
             | "$" IDENT                   # context variable: $as_of, $feature
             | "(" expr ")"
             | "case" "when" expr "then" expr ... ["else" expr] "end"
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ExpressionError(ValueError):
    pass


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>\d+\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
  | (?P<string>'(?:[^']|'')*'|"(?:[^"]|"")*")
  | (?P<op>==|!=|<>|<=|>=|&&|\|\||[-+*/%<>=!(),$])
  | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {
    "and", "or", "not", "is", "null", "in", "between", "case", "when",
    "then", "else", "end", "true", "false",
}


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ExpressionError(f"unexpected character {source[pos]!r} at {pos} in {source!r}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind == "ws":
            continue
        tokens.append(Token(kind, text, m.start()))
    tokens.append(Token("eof", "", len(source)))
    return tokens


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


class Node:
    def to_sql(self, dialect: "SqlDialect") -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def evaluate(self, frame: "Frame") -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def columns(self) -> set[str]:
        return set()

    def context_vars(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Literal(Node):
    value: Any

    def to_sql(self, d):
        if self.value is None:
            return "NULL"
        if isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        if isinstance(self.value, (int, float)):
            if isinstance(self.value, float) and (math.isnan(self.value) or math.isinf(self.value)):
                raise ExpressionError("non-finite literal")
            return repr(self.value)
        s = str(self.value).replace("'", "''")
        return f"'{s}'"

    def evaluate(self, frame):
        return frame.constant(self.value)


@dataclass(frozen=True)
class Column(Node):
    name: str

    def to_sql(self, d):
        return d.column(self.name)

    def evaluate(self, frame):
        return frame.column(self.name)

    def columns(self):
        return {self.name}


@dataclass(frozen=True)
class ContextVar(Node):
    name: str

    def to_sql(self, d):
        return d.context_var(self.name)

    def evaluate(self, frame):
        return frame.context(self.name)

    def context_vars(self):
        return {self.name}


@dataclass(frozen=True)
class UnaryOp(Node):
    op: str
    operand: Node

    def to_sql(self, d):
        inner = self.operand.to_sql(d)
        if self.op == "not":
            return f"(NOT {inner})"
        if self.op == "-":
            return f"(-{inner})"
        return f"(+{inner})"

    def evaluate(self, frame):
        v = self.operand.evaluate(frame)
        if self.op == "not":
            return _sql_not(v)
        if self.op == "-":
            return _num_neg(v)
        return v

    def columns(self):
        return self.operand.columns()

    def context_vars(self):
        return self.operand.context_vars()


_ARITH_SQL = {"+": "+", "-": "-", "*": "*", "/": "/", "%": "%"}
_CMP_SQL = {"=": "=", "==": "=", "!=": "<>", "<>": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
_LOGIC_SQL = {"and": "AND", "or": "OR"}


@dataclass(frozen=True)
class BinaryOp(Node):
    op: str
    left: Node
    right: Node

    def to_sql(self, d):
        if self.op in _LOGIC_SQL:
            return f"({self.left.to_sql(d)} {_LOGIC_SQL[self.op]} {self.right.to_sql(d)})"
        if self.op in _ARITH_SQL:
            # DuckDB: division by zero yields NULL for integer types and
            # +/-inf for floats; we normalise both engines via NULLIF.
            if self.op == "/":
                return f"({self.left.to_sql(d)} / NULLIF({self.right.to_sql(d)}, 0))"
            if self.op == "%":
                return f"({self.left.to_sql(d)} % NULLIF({self.right.to_sql(d)}, 0))"
            return f"({self.left.to_sql(d)} {_ARITH_SQL[self.op]} {self.right.to_sql(d)})"
        if self.op in _CMP_SQL:
            return f"({self.left.to_sql(d)} {_CMP_SQL[self.op]} {self.right.to_sql(d)})"
        raise ExpressionError(f"bad binary op {self.op!r}")

    def evaluate(self, frame):
        left = self.left.evaluate(frame)
        right = self.right.evaluate(frame)
        if self.op == "and":
            return _sql_and(_as_bool(left), _as_bool(right))
        if self.op == "or":
            return _sql_or(_as_bool(left), _as_bool(right))
        if self.op in _CMP_SQL:
            return _compare(self.op, left, right)
        return _arithmetic(self.op, left, right)

    def columns(self):
        return self.left.columns() | self.right.columns()

    def context_vars(self):
        return self.left.context_vars() | self.right.context_vars()


@dataclass(frozen=True)
class InList(Node):
    value: Node
    options: tuple[Node, ...]
    negated: bool = False

    def to_sql(self, d):
        opts = ", ".join(o.to_sql(d) for o in self.options)
        sql = f"({self.value.to_sql(d)} IN ({opts}))"
        return f"(NOT {sql})" if self.negated else sql

    def evaluate(self, frame):
        val = np.asarray(self.value.evaluate(frame), dtype=object)
        n = len(val)
        val_null = _is_null(val)
        any_null_option = False
        hit = np.zeros(n, dtype=bool)
        for o in self.options:
            opt = np.asarray(o.evaluate(frame), dtype=object)
            if len(opt) == 1:
                opt = np.broadcast_to(opt, (n,))
            any_null_option = any_null_option or bool(_is_null(opt).any())
            eq = _compare("==", val, opt)
            hit |= np.frompyfunc(lambda x: x is True or x == True, 1, 1)(eq).astype(bool)  # noqa: E712
        out = hit.astype(object)
        out[val_null] = None
        if any_null_option:
            out[~hit & ~val_null] = None
        return _sql_not(out) if self.negated else out

    def columns(self):
        s = self.value.columns()
        for o in self.options:
            s |= o.columns()
        return s

    def context_vars(self):
        s = self.value.context_vars()
        for o in self.options:
            s |= o.context_vars()
        return s


@dataclass(frozen=True)
class IsNull(Node):
    operand: Node
    negated: bool = False

    def to_sql(self, d):
        sql = f"({self.operand.to_sql(d)} IS NULL)"
        return f"(NOT {sql})" if self.negated else sql

    def evaluate(self, frame):
        v = self.operand.evaluate(frame)
        isnull = _is_null(v)
        return (~isnull) if self.negated else isnull

    def columns(self):
        return self.operand.columns()

    def context_vars(self):
        return self.operand.context_vars()


@dataclass(frozen=True)
class Between(Node):
    value: Node
    low: Node
    high: Node
    negated: bool = False

    def to_sql(self, d):
        sql = f"({self.value.to_sql(d)} BETWEEN {self.low.to_sql(d)} AND {self.high.to_sql(d)})"
        return f"(NOT {sql})" if self.negated else sql

    def evaluate(self, frame):
        v = self.value.evaluate(frame)
        lo = self.low.evaluate(frame)
        hi = self.high.evaluate(frame)
        ge = _as_bool(_compare(">=", v, lo))
        le = _as_bool(_compare("<=", v, hi))
        res = _sql_and(ge, le)
        return _sql_not(res) if self.negated else res

    def columns(self):
        return self.value.columns() | self.low.columns() | self.high.columns()

    def context_vars(self):
        return self.value.context_vars() | self.low.context_vars() | self.high.context_vars()


@dataclass(frozen=True)
class Case(Node):
    branches: tuple[tuple[Node, Node], ...]
    otherwise: Node

    def to_sql(self, d):
        parts = ["CASE"]
        for cond, val in self.branches:
            parts.append(f"WHEN {cond.to_sql(d)} THEN {val.to_sql(d)}")
        parts.append(f"ELSE {self.otherwise.to_sql(d)} END")
        return "(" + " ".join(parts) + ")"

    def evaluate(self, frame):
        n = frame.n
        out = self.otherwise.evaluate(frame).astype(object)
        # evaluate in reverse so earlier branches win
        assigned = np.zeros(n, dtype=bool)
        for cond, val in reversed(self.branches):
            mask = _as_bool(cond.evaluate(frame))
            take = (mask is True) if np.isscalar(mask) else np.asarray(mask, dtype=bool)
            v = val.evaluate(frame)
            if np.isscalar(take):
                if take:
                    out = v.astype(object)
                    assigned[:] = True
            else:
                take = take & ~assigned
                out[take] = np.asarray(v, dtype=object)[take]
                assigned |= take
        return out

    def columns(self):
        s = self.otherwise.columns()
        for c, v in self.branches:
            s |= c.columns() | v.columns()
        return s

    def context_vars(self):
        s = self.otherwise.context_vars()
        for c, v in self.branches:
            s |= c.context_vars() | v.context_vars()
        return s


@dataclass(frozen=True)
class FuncCall(Node):
    name: str
    args: tuple[Node, ...]

    def to_sql(self, d):
        fn = d.function(self.name)
        return f"{fn}({', '.join(a.to_sql(d) for a in self.args)})"

    def evaluate(self, frame):
        impl = _PY_FUNCS.get(self.name)
        if impl is None:
            raise ExpressionError(f"unknown function {self.name!r}")
        return impl(frame, *[a.evaluate(frame) for a in self.args])

    def columns(self):
        s: set[str] = set()
        for a in self.args:
            s |= a.columns()
        return s

    def context_vars(self):
        s: set[str] = set()
        for a in self.args:
            s |= a.context_vars()
        return s


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = tokenize(source)
        self.i = 0

    # -- helpers ---------------------------------------------------------
    def peek(self) -> Token:
        return self.tokens[self.i]

    def next(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def accept(self, text: str) -> bool:
        t = self.peek()
        if t.text.lower() == text and t.kind in ("op", "ident"):
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> Token:
        t = self.peek()
        if t.text.lower() != text:
            raise ExpressionError(f"expected {text!r} at {t.pos} in {self.source!r}, got {t.text!r}")
        self.i += 1
        return t

    # -- grammar ---------------------------------------------------------
    def parse(self) -> Node:
        node = self.parse_or()
        if self.peek().kind != "eof":
            raise ExpressionError(f"trailing token {self.peek().text!r} in {self.source!r}")
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while True:
            t = self.peek()
            if t.kind == "ident" and t.text.lower() == "or" or t.kind == "op" and t.text == "||":
                self.next()
                node = BinaryOp("or", node, self.parse_and())
            else:
                return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while True:
            t = self.peek()
            if t.kind == "ident" and t.text.lower() == "and" or t.kind == "op" and t.text == "&&":
                self.next()
                node = BinaryOp("and", node, self.parse_not())
            else:
                return node

    def parse_not(self) -> Node:
        t = self.peek()
        if (t.kind == "ident" and t.text.lower() == "not") or (t.kind == "op" and t.text == "!"):
            self.next()
            return UnaryOp("not", self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self) -> Node:
        node = self.parse_add()
        while True:
            t = self.peek()
            negated = False
            if t.kind == "ident" and t.text.lower() == "not":
                nxt = self.tokens[self.i + 1]
                if nxt.text.lower() in ("in", "between"):
                    self.next()
                    negated = True
                    t = self.peek()
            if t.kind == "op" and t.text in _CMP_SQL:
                self.next()
                node = BinaryOp(t.text, node, self.parse_add())
            elif t.kind == "ident" and t.text.lower() == "is":
                self.next()
                neg = self.accept("not")
                self.expect("null")
                node = IsNull(node, negated=neg)
            elif t.kind == "ident" and t.text.lower() == "in":
                self.next()
                self.expect("(")
                opts = [self.parse_or()]
                while self.accept(","):
                    opts.append(self.parse_or())
                self.expect(")")
                node = InList(node, tuple(opts), negated=negated)
            elif t.kind == "ident" and t.text.lower() == "between":
                self.next()
                low = self.parse_add()
                self.expect("and")
                high = self.parse_add()
                node = Between(node, low, high, negated=negated)
            else:
                return node

    def parse_add(self) -> Node:
        node = self.parse_mul()
        while True:
            t = self.peek()
            if t.kind == "op" and t.text in ("+", "-"):
                self.next()
                node = BinaryOp(t.text, node, self.parse_mul())
            else:
                return node

    def parse_mul(self) -> Node:
        node = self.parse_unary()
        while True:
            t = self.peek()
            if t.kind == "op" and t.text in ("*", "/", "%"):
                self.next()
                node = BinaryOp(t.text, node, self.parse_unary())
            else:
                return node

    def parse_unary(self) -> Node:
        t = self.peek()
        if t.kind == "op" and t.text in ("-", "+"):
            self.next()
            inner = self.parse_unary()
            return UnaryOp("-", inner) if t.text == "-" else inner
        return self.parse_primary()

    def parse_primary(self) -> Node:
        t = self.next()
        if t.kind == "number":
            return Literal(float(t.text) if ("." in t.text or "e" in t.text.lower()) else int(t.text))
        if t.kind == "string":
            q = t.text[0]
            body = t.text[1:-1]
            if q == "'":
                body = body.replace("''", "'")
            else:
                body = body.replace('""', '"')
            return Literal(body)
        if t.kind == "op" and t.text == "$":
            ident = self.next()
            if ident.kind != "ident":
                raise ExpressionError(f"expected identifier after $ at {ident.pos}")
            return ContextVar(ident.text)
        if t.kind == "op" and t.text == "(":
            node = self.parse_or()
            self.expect(")")
            return node
        if t.kind == "ident":
            low = t.text.lower()
            if low == "true":
                return Literal(True)
            if low == "false":
                return Literal(False)
            if low == "null":
                return Literal(None)
            if low == "case":
                return self.parse_case()
            if self.peek().text == "(":
                self.next()
                args: list[Node] = []
                if self.peek().text != ")":
                    args.append(self.parse_or())
                    while self.accept(","):
                        args.append(self.parse_or())
                self.expect(")")
                return FuncCall(low, tuple(args))
            return Column(t.text)
        raise ExpressionError(f"unexpected token {t.text!r} at {t.pos} in {self.source!r}")

    def parse_case(self) -> Node:
        branches: list[tuple[Node, Node]] = []
        while self.accept("when"):
            cond = self.parse_or()
            self.expect("then")
            val = self.parse_or()
            branches.append((cond, val))
        if not branches:
            raise ExpressionError("CASE without WHEN")
        other: Node = Literal(None)
        if self.accept("else"):
            other = self.parse_or()
        self.expect("end")
        return Case(tuple(branches), other)


def parse(source: str) -> Node:
    if not isinstance(source, str) or not source.strip():
        raise ExpressionError("expression must be a non-empty string")
    return Parser(source).parse()


# --------------------------------------------------------------------------
# SQL dialect hooks
# --------------------------------------------------------------------------


class SqlDialect:
    """Maps logical names to physical SQL.  The default emits DuckDB."""

    def __init__(self, context_vars: dict[str, str] | None = None, function_map: dict[str, str] | None = None,
                 column_prefix: str = "", column_map: dict[str, str] | None = None):
        self._ctx = context_vars or {}
        self._fn = function_map or {}
        self._prefix = column_prefix
        self._colmap = column_map or {}

    def is_qualified(self, name: str) -> bool:
        return "." in name

    def column(self, name: str) -> str:
        if name in self._colmap:
            return self._colmap[name]
        if "." in name:
            return name
        if self._prefix:
            return f'{self._prefix}"{name}"'
        return f'"{name}"'

    def context_var(self, name: str) -> str:
        try:
            return self._ctx[name]
        except KeyError:
            raise ExpressionError(f"unbound context variable ${name}") from None

    def function(self, name: str) -> str:
        fn = self._fn.get(name, name)
        if fn.upper() in _FORBIDDEN_SQL_FUNCS:
            raise ExpressionError(f"function {name!r} is not allowed in feature definitions")
        return fn


_FORBIDDEN_SQL_FUNCS = {"read_csv", "read_parquet", "query", "getvariable", "system"}


# --------------------------------------------------------------------------
# Python (NumPy) evaluation
# --------------------------------------------------------------------------


class Frame:
    """Column store used by the online engine and by parity tests.

    ``columns`` maps name -> numpy object array (or pandas Series).  Values
    use ``None``/``np.nan``/``pd.NaT`` for SQL NULL.
    """

    def __init__(self, columns: dict[str, Any], n: int, ctx: dict[str, Any] | None = None):
        self.columns = columns
        self.n = n
        self.ctx = ctx or {}

    def column(self, name: str) -> np.ndarray:
        try:
            return _to_array(self.columns[name], self.n)
        except KeyError:
            raise ExpressionError(f"unknown column {name!r}") from None

    def context(self, name: str) -> np.ndarray:  # type: ignore[override]
        try:
            v = self.ctx[name]
        except KeyError:
            raise ExpressionError(f"unbound context variable ${name}") from None
        return _to_array(v, self.n)

    def constant(self, value: Any) -> np.ndarray:
        arr = np.empty(self.n, dtype=object)
        arr[:] = value
        return arr

    def equal_mask(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return _compare("==", left, right)


def _to_array(v: Any, n: int) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v if v.dtype == object else v.astype(object)
    if hasattr(v, "to_numpy"):
        return v.to_numpy(dtype=object)
    arr = np.empty(n, dtype=object)
    arr[:] = v
    return arr


def _is_null(v) -> np.ndarray:
    arr = np.asarray(v, dtype=object)
    out = np.frompyfunc(_scalar_null, 1, 1)(arr)
    return out.astype(bool)


def _as_bool(v) -> np.ndarray:
    arr = np.asarray(v, dtype=object)
    out = np.empty(len(arr), dtype=object)
    for i, x in enumerate(arr):
        if _scalar_null(x):
            out[i] = None
        elif isinstance(x, (bool, np.bool_)):
            out[i] = bool(x)
        elif isinstance(x, (int, float)):
            out[i] = bool(x)
        else:
            raise ExpressionError(f"cannot use {type(x).__name__} as boolean")
    return out


def _sql_not(v) -> np.ndarray:
    out = np.empty(len(v), dtype=object)
    for i, x in enumerate(np.asarray(v, dtype=object)):
        out[i] = None if _scalar_null(x) else (not bool(x))
    return out


def _sql_and(a, b) -> np.ndarray:
    out = np.empty(len(a), dtype=object)
    for i, (x, y) in enumerate(zip(a, b)):
        xb = None if x is None else bool(x)
        yb = None if y is None else bool(y)
        if xb is False or yb is False:
            out[i] = False
        elif xb is None or yb is None:
            out[i] = None
        else:
            out[i] = True
    return out


def _sql_or(a, b) -> np.ndarray:
    out = np.empty(len(a), dtype=object)
    for i, (x, y) in enumerate(zip(a, b)):
        xb = None if x is None else bool(x)
        yb = None if y is None else bool(y)
        if xb is True or yb is True:
            out[i] = True
        elif xb is None or yb is None:
            out[i] = None
        else:
            out[i] = False
    return out


def _num_neg(v) -> np.ndarray:
    out = np.empty(len(v), dtype=object)
    for i, x in enumerate(np.asarray(v, dtype=object)):
        out[i] = None if _scalar_null(x) else -x
    return out


def _scalar_null(x) -> bool:
    if x is None or x is pd.NaT:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, np.datetime64):
        return bool(np.isnat(x))
    return False


def _compare(op: str, left, right) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    n = max(len(left), len(right))
    left = np.broadcast_to(left, (n,))
    right = np.broadcast_to(right, (n,))
    out = np.empty(n, dtype=object)
    for i in range(n):
        a, b = left[i], right[i]
        if _scalar_null(a) or _scalar_null(b):
            out[i] = None
            continue
        # numeric comparison only when both sides are numbers; otherwise
        # compare as strings to mirror SQL's implicit casts in our schemas.
        if isinstance(a, bool) or isinstance(b, bool):
            a, b = bool(a), bool(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            pass
        elif type(a) is not type(b) and not (isinstance(a, str) and isinstance(b, str)):
            a, b = str(a), str(b)
        if op in ("=", "=="):
            out[i] = a == b
        elif op in ("!=", "<>"):
            out[i] = a != b
        elif op == "<":
            out[i] = a < b
        elif op == "<=":
            out[i] = a <= b
        elif op == ">":
            out[i] = a > b
        elif op == ">=":
            out[i] = a >= b
        else:
            raise ExpressionError(f"bad comparison {op!r}")
    return out


def _arithmetic(op: str, left, right) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    n = max(len(left), len(right))
    left = np.broadcast_to(left, (n,))
    right = np.broadcast_to(right, (n,))
    out = np.empty(n, dtype=object)
    for i in range(n):
        a, b = left[i], right[i]
        if _scalar_null(a) or _scalar_null(b):
            out[i] = None
            continue
        a = float(a) if isinstance(a, (int, float, bool)) else a
        b = float(b) if isinstance(b, (int, float, bool)) else b
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ExpressionError(f"arithmetic on non-numeric values in {op!r}")
        if op == "+":
            out[i] = a + b
        elif op == "-":
            out[i] = a - b
        elif op == "*":
            out[i] = a * b
        elif op == "/":
            out[i] = None if b == 0 else a / b
        elif op == "%":
            out[i] = None if b == 0 else math.fmod(a, b)
        else:
            raise ExpressionError(f"bad arithmetic {op!r}")
    return out


# -- functions --------------------------------------------------------------

def _fn_coalesce(frame, *args):
    n = frame.n
    out = np.empty(n, dtype=object)
    stacks = [np.asarray(a, dtype=object) for a in args]
    for i in range(n):
        val = None
        for s in stacks:
            x = s[i]
            if not _scalar_null(x):
                val = x
                break
        out[i] = val
    return out


def _fn_nullif(frame, a, b):
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    n = max(len(a), len(b))
    a = np.broadcast_to(a, (n,))
    b = np.broadcast_to(b, (n,))
    out = np.empty(n, dtype=object)
    for i in range(n):
        # Standard SQL: NULLIF(a, b) = CASE WHEN a = b THEN NULL ELSE a END.
        # If b is NULL the comparison is UNKNOWN, so the ELSE branch applies
        # and a is returned (verified against DuckDB).
        if _scalar_null(a[i]):
            out[i] = None
        elif not _scalar_null(b[i]) and a[i] == b[i]:
            out[i] = None
        else:
            out[i] = a[i]
    return out


def _numeric_map(frame, fn, *args) -> np.ndarray:
    arrays = [np.asarray(a, dtype=object) for a in args]
    n = max(len(a) for a in arrays)
    arrays = [np.broadcast_to(a, (n,)) for a in arrays]
    out = np.empty(n, dtype=object)
    for i in range(n):
        vals = [a[i] for a in arrays]
        if any(_scalar_null(v) for v in vals):
            out[i] = None
        else:
            out[i] = fn(*[float(v) for v in vals])
    return out


def _fn_abs(frame, a):
    return _numeric_map(frame, abs, a)


def _fn_round(frame, a, digits=None):
    if digits is None:
        return _numeric_map(frame, lambda x: float(round(x)), a)
    return _numeric_map(frame, lambda x, d: float(round(x, int(d))), a, digits)


def _fn_floor(frame, a):
    return _numeric_map(frame, math.floor, a)


def _fn_ceil(frame, a):
    return _numeric_map(frame, math.ceil, a)


def _fn_greatest(frame, *args):
    arrays = [np.asarray(a, dtype=object) for a in args]
    n = max(len(a) for a in arrays)
    arrays = [np.broadcast_to(a, (n,)) for a in arrays]
    out = np.empty(n, dtype=object)
    for i in range(n):
        vals = [a[i] for a in arrays if not _scalar_null(a[i])]
        out[i] = max(vals) if vals else None
    return out


def _fn_least(frame, *args):
    arrays = [np.asarray(a, dtype=object) for a in args]
    n = max(len(a) for a in arrays)
    arrays = [np.broadcast_to(a, (n,)) for a in arrays]
    out = np.empty(n, dtype=object)
    for i in range(n):
        vals = [a[i] for a in arrays if not _scalar_null(a[i])]
        out[i] = min(vals) if vals else None
    return out


def _fn_lower(frame, a):
    arr = np.asarray(a, dtype=object)
    return np.frompyfunc(lambda x: None if _scalar_null(x) else str(x).lower(), 1, 1)(arr)


def _fn_upper(frame, a):
    arr = np.asarray(a, dtype=object)
    return np.frompyfunc(lambda x: None if _scalar_null(x) else str(x).upper(), 1, 1)(arr)


def _fn_is_null(frame, a):
    return np.frompyfunc(lambda x: _scalar_null(x), 1, 1)(np.asarray(a, dtype=object)).astype(bool)


def _fn_date_diff(frame, unit, start, end):
    units = {"day": 86400.0, "hour": 3600.0, "minute": 60.0, "second": 1.0}
    if not isinstance(unit, np.ndarray):
        raise ExpressionError("date_diff unit must be a literal")
    u = str(np.asarray(unit, dtype=object)[0]).lower()
    if u not in units:
        raise ExpressionError(f"unsupported date_diff unit {u!r}")
    secs = units[u]
    start = np.asarray(start, dtype=object)
    end = np.asarray(end, dtype=object)
    n = max(len(start), len(end))
    start = np.broadcast_to(start, (n,))
    end = np.broadcast_to(end, (n,))
    out = np.empty(n, dtype=object)
    for i in range(n):
        if _scalar_null(start[i]) or _scalar_null(end[i]):
            out[i] = None
        else:
            import pandas as pd

            delta = (pd.Timestamp(end[i]) - pd.Timestamp(start[i])).total_seconds()
            out[i] = int(delta // secs)
    return out


def _fn_strftime(frame, value, fmt):
    fmt_arr = np.asarray(fmt, dtype=object)
    f = str(fmt_arr[0]) if len(fmt_arr) else "%Y-%m-%d"
    import pandas as pd

    arr = np.asarray(value, dtype=object)
    return np.frompyfunc(lambda x: None if _scalar_null(x) else pd.Timestamp(x).strftime(f), 1, 1)(arr)


def _fn_concat(frame, *args):
    arrays = [np.asarray(a, dtype=object) for a in args]
    n = max(len(a) for a in arrays)
    arrays = [np.broadcast_to(a, (n,)) for a in arrays]
    out = np.empty(n, dtype=object)
    for i in range(n):
        vals = [a[i] for a in arrays]
        out[i] = None if any(_scalar_null(v) for v in vals) else "".join(str(v) for v in vals)
    return out


_PY_FUNCS: dict[str, Callable] = {
    "coalesce": _fn_coalesce,
    "nullif": _fn_nullif,
    "abs": _fn_abs,
    "round": _fn_round,
    "floor": _fn_floor,
    "ceil": _fn_ceil,
    "greatest": _fn_greatest,
    "least": _fn_least,
    "lower": _fn_lower,
    "upper": _fn_upper,
    "is_null": _fn_is_null,
    "date_diff": _fn_date_diff,
    "strftime": _fn_strftime,
    "concat": _fn_concat,
}


def evaluate(node: Node, columns: dict[str, Any], n: int, context: dict[str, Any] | None = None) -> np.ndarray:
    return node.evaluate(Frame(columns, n, context))


def referenced_columns(node: Node) -> set[str]:
    return node.columns()


def referenced_context(node: Node) -> set[str]:
    return node.context_vars()
