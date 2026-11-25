"""Secure mathematical calculator.

Expressions are parsed with ``ast`` and validated against a strict whitelist of
node types, operators and functions BEFORE evaluation. ``eval``/``exec`` are
never used, so attribute access, imports, comprehensions, lambdas, strings and
arbitrary names are all rejected.
"""
from __future__ import annotations

import ast
import math
import operator
import re
import statistics

from pydantic import BaseModel, Field

from core.exceptions import CalculatorError, ToolInputError
from core.models import ToolResult
from core.validators import validate_text

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 100
MAX_FACTORIAL_ARG = 1000
MAX_RESULT_BITS = 100_000

# '^' passes this coarse filter only so the AST check can reject it with a helpful hint.
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/%^().,\s\[\]]+$")


class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        min_length=1,
        max_length=MAX_EXPRESSION_LENGTH,
        description=(
            "Math expression, e.g. '1450000000 - 123000000', 'sqrt(144) + 2**3', "
            "'sum([12.5, 40, 7.25])' or 'mean([88, 92, 79])'. Use ** for powers, "
            "no thousands separators."
        ),
    )


# ---------------------------------------------------------------- functions
def _as_int(value: float, fn: str) -> int:
    if isinstance(value, bool):
        raise CalculatorError(f"{fn}() needs a number.")
    if isinstance(value, float):
        if not value.is_integer():
            raise CalculatorError(f"{fn}() requires a whole number.")
        value = int(value)
    return value


def _factorial(x):
    n = _as_int(x, "factorial")
    if n < 0:
        raise CalculatorError("factorial() is not defined for negative numbers.")
    if n > MAX_FACTORIAL_ARG:
        raise CalculatorError(f"factorial() argument too large (max {MAX_FACTORIAL_ARG}).")
    return math.factorial(n)


def _log(x, base=None):
    if x <= 0:
        raise CalculatorError("log() is only defined for positive numbers.")
    return math.log(x) if base is None else math.log(x, base)


def _sqrt(x):
    if x < 0:
        raise CalculatorError("sqrt() of a negative number is not supported (real numbers only).")
    return math.sqrt(x)


def _flatten(args):
    if len(args) == 1 and isinstance(args[0], list):
        return args[0]
    return list(args)


def _sum(*args):
    return sum(_flatten(args))


def _mean(*args):
    values = _flatten(args)
    if not values:
        raise CalculatorError("mean() needs at least one value.")
    return statistics.fmean(values)


def _min(*args):
    values = _flatten(args)
    if not values:
        raise CalculatorError("min() needs at least one value.")
    return min(values)


def _max(*args):
    values = _flatten(args)
    if not values:
        raise CalculatorError("max() needs at least one value.")
    return max(values)


_FUNCTIONS = {
    "sqrt": _sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": _log, "log10": lambda x: _log(x, 10), "log2": lambda x: _log(x, 2),
    "exp": math.exp, "factorial": _factorial,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "radians": math.radians, "degrees": math.degrees,
    "sum": _sum, "mean": _mean, "min": _min, "max": _max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


# --------------------------------------------------------------- validation
def _validate_node(node: ast.AST) -> None:
    """Recursively reject anything that is not plainly arithmetic."""
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPS:
            hint = " Use ** for powers." if isinstance(node.op, ast.BitXor) else ""
            raise CalculatorError(f"Operator '{type(node.op).__name__}' is not allowed.{hint}")
        _validate_node(node.left)
        _validate_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise CalculatorError("Only unary + and - are allowed.")
        _validate_node(node.operand)
    elif isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise CalculatorError("Only plain numbers are allowed (no strings, booleans or complex).")
    elif isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise CalculatorError(f"Unknown name '{node.id}'. Allowed constants: pi, e, tau.")
    elif isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            _validate_node(item)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise CalculatorError("Only these functions are allowed: " + ", ".join(sorted(_FUNCTIONS)) + ".")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not allowed.")
        for arg in node.args:
            _validate_node(arg)
    else:
        raise CalculatorError(f"Unsupported syntax: {type(node).__name__}.")


def validate_expression(expression: object) -> ast.Expression:
    """Validate and parse an expression. Raises CalculatorError if unsafe/invalid."""
    try:
        text = validate_text(expression, name="expression", max_len=MAX_EXPRESSION_LENGTH)
    except ToolInputError as exc:
        raise CalculatorError(str(exc)) from exc
    if not _ALLOWED_CHARS.match(text):
        raise CalculatorError("Expression contains characters that are not allowed.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid syntax in expression: {exc.msg}.") from exc
    except (RecursionError, MemoryError) as exc:
        raise CalculatorError("Expression is too complex.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalculatorError("Expression is too complex.")
    _validate_node(tree)
    return tree


# ---------------------------------------------------------------- evaluation
def _check_size(value):
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > MAX_RESULT_BITS:
        raise CalculatorError("Result is too large.")
    return value


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _CONSTANTS[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if right * max(left.bit_length(), 1) > MAX_RESULT_BITS:
                    raise CalculatorError("Exponent too large; result would be enormous.")
        return _check_size(_BINARY_OPS[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        args = [_eval(a) for a in node.args]
        return _check_size(_FUNCTIONS[node.func.id](*args))
    raise CalculatorError("Unsupported syntax.")  # unreachable after validation


def evaluate(expression: str) -> int | float:
    """Safely evaluate ``expression`` and return a number."""
    tree = validate_expression(expression)
    try:
        value = _eval(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except OverflowError as exc:
        raise CalculatorError("Result is too large to represent.") from exc
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise CalculatorError(f"Math error: {exc}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Expression must evaluate to a single number.")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise CalculatorError("Result is not a finite number.")
    return value


def format_number(value: int | float) -> tuple[str, str]:
    """Return (human readable, raw) strings for a numeric result."""
    if isinstance(value, int):
        return f"{value:,}", str(value)
    if value.is_integer() and abs(value) < 1e15:
        return f"{int(value):,}", str(int(value))
    return f"{value:,.10g}", f"{value:.12g}"


def calculate(expression: str) -> ToolResult:
    """Tool entry point: never raises, always returns a ToolResult."""
    try:
        value = evaluate(expression)
    except CalculatorError as exc:
        return ToolResult.failure("calculator", str(exc))
    pretty, raw = format_number(value)
    expr = expression.strip()
    shown = expr if len(expr) <= 60 else expr[:57] + "..."   # keep the console trace compact
    return ToolResult.success(
        "calculator",
        content=f"{expr} = {pretty} (raw value: {raw})",
        summary=f"{shown} = {pretty}",
    )
