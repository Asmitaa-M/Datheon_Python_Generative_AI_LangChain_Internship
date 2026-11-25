import math

import pytest

from core.exceptions import CalculatorError
from tools.calculator import calculate, evaluate, format_number


@pytest.mark.parametrize("expr,expected", [
    ("2 + 3", 5),
    ("10 - 4", 6),
    ("6 * 7", 42),
    ("7 / 2", 3.5),
    ("2 ** 10", 1024),
    ("17 % 5", 2),
    ("17 // 5", 3),
    ("(2 + 3) * 4", 20),
    ("-5 + 2", -3),
    ("1450000000 - 123000000", 1327000000),
    ("2 * (3 + (4 - 1)) ** 2", 72),
    ("1.5e3 + 1", 1501),
])
def test_basic_arithmetic(expr, expected):
    assert evaluate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr,expected", [
    ("sqrt(144)", 12),
    ("sin(0)", 0),
    ("cos(0)", 1),
    ("tan(0)", 0),
    ("log(e)", 1),
    ("log(100, 10)", 2),
    ("log10(1000)", 3),
    ("log2(8)", 3),
    ("factorial(5)", 120),
    ("factorial(5.0)", 120),
    ("abs(-3)", 3),
    ("round(2.567, 2)", 2.57),
    ("floor(2.9) + ceil(2.1)", 5),
    ("pi * 2", 2 * math.pi),
    ("sin(radians(90))", 1),
    ("exp(0)", 1),
])
def test_math_functions(expr, expected):
    assert evaluate(expr) == pytest.approx(expected)


def test_aggregate_functions_over_lists():
    assert evaluate("sum([1, 2, 3.5])") == 6.5
    assert evaluate("mean([88, 92, 79])") == pytest.approx(86.3333333)
    assert evaluate("max([4, 9, 2])") == 9
    assert evaluate("min(4, 9, 2)") == 2


def test_division_by_zero():
    with pytest.raises(CalculatorError, match="Division by zero"):
        evaluate("1 / 0")
    with pytest.raises(CalculatorError, match="Division by zero"):
        evaluate("5 % 0")
    assert calculate("10 / (5 - 5)").ok is False


@pytest.mark.parametrize("expr", ["2 +", "(1 + 2", "1 2", "* 3", "", "   ", "()", "1 +* 2"])
def test_invalid_syntax(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


@pytest.mark.parametrize("expr", [
    "sqrt(-1)", "log(0)", "log(-5)", "factorial(-1)", "factorial(2.5)",
    "factorial(100000)", "mean([])", "9 ** 9 ** 9", "10 ** 100000", "2 ^ 3",
])
def test_math_errors_are_readable(expr):
    with pytest.raises(CalculatorError) as exc:
        evaluate(expr)
    assert str(exc.value)  # readable, non-empty message


def test_xor_hint_suggests_power_operator():
    with pytest.raises(CalculatorError, match=r"\*\*"):
        evaluate("2 ^ 3")


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hacked')",
    "__import__(1)",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "().__class__.__bases__",
    "(1).real",
    "pi.real",
    "[x for x in range(3)]",
    "(lambda: 1)()",
    "'a' * 3",
    "True + 1",
    "1j + 1",
    "x + 1",
    "print(1)",
    "sqrt(x=4)",
    "sqrt(*[4])",
    "1 if 1 else 2",
    "a.b",
    "{1: 2}",
    "1; 2",
    "not 1",
    "1 < 2",
])
def test_code_execution_attempts_rejected(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_expression_too_long():
    with pytest.raises(CalculatorError, match="too long"):
        evaluate("1+" * 150 + "1")


def test_expression_with_too_many_nodes_is_rejected():
    expr = "1+" * 70 + "1"            # 141 chars, but 140+ AST nodes
    assert len(expr) < 200
    with pytest.raises(CalculatorError, match="too complex"):
        evaluate(expr)


def test_redundant_parentheses_are_harmless():
    assert evaluate("(" * 40 + "7" + ")" * 40) == 7


def test_non_string_input():
    with pytest.raises(CalculatorError):
        evaluate(None)  # type: ignore[arg-type]
    with pytest.raises(CalculatorError):
        evaluate(123)  # type: ignore[arg-type]


def test_tool_returns_readable_structured_result():
    result = calculate("1450000000 - 123000000")
    assert result.ok and result.tool == "calculator"
    assert "1,327,000,000" in result.content
    assert "raw value: 1327000000" in result.content


def test_tool_failure_is_structured_not_raised():
    result = calculate("__import__('os')")
    assert result.ok is False and result.error


def test_float_noise_is_cleaned():
    assert format_number(0.1 + 0.2) == ("0.3", "0.3")
    assert format_number(4 / 2) == ("2", "2")
    assert format_number(1234567) == ("1,234,567", "1234567")
