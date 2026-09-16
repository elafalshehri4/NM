from __future__ import annotations

"""Restricted parsing for user-entered mathematical expressions.

SymPy's normal ``sympify``/``parse_expr`` helpers ultimately evaluate generated
Python code.  This module first validates the user's text with Python's AST and
then parses it with an empty built-in namespace and an explicit SymPy namespace.
Only ordinary arithmetic, approved names, and direct calls to approved
mathematical functions are accepted.
"""

import ast
from collections.abc import Mapping
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

_MAX_EXPRESSION_LENGTH = 1200
_MAX_AST_NODES = 300
_MAX_NESTING_DEPTH = 40
_MAX_INTEGER_DIGITS = 120
_MAX_LITERAL_EXPONENT = 1000

_ALLOWED_BINARY_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)

_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
}

_TRANSFORMATIONS = standard_transformations + (convert_xor,)


class SafeExpressionError(ValueError):
    """Raised when a mathematical expression contains unsafe syntax."""


def _validate_numeric_constant(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafeExpressionError("Only real numerical constants are allowed.")

    if isinstance(value, int) and len(str(abs(value))) > _MAX_INTEGER_DIGITS:
        raise SafeExpressionError("An integer literal in the expression is too large.")


def _validate_ast(
    text: str,
    local_dictionary: Mapping[str, Any],
) -> None:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise SafeExpressionError("The mathematical expression has invalid syntax.") from error

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise SafeExpressionError("The mathematical expression is too complex.")

    allowed_names = set(local_dictionary)
    callable_names = {
        name for name, value in local_dictionary.items() if callable(value)
    }

    def visit(node: ast.AST, depth: int = 0) -> None:
        if depth > _MAX_NESTING_DEPTH:
            raise SafeExpressionError("The mathematical expression is nested too deeply.")

        if isinstance(node, ast.Expression):
            visit(node.body, depth + 1)
            return

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
                raise SafeExpressionError("That arithmetic operator is not supported.")
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                exponent = node.right.value
                _validate_numeric_constant(exponent)
                if abs(float(exponent)) > _MAX_LITERAL_EXPONENT:
                    raise SafeExpressionError("The literal exponent is too large.")
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
                raise SafeExpressionError("That unary operator is not supported.")
            visit(node.operand, depth + 1)
            return

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SafeExpressionError(
                    "Only direct calls to supported mathematical functions are allowed."
                )
            if node.func.id not in callable_names:
                raise SafeExpressionError(
                    f"Unsupported function: {node.func.id}."
                )
            if node.keywords:
                raise SafeExpressionError("Keyword arguments are not supported.")
            if not 1 <= len(node.args) <= 2:
                raise SafeExpressionError(
                    "Supported mathematical functions accept one or two arguments."
                )
            for argument in node.args:
                visit(argument, depth + 1)
            return

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise SafeExpressionError(f"Unsupported name: {node.id}.")
            if node.id.startswith("_") or "__" in node.id:
                raise SafeExpressionError("Private Python names are not allowed.")
            return

        if isinstance(node, ast.Constant):
            _validate_numeric_constant(node.value)
            return

        # Attributes, subscripts, comparisons, Boolean operations, strings,
        # containers, lambdas, comprehensions, assignments, and statements are
        # deliberately rejected here.
        raise SafeExpressionError(
            f"Unsupported syntax: {type(node).__name__}."
        )

    visit(tree)


def safe_sympify(
    value: Any,
    *,
    locals: Mapping[str, Any] | None = None,
    evaluate: bool = True,
) -> sp.Basic:
    """Safely parse one user-entered mathematical expression.

    The function intentionally mirrors the small subset of ``sympy.sympify``
    used by this project, so solver pages can share one secure implementation.
    """

    if not isinstance(value, str):
        raise SafeExpressionError("The mathematical expression must be text.")

    text = value.strip()
    if not text:
        raise SafeExpressionError("The mathematical expression cannot be empty.")
    if len(text) > _MAX_EXPRESSION_LENGTH:
        raise SafeExpressionError("The mathematical expression is too long.")
    if "\x00" in text:
        raise SafeExpressionError("The mathematical expression contains invalid text.")

    local_dictionary = dict(locals or {})
    normalized_text = text.replace("^", "**")
    _validate_ast(normalized_text, local_dictionary)

    try:
        result = parse_expr(
            normalized_text,
            local_dict=local_dictionary,
            global_dict=dict(_SAFE_GLOBALS),
            transformations=_TRANSFORMATIONS,
            evaluate=evaluate,
        )
    except (SyntaxError, TypeError, ValueError, NameError, AttributeError) as error:
        raise SafeExpressionError(
            "The mathematical expression could not be parsed safely."
        ) from error

    if not isinstance(result, sp.Basic):
        raise SafeExpressionError("The input did not produce a SymPy expression.")
    return result
