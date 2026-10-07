from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Callable
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from matplotlib.figure import Figure
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from components.navigation import navbar
from utilities.ui import load_css
from utilities.safe_math import safe_sympify
CONFIG = {'method_id': 'newton', 'title': 'Newton-Raphson Solver', 'label': 'NEWTON-RAPHSON METHOD TOOL', 'description': 'Enter a differentiable function and one starting value to approximate a root using tangent-line updates.', 'lesson': 'Newton_Raphson_Method', 'quiz': 'Newton_Raphson_Quiz', 'footer': 'Newton-Raphson Solver • Root Finding', 'default_eq': 'x**3 - x - 2', 'conditions': ['The first and second derivatives are calculated automatically.', 'The derivative must remain nonzero during iteration.', 'A starting value near a simple root improves reliability.', 'The function and derivative must remain finite at every iterate.'], 'formula': "x_{n+1}=x_n-\\frac{f(x_n)}{f'(x_n)}"}
METHOD_NAME = 'Newton-Raphson Method'
DISPLAY_DECIMALS = 3
DEFAULT_X0 = 1.5
DEFAULT_TOLERANCE = 1e-08
DEFAULT_MAX_ITERATIONS = 100
MIN_ITERATIONS = 1
MAX_ITERATIONS = 1000
MIN_TOLERANCE = 1e-14
ZERO_TOLERANCE = 1e-15
VALUE_MAGNITUDE_WARNING = 1000000000000.0
DERIVATIVE_SENSITIVITY_WARNING = 1000000000000.0
MULTIPLE_ROOT_WARNING_THRESHOLD = 1.5
QUADRATIC_ORDER_WARNING_THRESHOLD = 1.45
REPORT_TIME_ZONE = 'Asia/Riyadh'
X_SYMBOL = sp.Symbol('x', real=True)
ALLOWED_FUNCTION_NAMES = {'x': X_SYMBOL, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan, 'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh, 'exp': sp.exp, 'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'Abs': sp.Abs, 'abs': sp.Abs, 'pi': sp.pi, 'E': sp.E}
_SUPERSCRIPT_TRANSLATION = str.maketrans('0123456789-+', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺')

def format_scientific_power(value: float | int | None, decimals: int=DISPLAY_DECIMALS, unavailable: str='—') -> str:
    """Format scientific notation as a coefficient multiplied by 10ⁿ."""
    if value is None:
        return unavailable
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0.0:
        return f'{0.0:.{decimals}f}'
    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / 10.0 ** exponent
    exponent_text = str(exponent).translate(_SUPERSCRIPT_TRANSLATION)
    return f'{mantissa:.{decimals}f} × 10{exponent_text}'

def format_display_number(value: float | int | None, decimals: int=DISPLAY_DECIMALS, unavailable: str='—') -> str:
    """Use fixed notation or scientific notation according to magnitude."""
    if value is None:
        return unavailable
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude < 10.0 ** (-decimals) or magnitude >= 1000000.0):
        return format_scientific_power(number, decimals, unavailable)
    return f'{number:.{decimals}f}'

def format_number(value: float | int | None, decimals: int=DISPLAY_DECIMALS) -> str:
    """Format a final numerical value for display."""
    return format_display_number(value, decimals, unavailable='Not available')

def round_numeric_dataframe(dataframe: pd.DataFrame, decimals: int=DISPLAY_DECIMALS) -> pd.DataFrame:
    """Round numeric columns only in a display copy."""
    rounded = dataframe.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
    return rounded

@dataclass(frozen=True)
class NewtonIteration:
    """One Newton-Raphson tangent-line update."""
    iteration: int
    x_n: float
    function_value: float
    derivative_value: float
    second_derivative_value: float
    correction: float
    x_next: float
    function_next: float
    absolute_step: float
    relative_step_percent: float
    residual: float
    derivative_sensitivity_indicator: float | None
    estimated_multiplicity: float | None
    exact_root_error: float | None
    exact_root_relative_error_percent: float | None
    observed_order: float | None
    quadratic_indicator: float | None
    operation: str
    status: str

@dataclass(frozen=True)
class NewtonResult:
    status: str
    success: bool
    converged: bool
    method: str
    message: str
    stopping_reason: str
    function_text: str
    function_expression: sp.Expr | None
    derivative_expression: sp.Expr | None
    second_derivative_expression: sp.Expr | None
    initial_x: float | None
    tolerance: float | None
    maximum_iterations: int
    known_exact_root: float | None
    exact_root_verified: bool | None
    symbolic_multiplicity_at_exact_root: int | None
    iterations: tuple[NewtonIteration, ...]
    approximate_root: float | None
    function_at_root: float | None
    derivative_at_root: float | None
    final_absolute_step: float | None
    final_relative_step_percent: float | None
    final_residual: float | None
    final_exact_error: float | None
    latest_observed_order: float | None
    latest_quadratic_indicator: float | None
    warnings: tuple[str, ...]
    input_signature: str
    execution_datetime: datetime

def current_report_datetime() -> datetime:
    """Return a timezone-aware report timestamp."""
    return datetime.now(ZoneInfo(REPORT_TIME_ZONE))

def validate_finite_real(raw_value: Any, value_name: str) -> float:
    """Convert one input to a finite real float."""
    try:
        complex_value = complex(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{value_name} must be a valid real number.') from error
    if abs(complex_value.imag) > ZERO_TOLERANCE:
        raise ValueError(f'{value_name} must be real, not complex.')
    value = float(complex_value.real)
    if not math.isfinite(value):
        raise ValueError(f'{value_name} must be finite; NaN and infinity are not allowed.')
    return value

def validate_integer(raw_value: Any, value_name: str, minimum: int, maximum: int) -> int:
    """Validate an integer without silently truncating decimals."""
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{value_name} must be an integer.') from error
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f'{value_name} must be an integer.')
    value = int(numeric_value)
    if value < minimum or value > maximum:
        raise ValueError(f'{value_name} must be between {minimum} and {maximum}.')
    return value

def parse_optional_exact_root(raw_value: Any) -> float | None:
    """Parse an optional exact root such as 2 or sqrt(2)."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        expression = safe_sympify(text.replace('^', '**'), locals={'pi': sp.pi, 'E': sp.E, 'sqrt': sp.sqrt})
        evaluated = complex(sp.N(expression, 40))
    except (sp.SympifyError, TypeError, ValueError, OverflowError, SyntaxError) as error:
        raise ValueError('The optional exact root must be a finite real number or a simple expression such as sqrt(2).') from error
    if abs(evaluated.imag) > ZERO_TOLERANCE or not math.isfinite(evaluated.real):
        raise ValueError('The optional exact root must evaluate to a finite real value.')
    return float(evaluated.real)

def create_input_signature(function_text: str, initial_x: Any, tolerance: Any, maximum_iterations: Any, exact_root_text: str) -> str:
    """Create a stable signature to prevent stale Streamlit results."""
    payload = '|'.join([str(function_text).strip(), repr(initial_x), repr(tolerance), repr(maximum_iterations), str(exact_root_text).strip()])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def reject_unsupported_constructs(expression: sp.Expr) -> None:
    """Reject symbolic objects that are unsafe or irrelevant here."""
    unsupported_atoms = (AppliedUndef, sp.Derivative, sp.Integral, sp.Sum, sp.Product, sp.Limit)
    if expression.has(*unsupported_atoms):
        raise ValueError('The function contains unsupported symbolic operations.')
    if isinstance(expression, Relational) or expression.has(Relational):
        raise ValueError('Enter a function expression, not an equation or inequality.')
    unexpected_symbols = expression.free_symbols.difference({X_SYMBOL})
    if unexpected_symbols:
        names = ', '.join(sorted((str(symbol) for symbol in unexpected_symbols)))
        raise ValueError(f'Only x is allowed as a variable. Unsupported: {names}.')
    if expression.has(sp.I, sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError('The function contains a complex or non-finite symbolic value.')

def parse_function(function_text: str) -> tuple[sp.Expr, sp.Expr, sp.Expr, Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any]]:
    """Parse f, f′, and f″ and create numerical functions."""
    if not isinstance(function_text, str) or not function_text.strip():
        raise ValueError('Enter a function before solving.')
    text = function_text.strip().replace('^', '**')
    if '=' in text:
        raise ValueError('Enter only f(x), without an equals sign.')
    try:
        expression = safe_sympify(text, locals=ALLOWED_FUNCTION_NAMES, evaluate=True)
    except (sp.SympifyError, TypeError, ValueError, SyntaxError) as error:
        raise ValueError('The function has an invalid format. Use expressions such as x**3 - x - 2, exp(-x) - x, or x**2 - 2.') from error
    if not isinstance(expression, sp.Expr):
        raise ValueError('The function could not be interpreted.')
    reject_unsupported_constructs(expression)
    derivative_expression = sp.diff(expression, X_SYMBOL)
    second_derivative_expression = sp.diff(expression, X_SYMBOL, 2)
    if derivative_expression == 0:
        raise ValueError('The derivative is identically zero. A constant function cannot be solved with Newton-Raphson.')
    try:
        function = sp.lambdify(X_SYMBOL, expression, modules=['numpy'])
        derivative = sp.lambdify(X_SYMBOL, derivative_expression, modules=['numpy'])
        second_derivative = sp.lambdify(X_SYMBOL, second_derivative_expression, modules=['numpy'])
    except (TypeError, ValueError, NameError) as error:
        raise ValueError('The function or its derivatives could not be converted to numerical form.') from error
    return (expression, derivative_expression, second_derivative_expression, function, derivative, second_derivative)

def evaluate_real_scalar(numerical_function: Callable[[Any], Any], x_value: float, value_name: str) -> float:
    """Evaluate one finite real scalar safely."""
    try:
        with np.errstate(all='raise'):
            raw_value = numerical_function(float(x_value))
        array = np.asarray(raw_value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as error:
        raise ValueError(f'{value_name} is undefined at x = {x_value:.12g}. Reason: {error}') from error
    if array.size != 1:
        raise ValueError(f'{value_name} did not return a scalar at x = {x_value:.12g}.')
    scalar = array.reshape(-1)[0]
    if np.iscomplexobj(scalar):
        complex_value = complex(scalar)
        if abs(complex_value.imag) > ZERO_TOLERANCE:
            raise ValueError(f'{value_name} is complex at x = {x_value:.12g}.')
        scalar = complex_value.real
    return validate_finite_real(scalar, f'{value_name} at x = {x_value:.12g}')

def evaluate_real_array(numerical_function: Callable[[Any], Any], x_values: np.ndarray) -> np.ndarray:
    """Evaluate a function for plotting, replacing invalid points by NaN."""
    try:
        with np.errstate(all='ignore'):
            raw_values = numerical_function(x_values)
        values = np.asarray(raw_values)
    except Exception:
        return np.full_like(x_values, np.nan, dtype=float)
    if values.ndim == 0:
        values = np.full_like(x_values, values, dtype=complex if np.iscomplexobj(values) else float)
    else:
        try:
            values = np.broadcast_to(values, x_values.shape)
        except ValueError:
            return np.full_like(x_values, np.nan, dtype=float)
    if np.iscomplexobj(values):
        imaginary = np.abs(np.imag(values))
        result = np.real(values).astype(float)
        result[imaginary > ZERO_TOLERANCE] = np.nan
    else:
        try:
            result = values.astype(float)
        except (TypeError, ValueError):
            return np.full_like(x_values, np.nan, dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result

def estimate_multiplicity(function_value: float, derivative_value: float, second_derivative_value: float) -> float | None:
    """Estimate local multiplicity from f′²/(f′²−ff″)."""
    denominator = derivative_value ** 2 - function_value * second_derivative_value
    scale = max(1.0, abs(derivative_value ** 2), abs(function_value * second_derivative_value))
    if abs(denominator) <= 100.0 * np.finfo(float).eps * scale:
        return None
    estimate = derivative_value ** 2 / denominator
    if not math.isfinite(estimate):
        return None
    return float(estimate)

def derivative_sensitivity_indicator(x_value: float, function_value: float, derivative_value: float) -> float | None:
    """Return a scale-aware indicator of sensitivity in f/f′."""
    derivative_magnitude = abs(derivative_value)
    if derivative_magnitude <= np.finfo(float).tiny:
        return None
    indicator = max(1.0, abs(function_value)) / derivative_magnitude / max(1.0, abs(x_value))
    return float(indicator) if math.isfinite(indicator) else None

def observed_order_from_errors(errors: list[float]) -> float | None:
    """Estimate convergence order from three positive error indicators."""
    if len(errors) < 3:
        return None
    error_0 = float(errors[-3])
    error_1 = float(errors[-2])
    error_2 = float(errors[-1])
    if error_0 <= ZERO_TOLERANCE or error_1 <= ZERO_TOLERANCE or error_2 <= ZERO_TOLERANCE:
        return None
    denominator = math.log(error_1 / error_0)
    if abs(denominator) <= ZERO_TOLERANCE:
        return None
    observed_order = math.log(error_2 / error_1) / denominator
    return float(observed_order) if math.isfinite(observed_order) else None

def quadratic_indicator_from_errors(errors: list[float]) -> float | None:
    """Calculate e_(n+1)/e_n² from two positive error indicators."""
    if len(errors) < 2:
        return None
    previous_error = float(errors[-2])
    current_error = float(errors[-1])
    if previous_error <= math.sqrt(np.finfo(float).tiny) or current_error < 0.0:
        return None
    indicator = current_error / previous_error ** 2
    return float(indicator) if math.isfinite(indicator) else None

def determine_symbolic_multiplicity(expression: sp.Expr, exact_root: float, maximum_order: int=12) -> int | None:
    """Attempt to identify root multiplicity from successive derivatives."""
    root_value = sp.Float(exact_root, 40)
    for order in range(0, maximum_order + 1):
        derivative_expression = sp.diff(expression, X_SYMBOL, order)
        try:
            value = complex(sp.N(derivative_expression.subs(X_SYMBOL, root_value), 50))
        except (TypeError, ValueError, OverflowError):
            return None
        if abs(value.imag) > ZERO_TOLERANCE or not math.isfinite(value.real):
            return None
        if abs(value.real) > 1e-10:
            return order
    return None

def error_result(message: str, input_signature: str='') -> NewtonResult:
    """Create a complete failed result."""
    return NewtonResult(status='error', success=False, converged=False, method=METHOD_NAME, message=message, stopping_reason='Execution stopped during input validation or numerical evaluation.', function_text='', function_expression=None, derivative_expression=None, second_derivative_expression=None, initial_x=None, tolerance=None, maximum_iterations=0, known_exact_root=None, exact_root_verified=None, symbolic_multiplicity_at_exact_root=None, iterations=(), approximate_root=None, function_at_root=None, derivative_at_root=None, final_absolute_step=None, final_relative_step_percent=None, final_residual=None, final_exact_error=None, latest_observed_order=None, latest_quadratic_indicator=None, warnings=(), input_signature=input_signature, execution_datetime=current_report_datetime())

def solve_newton_raphson(function_text: str, initial_x_input: Any, tolerance_input: Any, maximum_iterations_input: Any, exact_root_text: str='') -> NewtonResult:
    """Solve f(x)=0 with the standard Newton-Raphson update."""
    input_signature = create_input_signature(function_text=function_text, initial_x=initial_x_input, tolerance=tolerance_input, maximum_iterations=maximum_iterations_input, exact_root_text=exact_root_text)
    try:
        initial_x = validate_finite_real(initial_x_input, 'Initial approximation x0')
        tolerance = validate_finite_real(tolerance_input, 'Tolerance')
        maximum_iterations = validate_integer(maximum_iterations_input, 'Maximum iterations', MIN_ITERATIONS, MAX_ITERATIONS)
        known_exact_root = parse_optional_exact_root(exact_root_text)
        if tolerance <= 0.0:
            raise ValueError('Tolerance must be greater than zero.')
        expression, derivative_expression, second_derivative_expression, function, derivative, second_derivative = parse_function(function_text)
        warnings: list[str] = []
        exact_root_verified: bool | None = None
        symbolic_multiplicity: int | None = None
        current_x = float(initial_x)
        current_function = evaluate_real_scalar(function, current_x, 'f(x)')
        initial_function_scale = max(1.0, abs(current_function))
        residual_tolerance = tolerance * initial_function_scale
        if known_exact_root is not None:
            exact_function_value = evaluate_real_scalar(function, known_exact_root, 'f(x)')
            verification_tolerance = max(tolerance, 1e-10)
            exact_root_verified = abs(exact_function_value) <= verification_tolerance
            if not exact_root_verified:
                raise ValueError('The supplied exact root does not satisfy f(x)=0 within the numerical verification tolerance.')
            symbolic_multiplicity = determine_symbolic_multiplicity(expression, known_exact_root)
            if symbolic_multiplicity is not None and symbolic_multiplicity > 1:
                warnings.append(f'The supplied root is multiple (detected multiplicity m = {symbolic_multiplicity}). Standard Newton-Raphson is generally only linearly convergent for repeated roots; use the Multiple Roots Solver for faster convergence.')
        history: list[NewtonIteration] = []
        convergence_indicators: list[float] = []
        converged = False
        stopping_reason = 'Maximum iterations reached before both the step and residual criteria were satisfied.'
        if current_function == 0.0:
            converged = True
            stopping_reason = 'The initial approximation evaluates to an exact floating-point root.'
        for iteration in range(1, maximum_iterations + 1):
            if converged:
                break
            derivative_value = evaluate_real_scalar(derivative, current_x, "f'(x)")
            second_derivative_value = evaluate_real_scalar(second_derivative, current_x, "f''(x)")
            derivative_zero_threshold = np.finfo(float).tiny * max(1.0, abs(current_x), abs(current_function))
            if abs(derivative_value) <= derivative_zero_threshold:
                raise ValueError('The derivative is numerically zero while f(x) is nonzero. Newton-Raphson cannot form a tangent update at this point.')
            correction = current_function / derivative_value
            if not math.isfinite(correction):
                raise ValueError("The Newton correction f(x)/f'(x) became NaN or infinity.")
            next_x = current_x - correction
            if not math.isfinite(next_x):
                raise ValueError('The next approximation became NaN or infinity.')
            next_function = evaluate_real_scalar(function, next_x, 'f(x)')
            absolute_step = abs(next_x - current_x)
            relative_step_percent = absolute_step / max(1.0, abs(next_x)) * 100.0
            residual = abs(next_function)
            sensitivity_indicator = derivative_sensitivity_indicator(current_x, current_function, derivative_value)
            estimated_multiplicity = estimate_multiplicity(current_function, derivative_value, second_derivative_value)
            exact_root_error: float | None = None
            exact_root_relative_percent: float | None = None
            if known_exact_root is not None:
                exact_root_error = abs(next_x - known_exact_root)
                exact_root_relative_percent = exact_root_error / max(1.0, abs(known_exact_root)) * 100.0
                convergence_indicators.append(exact_root_error)
            else:
                convergence_indicators.append(absolute_step)
            observed_order = observed_order_from_errors(convergence_indicators)
            quadratic_indicator = quadratic_indicator_from_errors(convergence_indicators)
            operation = f'x_{iteration} = {current_x:.15g} - ({current_function:.15g})/({derivative_value:.15g}) = {next_x:.15g}'
            history.append(NewtonIteration(iteration=iteration, x_n=float(current_x), function_value=float(current_function), derivative_value=float(derivative_value), second_derivative_value=float(second_derivative_value), correction=float(correction), x_next=float(next_x), function_next=float(next_function), absolute_step=float(absolute_step), relative_step_percent=float(relative_step_percent), residual=float(residual), derivative_sensitivity_indicator=sensitivity_indicator, estimated_multiplicity=estimated_multiplicity, exact_root_error=exact_root_error, exact_root_relative_error_percent=exact_root_relative_percent, observed_order=observed_order, quadratic_indicator=quadratic_indicator, operation=operation, status='Completed'))
            scaled_step_tolerance = tolerance * max(1.0, abs(next_x))
            step_condition = absolute_step <= scaled_step_tolerance
            residual_condition = residual <= residual_tolerance
            exact_condition = known_exact_root is not None and exact_root_error is not None and (exact_root_error <= tolerance * max(1.0, abs(known_exact_root)))
            if next_function == 0.0:
                converged = True
                stopping_reason = 'The function evaluated to an exact floating-point zero at the new approximation.'
                current_x = float(next_x)
                current_function = float(next_function)
                break
            if step_condition and residual_condition:
                converged = True
                stopping_reason = 'Both the scaled step-size tolerance and residual tolerance were satisfied.'
                current_x = float(next_x)
                current_function = float(next_function)
                break
            if exact_condition and residual_condition:
                converged = True
                stopping_reason = 'The known-root error and residual tolerance were satisfied.'
                current_x = float(next_x)
                current_function = float(next_function)
                break
            if next_x == current_x and (not residual_condition):
                raise ValueError('The iteration stagnated because floating-point arithmetic could not change x while the residual remained above tolerance.')
            if abs(next_x) >= VALUE_MAGNITUDE_WARNING or abs(next_function) >= VALUE_MAGNITUDE_WARNING:
                warnings.append("The iteration reached a very large magnitude. The initial approximation may be outside the method's convergence region.")
            current_x = float(next_x)
            current_function = float(next_function)
        approximate_root = float(current_x)
        function_at_root = evaluate_real_scalar(function, approximate_root, 'f(x)')
        derivative_at_root = evaluate_real_scalar(derivative, approximate_root, "f'(x)")
        if history:
            final_absolute_step = history[-1].absolute_step
            final_relative_step_percent = history[-1].relative_step_percent
            latest_observed_order = next((item.observed_order for item in reversed(history) if item.observed_order is not None and math.isfinite(item.observed_order)), None)
            latest_quadratic_indicator = next((item.quadratic_indicator for item in reversed(history) if item.quadratic_indicator is not None and math.isfinite(item.quadratic_indicator)), None)
            latest_multiplicity_estimate = next((item.estimated_multiplicity for item in reversed(history) if item.estimated_multiplicity is not None and math.isfinite(item.estimated_multiplicity)), None)
            maximum_sensitivity = max((item.derivative_sensitivity_indicator for item in history if item.derivative_sensitivity_indicator is not None), default=None)
        else:
            final_absolute_step = 0.0
            final_relative_step_percent = 0.0
            latest_observed_order = None
            latest_quadratic_indicator = None
            latest_multiplicity_estimate = None
            maximum_sensitivity = None
        final_residual = abs(function_at_root)
        final_exact_error = abs(approximate_root - known_exact_root) if known_exact_root is not None else None
        if latest_multiplicity_estimate is not None and latest_multiplicity_estimate >= MULTIPLE_ROOT_WARNING_THRESHOLD:
            warnings.append(f'The derivative-based multiplicity estimate suggests a repeated root (m ≈ {latest_multiplicity_estimate:.6g}). Standard Newton-Raphson may converge only linearly; consider the Multiple Roots Solver.')
        if maximum_sensitivity is not None and maximum_sensitivity >= DERIVATIVE_SENSITIVITY_WARNING:
            warnings.append('The Newton correction became highly sensitive because the derivative was extremely small relative to the local function scale.')
        if converged and latest_observed_order is not None and (latest_observed_order < QUADRATIC_ORDER_WARNING_THRESHOLD):
            warnings.append('The observed convergence was weaker than the expected quadratic behavior near a simple root. A repeated root, a distant initial approximation, or round-off may be responsible.')
        if not converged:
            warnings.append('The final approximation is reported, but the complete convergence criteria were not satisfied.')
        return NewtonResult(status='success', success=True, converged=converged, method=METHOD_NAME, message='Root refined successfully.' if converged else 'Maximum iterations reached; the final approximation is shown.', stopping_reason=stopping_reason, function_text=function_text.strip(), function_expression=expression, derivative_expression=derivative_expression, second_derivative_expression=second_derivative_expression, initial_x=initial_x, tolerance=tolerance, maximum_iterations=maximum_iterations, known_exact_root=known_exact_root, exact_root_verified=exact_root_verified, symbolic_multiplicity_at_exact_root=symbolic_multiplicity, iterations=tuple(history), approximate_root=approximate_root, function_at_root=function_at_root, derivative_at_root=derivative_at_root, final_absolute_step=final_absolute_step, final_relative_step_percent=final_relative_step_percent, final_residual=final_residual, final_exact_error=final_exact_error, latest_observed_order=latest_observed_order, latest_quadratic_indicator=latest_quadratic_indicator, warnings=tuple(dict.fromkeys(warnings)), input_signature=input_signature, execution_datetime=current_report_datetime())
    except (ValueError, TypeError, ArithmeticError, OverflowError) as error:
        return error_result(message=str(error), input_signature=input_signature)

def iterations_dataframe(result: NewtonResult) -> pd.DataFrame:
    """Return the complete Newton-Raphson iteration table."""
    return pd.DataFrame([{'Iteration': item.iteration, 'x_n': item.x_n, 'f(x_n)': item.function_value, "f'(x_n)": item.derivative_value, "f''(x_n)": item.second_derivative_value, "Correction f/f'": item.correction, 'x_(n+1)': item.x_next, 'f(x_(n+1))': item.function_next, 'Absolute Step': item.absolute_step, 'Scaled Relative Step (%)': item.relative_step_percent, 'Residual |f(x_(n+1))|': item.residual, 'Derivative Sensitivity Indicator': item.derivative_sensitivity_indicator, 'Estimated Multiplicity': item.estimated_multiplicity, 'Exact Root Error': item.exact_root_error, 'Exact Root Relative Error (%)': item.exact_root_relative_error_percent, 'Observed Order': item.observed_order, 'Quadratic Indicator e_(n+1)/e_n^2': item.quadratic_indicator, 'Operation': item.operation, 'Status': item.status} for item in result.iterations])

def method_formula_dataframe(result: NewtonResult) -> pd.DataFrame:
    """Return formulas and numerical interpretations."""
    return pd.DataFrame({'Item': ['Problem', 'Newton-Raphson Update', 'Tangent Slope', 'Absolute Step', 'Scaled Relative Step', 'Residual', 'Multiplicity Estimate', 'Expected Simple-Root Convergence', 'Quadratic Indicator', 'Complete Stopping Test'], 'Formula / Meaning': ['Solve f(x)=0', "x_(n+1) = x_n - f(x_n)/f'(x_n)", "f'(x_n)", '|x_(n+1) - x_n|', '|x_(n+1)-x_n| / max(1,|x_(n+1)|) × 100%', '|f(x_(n+1))|', "m_est = f'(x)^2 / (f'(x)^2 - f(x)f''(x))", 'Quadratic near a simple root under standard assumptions', 'e_(n+1)/e_n^2 should approach a finite constant', 'Require both a sufficiently small scaled step and a sufficiently small residual']})

def convergence_diagnostics_dataframe(result: NewtonResult) -> pd.DataFrame:
    """Return convergence and root diagnostics."""
    rows = [{'Diagnostic': 'Known exact root', 'Value': result.known_exact_root, 'Interpretation': 'Optional reference supplied by the user'}, {'Diagnostic': 'Detected multiplicity at known root', 'Value': result.symbolic_multiplicity_at_exact_root, 'Interpretation': 'Available when a known exact root was supplied'}, {'Diagnostic': 'Latest observed convergence order', 'Value': result.latest_observed_order, 'Interpretation': 'Expected to approach 2 near a simple root'}, {'Diagnostic': 'Latest quadratic indicator', 'Value': result.latest_quadratic_indicator, 'Interpretation': 'e_(n+1)/e_n² or its step-based analogue'}, {'Diagnostic': 'Final exact-root error', 'Value': result.final_exact_error, 'Interpretation': 'Available only with a known exact root'}]
    for item in result.iterations:
        rows.append({'Diagnostic': f'Estimated multiplicity at iteration {item.iteration}', 'Value': item.estimated_multiplicity, 'Interpretation': 'A value near 1 indicates a simple root'})
    return pd.DataFrame(rows)

def summary_dataframe(result: NewtonResult) -> pd.DataFrame:
    return pd.DataFrame({'Property': ['Method', 'Status', 'Converged', 'Function', 'First Derivative', 'Second Derivative', 'Initial Approximation', 'Known Exact Root', 'Exact Root Verified', 'Detected Multiplicity at Known Root', 'Tolerance', 'Maximum Iterations', 'Iterations Used', 'Approximate Root', 'f(Approximate Root)', "f'(Approximate Root)", 'Final Absolute Step', 'Final Relative Step (%)', 'Final Residual', 'Final Exact-Root Error', 'Latest Observed Order', 'Latest Quadratic Indicator', 'Stopping Reason', 'Warnings', 'Execution Date'], 'Value': [result.method, result.status, 'Yes' if result.converged else 'No', str(result.function_expression), str(result.derivative_expression), str(result.second_derivative_expression), result.initial_x, result.known_exact_root, result.exact_root_verified, result.symbolic_multiplicity_at_exact_root, result.tolerance, result.maximum_iterations, len(result.iterations), result.approximate_root, result.function_at_root, result.derivative_at_root, result.final_absolute_step, result.final_relative_step_percent, result.final_residual, result.final_exact_error, result.latest_observed_order, result.latest_quadratic_indicator, result.stopping_reason, ' | '.join(result.warnings) if result.warnings else 'None', result.execution_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')]})

def build_plot_dataframe(result: NewtonResult, sample_count: int=600) -> pd.DataFrame:
    if result.function_expression is None or result.approximate_root is None or result.initial_x is None:
        return pd.DataFrame()
    function = sp.lambdify(X_SYMBOL, result.function_expression, modules=['numpy'])
    points = [result.initial_x, result.approximate_root]
    points.extend((item.x_next for item in result.iterations))
    if result.known_exact_root is not None:
        points.append(result.known_exact_root)
    minimum_x = min(points)
    maximum_x = max(points)
    span = maximum_x - minimum_x
    padding = 1.0 if span <= 0.0 else max(0.25 * span, 0.5)
    graph_x = np.linspace(minimum_x - padding, maximum_x + padding, sample_count)
    graph_y = evaluate_real_array(function, graph_x)
    row_count = max(sample_count, len(result.iterations), 1)
    dataframe = pd.DataFrame(index=range(row_count))
    dataframe['Function x'] = pd.Series(graph_x)
    dataframe['f(x)'] = pd.Series(graph_y)
    dataframe['Iteration'] = pd.Series([item.iteration for item in result.iterations])
    dataframe['Approximation'] = pd.Series([item.x_next for item in result.iterations])
    dataframe['Absolute Step'] = pd.Series([item.absolute_step for item in result.iterations])
    dataframe['Residual'] = pd.Series([item.residual for item in result.iterations])
    dataframe['Exact Root Error'] = pd.Series([item.exact_root_error for item in result.iterations])
    dataframe['Estimated Multiplicity'] = pd.Series([item.estimated_multiplicity for item in result.iterations])
    dataframe['Observed Order'] = pd.Series([item.observed_order for item in result.iterations])
    dataframe['Quadratic Indicator'] = pd.Series([item.quadratic_indicator for item in result.iterations])
    return dataframe

def create_function_figure(result: NewtonResult) -> Figure:
    """Plot f(x), iteration points, and the final root."""
    if result.function_expression is None or result.approximate_root is None:
        raise ValueError('A successful result is required for plotting.')
    plot_data = build_plot_dataframe(result)
    if plot_data.empty:
        raise ValueError('Plot data are unavailable.')
    function_x = plot_data['Function x'].to_numpy(dtype=float)
    function_y = plot_data['f(x)'].to_numpy(dtype=float)
    valid = np.isfinite(function_x) & np.isfinite(function_y)
    if np.count_nonzero(valid) < 2:
        raise ValueError('The function has insufficient finite values for plotting.')
    figure, axis = plt.subplots(figsize=(5, 3))
    axis.plot(function_x[valid], function_y[valid], linewidth=2.0, label='f(x)')
    axis.axhline(0.0, linewidth=1.0)
    if result.iterations:
        iteration_x = np.asarray([item.x_next for item in result.iterations], dtype=float)
        iteration_y = np.asarray([item.function_next for item in result.iterations], dtype=float)
        axis.scatter(iteration_x, iteration_y, s=55, label='Newton iteration points', zorder=5)
    axis.scatter([result.approximate_root], [result.function_at_root], s=110, marker='*', label=f'Approximate root = {result.approximate_root:.8g}', zorder=6)
    axis.axvline(result.approximate_root, linestyle='--', linewidth=1.1)
    if result.known_exact_root is not None:
        axis.axvline(result.known_exact_root, linestyle=':', linewidth=1.3, label=f'Known root = {result.known_exact_root:.8g}')
    axis.set_title('Function and Newton-Raphson Iterations')
    axis.set_xlabel('x')
    axis.set_ylabel('f(x)')
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure

def create_approximation_figure(result: NewtonResult) -> Figure:
    """Plot the root estimate by iteration."""
    figure, axis = plt.subplots(figsize=(5, 3))
    if result.iterations:
        iterations = np.asarray([item.iteration for item in result.iterations], dtype=int)
        approximations = np.asarray([item.x_next for item in result.iterations], dtype=float)
        axis.plot(iterations, approximations, marker='o', linewidth=2.0, label='Newton approximation')
        axis.axhline(result.approximate_root, linestyle='--', linewidth=1.0, label='Final approximation')
        if result.known_exact_root is not None:
            axis.axhline(result.known_exact_root, linestyle=':', linewidth=1.2, label='Known exact root')
    else:
        axis.scatter([0], [result.approximate_root], s=90, label='Initial value is the root')
    axis.set_title('Root Approximation by Iteration')
    axis.set_xlabel('Iteration')
    axis.set_ylabel('x')
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure

def create_convergence_figure(result: NewtonResult) -> Figure:
    """Plot step, residual, and optional exact-root error."""
    figure, axis = plt.subplots(figsize=(5, 3))
    if not result.iterations:
        axis.text(0.5, 0.5, 'No iteration was required.', ha='center', va='center', transform=axis.transAxes)
    else:
        iterations = np.asarray([item.iteration for item in result.iterations], dtype=int)
        steps = np.maximum(np.asarray([item.absolute_step for item in result.iterations], dtype=float), np.finfo(float).tiny)
        residuals = np.maximum(np.asarray([item.residual for item in result.iterations], dtype=float), np.finfo(float).tiny)
        axis.semilogy(iterations, steps, marker='o', linewidth=2.0, label='Absolute step')
        axis.semilogy(iterations, residuals, marker='o', linewidth=2.0, label='Residual')
        if result.known_exact_root is not None:
            true_errors = np.asarray([item.exact_root_error if item.exact_root_error is not None else np.nan for item in result.iterations], dtype=float)
            valid = np.isfinite(true_errors) & (true_errors >= 0.0)
            if np.any(valid):
                axis.semilogy(iterations[valid], np.maximum(true_errors[valid], np.finfo(float).tiny), marker='o', linewidth=2.0, label='Known-root error')
    axis.set_title('Newton-Raphson Convergence Indicators')
    axis.set_xlabel('Iteration')
    axis.set_ylabel('Magnitude (log scale)')
    axis.grid(True, which='both', alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure

def figure_to_png_bytes(figure: Figure) -> bytes:
    """Serialize a matplotlib figure to PNG bytes."""
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=180, bbox_inches='tight')
    buffer.seek(0)
    return buffer.getvalue()

def render_final_result(result: NewtonResult) -> None:
    """Render the compact final-result card."""
    if not result.success:
        st.error(result.message)
        return
    if result.converged:
        st.success(result.message)
    else:
        st.warning(result.message)
    first_metrics = st.columns(2)
    first_metrics[0].metric('Approximate Root', format_number(result.approximate_root))
    first_metrics[1].metric('Iterations', len(result.iterations))
    second_metrics = st.columns(2)
    second_metrics[0].metric('Final Residual', format_number(result.final_residual))
    second_metrics[1].metric('Final Absolute Step', format_number(result.final_absolute_step))
    st.markdown(f'**Stopping reason:** {result.stopping_reason}')
    if result.known_exact_root is not None:
        exact_metrics = st.columns(2)
        exact_metrics[0].metric('Known Exact Root', format_number(result.known_exact_root))
        exact_metrics[1].metric('Final Exact-Root Error', format_number(result.final_exact_error))
    if result.latest_observed_order is not None:
        st.metric('Latest Observed Order', format_number(result.latest_observed_order))
    for warning in result.warnings:
        st.warning(warning)

def render_method_summary(result: NewtonResult) -> None:
    """Render formulas and problem setup."""
    st.subheader('Method Formula and Problem Setup')
    formula_column, convergence_column = st.columns(2)
    with formula_column:
        st.latex("x_{n+1}=x_n-\\frac{f(x_n)}{f'(x_n)}")
        st.latex('e_a=\\left|x_{n+1}-x_n\\right|')
    with convergence_column:
        st.latex('E_{n+1}\\approx C E_n^2')
        st.latex("\\widehat m(x)=\\frac{[f'(x)]^2}{[f'(x)]^2-f(x)f''(x)}")
    summary = pd.DataFrame({'Property': ['Function f(x)', "First derivative f'(x)", "Second derivative f''(x)", 'Initial approximation', 'Tolerance', 'Maximum iterations', 'Expected local behavior'], 'Value': [str(result.function_expression), str(result.derivative_expression), str(result.second_derivative_expression), result.initial_x, result.tolerance, result.maximum_iterations, 'Quadratic convergence near a simple root']})
    st.dataframe(round_numeric_dataframe(summary), use_container_width=False, hide_index=True)

def render_iteration_table(result: NewtonResult) -> None:
    """Render every tangent-line update."""
    st.subheader('Newton-Raphson Iteration Table')
    dataframe = iterations_dataframe(result)
    if dataframe.empty:
        st.info('No iteration was required because the initial approximation evaluated to an exact root.')
        return
    display_columns = ['Iteration', 'x_n', 'f(x_n)', "f'(x_n)", "f''(x_n)", "Correction f/f'", 'x_(n+1)', 'f(x_(n+1))', 'Absolute Step', 'Scaled Relative Step (%)', 'Residual |f(x_(n+1))|', 'Estimated Multiplicity', 'Exact Root Error', 'Observed Order', 'Quadratic Indicator e_(n+1)/e_n^2']
    st.dataframe(round_numeric_dataframe(dataframe[[column for column in display_columns if column in dataframe.columns]]), use_container_width=False, hide_index=True)
    st.caption('The solver requires both a sufficiently small scaled step and a sufficiently small residual. Step Change is an approximate error indicator, not the true root error.')
    with st.expander('Detailed Tangent-Line Operations', expanded=False):
        for item in result.iterations:
            st.markdown(f'**Iteration {item.iteration}:**')
            st.code(item.operation, language=None)
            st.caption(f'Correction = {item.correction:.12g}; absolute step = {item.absolute_step:.12g}; residual = {item.residual:.12g}.')

def render_convergence_diagnostics(result: NewtonResult) -> None:
    """Render order, multiplicity, and exact-root diagnostics."""
    st.subheader('Convergence and Root Diagnostics')
    st.dataframe(round_numeric_dataframe(convergence_diagnostics_dataframe(result)), use_container_width=False, hide_index=True)
    if result.symbolic_multiplicity_at_exact_root is not None:
        if result.symbolic_multiplicity_at_exact_root == 1:
            st.success('The supplied exact root is a simple root, which is consistent with quadratic local convergence.')
        else:
            st.warning('The supplied exact root is multiple. Standard Newton-Raphson normally loses quadratic convergence.')

def render_graphs(result: NewtonResult) -> None:
    """Render function, approximation, and convergence plots."""
    graph_column, approximation_column, error_column = st.columns(3)
    with graph_column:
        with st.container(border=True):
            st.subheader('Function Graph')
            try:
                figure = create_function_figure(result)
                st.pyplot(figure, use_container_width=False)
                plt.close(figure)
            except (ValueError, TypeError, ArithmeticError) as error:
                st.warning(f'The function graph could not be displayed. Details: {error}')
    with approximation_column:
        with st.container(border=True):
            st.subheader('Root Approximation')
            figure = create_approximation_figure(result)
            st.pyplot(figure, use_container_width=False)
            plt.close(figure)
    with error_column:
        with st.container(border=True):
            st.subheader('Error Analysis')
            figure = create_convergence_figure(result)
            st.pyplot(figure, use_container_width=False)
            plt.close(figure)

def render_page() -> None:
    """Render the complete Newton-Raphson solver page."""
    st.set_page_config(page_title=f"{CONFIG['title']} | Numerical Methods", page_icon='📘', layout='wide')
    load_css()
    st.markdown(
        """
        <style>
        .stCode, code {
            color: #111827 !important;
            background: transparent !important;
            font-weight: 600 !important;
        }
        .solver-math {
            font-size: 1.05rem;
            color: #111827;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    navbar(active_page='solver')
    st.html(f"""\n        <section class="solver-hero">\n            <div>\n                <div class="page-label">\n                    {CONFIG['label']}\n                </div>\n                <h1>{CONFIG['title']}</h1>\n                <p>{CONFIG['description']}</p>\n                <div class="method-actions">\n                    <a\n                        href="/{CONFIG['lesson']}"\n                        target="_self"\n                        class="btn-outline-ui"\n                    >\n                        Review Lesson →\n                    </a>\n                    <a\n                        href="/{CONFIG['quiz']}"\n                        target="_self"\n                        class="btn-primary-ui"\n                    >\n                        Take Quiz →\n                    </a>\n                </div>\n            </div>\n        </section>\n        """)
    left_margin, main_area, right_margin = st.columns([0.035, 0.93, 0.035])
    with main_area:
        st.markdown('<main class="solver-wrapper solver-streamlit-area">', unsafe_allow_html=True)
        guide_column, conditions_column = st.columns(2)
        with guide_column:
            with st.container(border=True):
                st.subheader('How to Write the Function')
                st.markdown('\n                    Enter **f(x)** without an equals sign and use\n                    only **x** as the variable.\n\n                    - Powers: write `x**2`, not `x^2`.\n                    - Multiplication: write `2*x`, not `2x`.\n                    - Functions: `sin(x)`, `cos(x)`,\n                      `sqrt(x)`, `exp(x)`, and `log(x)`.\n                    - Example: `x**3 - x - 2`.\n                    ')
                st.markdown('**Method formula**')
                st.latex(CONFIG['formula'])
        with conditions_column:
            with st.container(border=True):
                st.subheader('Before Solving')
                for condition in CONFIG['conditions']:
                    st.markdown(f'- {condition}')
                st.info('Newton-Raphson is an open method: convergence is not guaranteed for every starting value. Inspect the step, residual, and diagnostic tables.')
        input_column, result_column = st.columns([1.35, 1.0])
        with input_column:
            with st.container(border=True):
                st.markdown('<h3 class="solver-box-title">Input</h3>', unsafe_allow_html=True)
                st.markdown('<div class="input-label-ui">Function f(x)</div>', unsafe_allow_html=True)
                function_text = st.text_input('Function f(x)', value=CONFIG['default_eq'], placeholder='Example: x**3 - x - 2', label_visibility='collapsed', key='newton_raphson_function')
                first_row = st.columns(2)
                with first_row[0]:
                    st.markdown('<div class="input-label-ui">Initial approximation x₀</div>', unsafe_allow_html=True)
                    initial_x = st.number_input('Initial approximation x0', value=DEFAULT_X0, format='%.12g', label_visibility='collapsed', key='newton_raphson_x0')
                with first_row[1]:
                    st.markdown('<div class="input-label-ui">Tolerance</div>', unsafe_allow_html=True)
                    tolerance = st.number_input('Tolerance', min_value=MIN_TOLERANCE, value=DEFAULT_TOLERANCE, format='%.12g', label_visibility='collapsed', key='newton_raphson_tolerance')
                second_row = st.columns(2)
                with second_row[0]:
                    st.markdown('<div class="input-label-ui">Maximum iterations</div>', unsafe_allow_html=True)
                    maximum_iterations = st.number_input('Maximum iterations', min_value=MIN_ITERATIONS, max_value=MAX_ITERATIONS, value=DEFAULT_MAX_ITERATIONS, step=1, label_visibility='collapsed', key='newton_raphson_max_iterations')
                with second_row[1]:
                    st.markdown('<div class="input-label-ui">Known exact root — optional</div>', unsafe_allow_html=True)
                    exact_root_text = st.text_input('Known exact root', value='', placeholder='Example: 1.5213797068', label_visibility='collapsed', key='newton_raphson_exact_root')
                st.caption('The optional exact root is used only for verification, true-error analysis, multiplicity detection, and convergence-order estimation.')
                st.markdown('**Equation Preview**')
                try:
                    preview_expression, preview_derivative, preview_second_derivative, _, _, _ = parse_function(function_text)
                    st.latex('f(x)=' + sp.latex(preview_expression))
                    st.latex("f'(x)=" + sp.latex(preview_derivative))
                    st.latex("f''(x)=" + sp.latex(preview_second_derivative))
                except ValueError as error:
                    st.warning(str(error))
                current_signature = create_input_signature(function_text=function_text, initial_x=initial_x, tolerance=tolerance, maximum_iterations=maximum_iterations, exact_root_text=exact_root_text)
                solve_column, reset_column = st.columns(2)
                with solve_column:
                    solve_button = st.button('Solve', type='primary', use_container_width=False, key='newton_raphson_solve')
                with reset_column:
                    reset_button = st.button('Reset Result', use_container_width=False, key='newton_raphson_reset')
                if reset_button:
                    st.session_state.pop('newton_raphson_result', None)
                    st.rerun()
                if solve_button:
                    st.session_state['newton_raphson_result'] = solve_newton_raphson(function_text=function_text, initial_x_input=initial_x, tolerance_input=tolerance, maximum_iterations_input=maximum_iterations, exact_root_text=exact_root_text)
                    st.rerun()
                with st.expander('Example Inputs'):
                    st.code('Function: x**3 - x - 2\nx0 = 1.5\nTolerance = 1e-8\nKnown exact root = 1.5213797068045676', language=None)
                    st.code('Function: exp(-x) - x\nx0 = 0\nTolerance = 1e-10\nKnown exact root = 0.5671432904097838', language=None)
        with result_column:
            with st.container(border=True):
                st.markdown('<h3 class="solver-box-title">Final Result</h3>', unsafe_allow_html=True)
                result = st.session_state.get('newton_raphson_result')
                if result is None:
                    st.info('Enter the function and settings, then click Solve.')
                elif result.input_signature != current_signature:
                    st.info('The function or numerical settings have changed. Click Solve to calculate a new result.')
                else:
                    render_final_result(result)
        result = st.session_state.get('newton_raphson_result')
        result_is_current = result is not None and result.input_signature == current_signature
        if result_is_current and result.success:
            st.divider()
            render_method_summary(result)
            st.divider()
            render_iteration_table(result)
            st.divider()
            render_convergence_diagnostics(result)
            st.divider()
            render_graphs(result)
            st.divider()
            st.divider()
            navigation_left, navigation_right = st.columns(2)
            with navigation_left:
                if st.button('Review Newton-Raphson Lesson', use_container_width=False, key='newton_raphson_lesson_button'):
                    st.switch_page('pages/Newton_Raphson_Method.py')
            with navigation_right:
                if st.button('Back to Solver Menu', use_container_width=False, key='newton_raphson_menu_button'):
                    st.switch_page('pages/Numerical_Solver.py')
        st.markdown('</main>', unsafe_allow_html=True)
    st.html(f"""\n        <footer class="footer-ui">\n            <div>\n                NM • © 2026 Numerical Methods\n            </div>\n            <div>{CONFIG['footer']}</div>\n        </footer>\n        """)
if __name__ == '__main__':
    render_page()
