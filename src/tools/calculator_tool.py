"""
Calculator Tool - Performs mathematical calculations for agents
"""

import ast
import logging
import math
import operator
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000


def _safe_pow(base: float, exponent: float) -> float:
    """Power with a bounded exponent (prevents 9**9**9 style CPU/memory bombs)."""
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError("Exponent too large")
    return float(base) ** float(exponent)


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs, "round": round, "max": max, "min": min, "sum": sum,
    "sqrt": math.sqrt, "pow": _safe_pow,
}


def _eval_node(node: ast.AST) -> Any:
    """Evaluate a whitelisted AST node (numbers, + - * / // % **, a few functions)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(e) for e in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


class CalculatorTool:
    """Calculator tool for mathematical operations"""
    
    def __init__(self):
        self.name = "Calculator"
    
    def evaluate(self, expression: str) -> Optional[float]:
        """
        Safely evaluate a mathematical expression (e.g. "2+2*3", "sqrt(16)+max(1,2)").

        Uses a whitelisted AST walker - never eval(). Returns None on any error.
        """
        try:
            if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
                logger.warning("Expression empty or too long")
                return None

            tree = ast.parse(expression.strip(), mode="eval")
            result = _eval_node(tree)
            logger.info(f"Calculation: {expression} = {result}")
            return float(result)

        except ZeroDivisionError:
            logger.error("Division by zero error")
            return None
        except (ValueError, TypeError, SyntaxError, OverflowError, RecursionError) as e:
            logger.warning(f"Calculation rejected/failed for {expression!r}: {e}")
            return None
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return None

    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide two numbers"""
        if b == 0:
            logger.error("Division by zero")
            return None
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to power"""
        return base ** exponent
    
    def square_root(self, x: float) -> Optional[float]:
        """Calculate square root"""
        if x < 0:
            logger.error("Cannot calculate square root of negative number")
            return None
        return x ** 0.5
    
    def percentage(self, value: float, percent: float) -> float:
        """Calculate percentage of value"""
        return value * (percent / 100)
    
    def compound_interest(self, principal: float, rate: float, time: float, compounds_per_year: int = 1) -> float:
        """
        Calculate compound interest
        
        Args:
            principal: Initial amount
            rate: Annual interest rate (as percentage, e.g., 5 for 5%)
            time: Time in years
            compounds_per_year: Compounding frequency
        
        Returns:
            Final amount
        """
        r = rate / 100  # Convert percentage to decimal
        n = compounds_per_year
        t = time
        
        amount = principal * (1 + r / n) ** (n * t)
        return amount
    
    def percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change"""
        if old_value == 0:
            logger.error("Cannot calculate percentage change from zero")
            return 0
        return ((new_value - old_value) / old_value) * 100
    
    def average(self, *values: float) -> Optional[float]:
        """Calculate average of values"""
        if not values:
            return None
        return sum(values) / len(values)
    
    def standard_deviation(self, *values: float) -> Optional[float]:
        """Calculate standard deviation"""
        if not values or len(values) < 2:
            return None
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# Singleton instance
calculator = CalculatorTool()


def execute_calculation(expression: str) -> dict:
    """
    Execute calculation and return formatted result
    
    Args:
        expression: Mathematical expression
    
    Returns:
        Result dict with status and value
    """
    try:
        result = calculator.evaluate(expression)
        
        if result is not None:
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "formatted": f"{result:,.2f}" if isinstance(result, float) else str(result)
            }
        else:
            return {
                "success": False,
                "expression": expression,
                "error": "Calculation failed"
            }
    
    except Exception as e:
        return {
            "success": False,
            "expression": expression,
            "error": str(e)
        }
