"""
Calculator Tool - Performs mathematical calculations for agents
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CalculatorTool:
    """Calculator tool for mathematical operations"""
    
    def __init__(self):
        self.name = "Calculator"
    
    def evaluate(self, expression: str) -> Optional[float]:
        """
        Evaluate mathematical expression
        
        Args:
            expression: Mathematical expression (e.g., "2+2*3")
        
        Returns:
            Result of calculation or None on error
        """
        try:
            # Sanitize input to prevent injection
            if not self._is_safe_expression(expression):
                logger.warning(f"Unsafe expression rejected: {expression}")
                return None
            
            # Use eval safely with restricted namespace
            safe_dict = {
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "sum": sum,
                "pow": pow,
                "sqrt": lambda x: x ** 0.5,
                "__builtins__": {},
            }
            
            result = eval(expression, safe_dict)
            logger.info(f"Calculation: {expression} = {result}")
            return float(result)
        
        except ZeroDivisionError:
            logger.error("Division by zero error")
            return None
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return None
    
    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe to evaluate"""
        # Only allow numbers, operators, parentheses, and math functions
        safe_pattern = r'^[0-9+\-*/%().\s]*$'
        
        # Additional check: no dangerous keywords
        dangerous_keywords = ['import', 'exec', 'eval', 'open', 'os', '__']
        for keyword in dangerous_keywords:
            if keyword in expression.lower():
                return False
        
        return bool(re.match(safe_pattern, expression))
    
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
