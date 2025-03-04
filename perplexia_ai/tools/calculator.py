"""Calculator tool for basic mathematical operations."""

import re
from typing import Union
import logging

class Calculator:
    """A calculator tool that can handle basic mathematical operations."""
    
    def __init__(self):
        """Initialize the calculator with logging."""
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, query: str) -> Union[float, str]:
        """Extract and calculate mathematical operations from text.
        
        Args:
            query: The text containing a mathematical query
            
        Returns:
            float or str: The result of the calculation
            
        Raises:
            ValueError: If no valid calculation could be performed
        """
        self.logger.info(f"Processing calculation query: {query}")
        
        # Handle percentage calculations
        if '%' in query:
            return self._handle_percentage(query)
        
        # Extract numbers and operation
        numbers = re.findall(r'-?\d*\.?\d+', query)
        if not numbers:
            raise ValueError("No numbers found in query")
        
        numbers = [float(n) for n in numbers]
        
        # Determine operation
        if '+' in query or 'plus' in query.lower() or 'add' in query.lower():
            return sum(numbers)
        elif '-' in query or 'minus' in query.lower() or 'subtract' in query.lower():
            return numbers[0] - sum(numbers[1:])
        elif '*' in query or 'x' in query or 'times' in query.lower() or 'multiply' in query.lower():
            result = 1
            for n in numbers:
                result *= n
            return result
        elif '/' in query or 'divided by' in query.lower():
            if 0 in numbers[1:]:
                raise ValueError("Cannot divide by zero")
            result = numbers[0]
            for n in numbers[1:]:
                result /= n
            return result
        else:
            raise ValueError("No valid operation found in query")
    
    def _handle_percentage(self, query: str) -> float:
        """Handle percentage calculations.
        
        Args:
            query: The text containing a percentage calculation
            
        Returns:
            float: The result of the percentage calculation
            
        Raises:
            ValueError: If the percentage calculation cannot be performed
        """
        # Extract numbers
        numbers = re.findall(r'-?\d*\.?\d+', query)
        if len(numbers) < 2:
            raise ValueError("Need at least two numbers for percentage calculation")
        
        # Convert to floats
        numbers = [float(n) for n in numbers]
        
        # Handle "X% of Y" format
        if 'of' in query.lower():
            percentage = numbers[0]
            base = numbers[1]
            return (percentage / 100) * base
        
        # Default to simple percentage
        return numbers[0] * (numbers[1] / 100)
