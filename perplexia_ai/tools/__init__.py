"""Tools and utilities for Perplexia AI."""

from typing import Union
from langchain.tools import StructuredTool
from .calculator import Calculator
from .datetime_tool import DateTimeTool

def calculate(query: str) -> Union[float, str]:
    """Use this tool for mathematical calculations.
    
    Args:
        query: A mathematical expression or question (e.g., "5 + 3" or "what is 15% of 80")
        
    Returns:
        The result of the calculation
    """
    calculator = Calculator()
    return calculator.calculate(query)

def get_datetime(query: str = "") -> str:
    """Use this tool for date and time related queries.
    
    Args:
        query: Optional query specifying what datetime info is needed
            (e.g., "date", "time", "day", "month", "year")
            
    Returns:
        Formatted date/time information
    """
    datetime_tool = DateTimeTool()
    return datetime_tool.get_datetime(query)

# Create structured tools
calculator_tool = StructuredTool(
    name="calculator",
    description="Useful for performing mathematical calculations",
    func=calculate,
    args_schema={"query": str}
)

datetime_tool = StructuredTool(
    name="datetime",
    description="Useful for getting current date and time information",
    func=get_datetime,
    args_schema={"query": str}
)

# Export available tools
available_tools = [calculator_tool, datetime_tool] 