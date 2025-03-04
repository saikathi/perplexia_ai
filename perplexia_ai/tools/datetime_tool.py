"""DateTime tool for handling time-related queries."""

from datetime import datetime
import logging

class DateTimeTool:
    """A tool for handling date and time related queries."""
    
    def __init__(self):
        """Initialize the datetime tool with logging."""
        self.logger = logging.getLogger(__name__)
    
    def get_datetime(self, query: str = "") -> str:
        """Get formatted date/time information based on the query.
        
        Args:
            query: Optional query specifying what datetime info is needed
                (e.g., "date", "time", "day", "month", "year")
                
        Returns:
            str: Formatted date/time information
        """
        self.logger.info(f"Processing datetime query: {query}")
        
        now = datetime.now()
        query = query.lower()
        
        try:
            if "time" in query:
                return now.strftime("%I:%M %p")
            elif "day" in query:
                return now.strftime("%A")
            elif "month" in query:
                return now.strftime("%B")
            elif "year" in query:
                return str(now.year)
            elif "date" in query:
                return now.strftime("%B %d, %Y")
            else:
                return now.strftime("%B %d, %Y at %I:%M %p")
                
        except Exception as e:
            self.logger.error(f"Error processing datetime query: {str(e)}")
            raise ValueError(f"Could not process datetime query: {str(e)}") 