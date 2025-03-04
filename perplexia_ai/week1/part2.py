"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.router.llm_router import RouterOutputParser
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
import logging

class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""
    
    def __init__(self):
        """Initialize the chat interface with logging."""
        super().__init__()
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for basic tools.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        """
        self.llm = ChatOpenAI()
        
        # Initialize tools
        from perplexia_ai.tools import available_tools
        self.tools = available_tools
        
        # Define router prompt template with tool detection
        router_template = """Given the following query, determine if it requires any tools and the most appropriate category.

        Available Tools:
        1. calculator - For mathematical calculations (e.g., "what is 5 + 3?", "calculate 15% of 80")
        2. datetime - For date/time queries (e.g., "what time is it?", "what's today's date?")
        
        Categories:
        1. CALCULATOR - For queries requiring mathematical calculations
        2. DATETIME - For queries about current date/time
        3. FACTUAL - For questions seeking specific facts
        4. OPINION - For questions seeking judgments
        5. PROCEDURAL - For questions about how to do something
        6. CLARIFICATION - For questions needing explanation
        
        Query: {input}
        
        Format your response as a JSON object with the following keys:
        {{
            "destination": "The category (must be one of: CALCULATOR, DATETIME, FACTUAL, OPINION, PROCEDURAL, CLARIFICATION)",
            "next_inputs": "the original input query"
        }}
        
        Example responses:
        {{
            "destination": "CALCULATOR",
            "next_inputs": "what is 5 + 3?"
        }}
        {{
            "destination": "DATETIME",
            "next_inputs": "what time is it?"
        }}
        {{
            "destination": "FACTUAL",
            "next_inputs": "What is machine learning?"
        }}
        
        Response:"""
        
        # Create the router prompt and chain
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"]
        )
        self.router_chain = router_prompt | self.llm | RouterOutputParser()
        
        # Initialize calculator
        self.calculator = Calculator()
        
        # Set up response prompts for each type
        self.response_prompts = {
            "CALCULATOR": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful math assistant.
                When given a calculation result, explain it clearly and concisely.
                Format: "The result of [operation] is [result]." """),
                ("user", "{query} = {result}")
            ]),
            "DATETIME": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant providing time information.
                Format the response in a natural, conversational way."""),
                ("user", "{result}")
            ]),
            "FACTUAL": ChatPromptTemplate.from_messages([
                ("system", """You provide factual, accurate information.
                Be concise and specific."""),
                ("user", "{query}")
            ]),
            "OPINION": ChatPromptTemplate.from_messages([
                ("system", """You provide balanced, well-reasoned perspectives.
                Consider multiple viewpoints."""),
                ("user", "{query}")
            ]),
            "PROCEDURAL": ChatPromptTemplate.from_messages([
                ("system", """You provide clear step-by-step instructions.
                Number each step."""),
                ("user", "{query}")
            ]),
            "CLARIFICATION": ChatPromptTemplate.from_messages([
                ("system", """You provide clear explanations with examples.
                Break down complex ideas."""),
                ("user", "{query}")
            ])
        }

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with tool support."""
        try:
            self.logger.info(f"\n🔍 Processing message: '{message}'")
            
            # Step 1: Use router chain to classify query
            self.logger.info("📋 Classifying query type...")
            route_result = self.router_chain.invoke({"input": message})
            query_type = route_result.get("destination", "FACTUAL").upper()
            self.logger.info(f"✨ Query classified as: {query_type}")
            
            # Step 2: Handle tool-based queries
            if query_type == "CALCULATOR":
                self.logger.info("🧮 Using calculator tool...")
                try:
                    result = self.tools[0].invoke({"query": message})
                    self.logger.info(f"✅ Calculation result: {result}")
                    response_prompt = self.response_prompts[query_type]
                    response = response_prompt.format_messages(query=message, result=result)
                    raw_response = self.llm.predict_messages(response)
                    return raw_response.content.strip()
                except Exception as calc_error:
                    self.logger.error(f"❌ Calculation error: {str(calc_error)}")
                    return f"I apologize, but I couldn't perform that calculation. Error: {str(calc_error)}"
                    
            elif query_type == "DATETIME":
                self.logger.info("🕒 Using datetime tool...")
                try:
                    result = self.tools[1].invoke({"query": message})
                    self.logger.info(f"✅ Datetime result: {result}")
                    response_prompt = self.response_prompts[query_type]
                    response = response_prompt.format_messages(result=result)  # Only pass result for datetime
                    raw_response = self.llm.predict_messages(response)
                    return raw_response.content.strip()
                except Exception as time_error:
                    self.logger.error(f"❌ Datetime error: {str(time_error)}")
                    return f"I apologize, but I couldn't process that time query. Error: {str(time_error)}"
            
            # Step 3: Handle regular queries
            self.logger.info(f"🤖 Generating response using {query_type} template...")
            response_prompt = self.response_prompts[query_type]
            response = response_prompt.format_messages(query=message)
            raw_response = self.llm.predict_messages(response)
            self.logger.info("✅ Response generated successfully")
            
            return raw_response.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error processing your request." 