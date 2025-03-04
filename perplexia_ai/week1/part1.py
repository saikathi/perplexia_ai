"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from perplexia_ai.core.chat_interface import ChatInterface
import logging

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        """Initialize the chat interface with logging."""
        super().__init__()
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding."""
        self.llm = ChatOpenAI()
        
        # Define router prompt template with properly escaped format instructions
        router_template = """Given the following query, determine the most appropriate category.
        
        Categories:
        1. FACTUAL - For direct questions seeking facts ("What is...?", "Who invented...?")
        2. ANALYTICAL - For process and reasoning questions ("How does...?", "Why do...?")
        3. COMPARISON - For questions about differences or similarities ("What's the difference between...?")
        4. DEFINITION - For requests to explain concepts ("Define...", "Explain...")
        
        Query: {input}
        
        Format your response as a JSON object with the following keys:
        {{
            "destination": "The category (must be one of: FACTUAL, ANALYTICAL, COMPARISON, DEFINITION)",
            "next_inputs": "the original input query"
        }}
        
        Example responses:
        {{
            "destination": "FACTUAL",
            "next_inputs": "What is photosynthesis?"
        }}
        {{
            "destination": "ANALYTICAL",
            "next_inputs": "How does a car engine work?"
        }}
        
        Response:"""
        
        # Create the router prompt
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"]
        )
        
        # Create the router chain using the newer syntax
        self.router_chain = router_prompt | self.llm | RouterOutputParser()
        
        # Set up response prompts for each type
        self.response_prompts = {
            "FACTUAL": ChatPromptTemplate.from_messages([
                ("system", """You provide direct, factual answers.
                Focus on key facts and verified information.
                Be concise (1-2 sentences).
                Example queries: "What is...?", "Who invented...?"
                Avoid opinions or detailed analysis."""),
                ("user", "{query}")
            ]),
            
            "ANALYTICAL": ChatPromptTemplate.from_messages([
                ("system", """You explain processes and reasoning.
                Break down complex topics into clear steps.
                Focus on how things work or why they happen.
                Example queries: "How does...?", "Why do...?"
                Include cause-and-effect relationships."""),
                ("user", "{query}")
            ]),
            
            "COMPARISON": ChatPromptTemplate.from_messages([
                ("system", """You compare and contrast items or concepts.
                Highlight key similarities and differences.
                Use parallel structure for comparisons.
                Example queries: "What's the difference between...?"
                Present information in a clear, organized format."""),
                ("user", "{query}")
            ]),
            
            "DEFINITION": ChatPromptTemplate.from_messages([
                ("system", """You provide clear definitions and explanations.
                Start with a concise definition.
                Follow with simple explanation and examples.
                Example queries: "Define...", "Explain..."
                Make complex concepts accessible."""),
                ("user", "{query}")
            ])
        }
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding."""
        try:
            self.logger.info(f"\n🔍 Processing message: '{message}'")
            
            # Step 1: Use router chain to classify query
            self.logger.info("📋 Classifying query type...")
            route_result = self.router_chain.invoke({"input": message})
            query_type = route_result.get("destination", "FACTUAL").upper()
            self.logger.info(f"✨ Query classified as: {query_type}")
            
            # Validate query type
            valid_types = set(self.response_prompts.keys())
            if query_type not in valid_types:
                self.logger.warning(f"⚠️ Invalid query type '{query_type}', defaulting to FACTUAL")
                query_type = "FACTUAL"
            
            # Step 2: Generate response using appropriate template
            self.logger.info(f"🤖 Generating response using {query_type} template...")
            response_prompt = self.response_prompts[query_type]
            response = response_prompt.format_messages(query=message)
            raw_response = self.llm.predict_messages(response)
            self.logger.info("✅ Response generated successfully")
            
            # Step 3: Format response
            self.logger.info("🎨 Formatting response...")
            formatted_response = self._format_response(query_type, raw_response.content)
            self.logger.info("✨ Processing complete")
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error processing your request."
    
    def _format_response(self, query_type: str, response: str) -> str:
        """Format the response based on query type."""
        self.logger.info(f"🎯 Formatting response for type: {query_type}")
        
        try:
            if query_type == "FACTUAL":
                self.logger.debug("Using factual formatting")
                # Keep only first two sentences for conciseness
                sentences = [s.strip() for s in response.split('.') if s.strip()]
                return '. '.join(sentences[:2]) + '.'
                
            elif query_type == "ANALYTICAL":
                self.logger.debug("Formatting analytical steps")
                steps = response.strip().split('\n')
                formatted_steps = []
                for i, step in enumerate(steps, 1):
                    if step.strip():
                        clean_step = step.lstrip('123456789.- ')
                        formatted_steps.append(f"{i}. {clean_step}")
                return "\n".join(formatted_steps)
                
            elif query_type == "COMPARISON":
                self.logger.debug("Formatting comparison")
                points = response.strip().split('\n')
                formatted_points = []
                for point in points:
                    if point.strip():
                        # Remove existing bullets/numbers and add consistent formatting
                        clean_point = point.lstrip('123456789.-• ')
                        formatted_points.append(f"• {clean_point}")
                return "\n".join(formatted_points)
                
            elif query_type == "DEFINITION":
                self.logger.debug("Formatting definition")
                parts = response.strip().split('\n\n')
                if len(parts) == 1:
                    # If no clear sections, add structure
                    return f"Definition: {parts[0].strip()}"
                return response.strip()
            
            self.logger.warning(f"⚠️ Unknown query type '{query_type}', using default formatting")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Error in response formatting: {str(e)}", exc_info=True)
            return response.strip()