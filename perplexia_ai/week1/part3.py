"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.memory import ConversationBufferMemory
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools import available_tools
import logging

class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""
    
    # Class-level chat history that persists between instances
    _shared_history = []
    _max_history_length = 10
    
    def __init__(self):
        """Initialize the chat interface with logging."""
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.tools = []
        self.router_chain = None
        self.response_prompts = {}
    
    @classmethod
    def set_history_limit(cls, limit: int) -> None:
        """Set the maximum number of conversation exchanges to remember."""
        if limit < 1:
            raise ValueError("History limit must be at least 1")
        cls._max_history_length = limit
        # Trim existing history if needed
        if len(cls._shared_history) > limit:
            cls._shared_history = cls._shared_history[-limit:]
    
    @classmethod
    def clear_history(cls) -> None:
        """Clear the conversation history."""
        cls._shared_history = []
    
    def initialize(self) -> None:
        """Initialize components for memory-enabled chat."""
        self.llm = ChatOpenAI()
        
        # Initialize tools
        self.tools = available_tools
        
        # Define router prompt template with tool detection and context
        router_template = """Given the following conversation history and query, determine if it requires any tools and the most appropriate category.

        Previous conversation:
        {chat_history}
        
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
        
        Current Query: {input}
        
        Format your response as a JSON object with the following keys:
        {{
            "destination": "The category (must be one of: CALCULATOR, DATETIME, FACTUAL, OPINION, PROCEDURAL, CLARIFICATION)",
            "next_inputs": "the original input query"
        }}
        
        Response:"""
        
        # Create the router prompt and chain
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input", "chat_history"]
        )
        self.router_chain = router_prompt | self.llm | RouterOutputParser()
        
        # Set up response prompts for each type with context
        self.response_prompts = {
            "CALCULATOR": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful math assistant with conversation memory.
                When given a calculation result, explain it clearly and reference previous context if relevant.
                Format: "The result of [operation] is [result]." """),
                ("human", "Previous conversation:\n{chat_history}"),
                ("human", "Current query: {query} = {result}")
            ]),
            "DATETIME": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant providing time information with context.
                Format the response naturally and reference previous queries if relevant."""),
                ("human", "Previous conversation:\n{chat_history}"),
                ("human", "Current time information: {result}")
            ]),
            "FACTUAL": ChatPromptTemplate.from_messages([
                ("system", """You provide factual information with context awareness.
                Reference previous conversation when relevant."""),
                ("human", "Previous conversation:\n{chat_history}"),
                ("human", "Current query: {query}")
            ]),
            "OPINION": ChatPromptTemplate.from_messages([
                ("system", """You provide balanced perspectives with context awareness.
                Build upon previous discussion when relevant."""),
                ("human", "Previous conversation:\n{chat_history}"),
                ("human", "Current query: {query}")
            ]),
            "PROCEDURAL": ChatPromptTemplate.from_messages([
                ("system", """You provide clear instructions with context awareness.
                Reference previous steps or related procedures when relevant."""),
                ("human", "Previous conversation:\n{chat_history}"),
                ("human", "Current query: {query}")
            ]),
            "CLARIFICATION": ChatPromptTemplate.from_messages([
                ("system", """You provide clear explanations with context awareness.
                Build upon previous explanations when relevant."""),
                ("human", "Previous conversation:\n{chat_history}"),
                ("human", "Current query: {query}")
            ])
        }
    
    def process_message(self, message: str, chat_history: Optional[List[List[str]]] = None) -> List[Dict[str, str]]:
        """Process a message with memory and tools."""
        try:
            self.logger.info(f"\n🔍 Processing message: '{message}'")
            
            # Format chat history for context
            formatted_history = ""
            if self.__class__._shared_history:
                formatted_history = "\n".join([
                    f"Human: {msg[0]}\nAssistant: {msg[1]}"
                    for msg in self.__class__._shared_history
                    if isinstance(msg, list) and len(msg) == 2
                ])
                self.logger.info(f"📜 Using shared history ({len(self.__class__._shared_history)} exchanges):\n{formatted_history}")
            else:
                self.logger.info("📜 No shared history available")
            
            # First, get context resolution from LLM
            context = self._resolve_context(message)
            resolved_message = context.get("resolved_query", message)
            suggested_type = context.get("context_type")
            reasoning = context.get("reasoning", "No context provided")
            
            self.logger.info(f"🤔 Context Resolution: {reasoning}")
            
            # Only proceed with router if we don't have a clear context type
            if not suggested_type:
                route_result = self.router_chain.invoke({
                    "input": resolved_message,
                    "chat_history": formatted_history if formatted_history else "No previous conversation."
                })
                query_type = route_result.get("destination", "FACTUAL").upper()
            else:
                query_type = suggested_type.upper()
                
            self.logger.info(f"✨ Query classified as: {query_type}")
            
            # Handle tool-based queries
            response = ""
            if query_type == "CALCULATOR":
                self.logger.info("🧮 Using calculator tool...")
                try:
                    # Handle percentage follow-ups specifically for calculator
                    if ('%' in message or 'percent' in message.lower()) and self.__class__._shared_history:
                        # Try to extract percentage and find base amount
                        import re
                        percent_match = re.search(r'(\d+)%', message)
                        if percent_match:
                            percentage = percent_match.group(1)
                            # Look for base amount in previous exchanges
                            for prev_msg in reversed(self.__class__._shared_history):
                                if '$' in prev_msg[0]:
                                    base_match = re.search(r'\$(\d+)', prev_msg[0])
                                    if base_match:
                                        base = base_match.group(1)
                                        resolved_message = f"calculate {percentage}% of {base}"
                                        self.logger.info(f"💡 Resolved percentage follow-up: {resolved_message}")
                                        break
                    
                    # Use the resolved query from context
                    if "calculate" not in resolved_message.lower():
                        resolved_message = f"calculate {resolved_message}"
                    
                    result = self.tools[0].invoke({"query": resolved_message})
                    self.logger.info(f"✅ Calculation result: {result}")
                    response = self._format_response(query_type, resolved_message, result, formatted_history)
                except Exception as calc_error:
                    self.logger.error(f"❌ Calculation error: {str(calc_error)}")
                    response = f"I apologize, but I couldn't perform that calculation. Error: {str(calc_error)}"
                    
            elif query_type == "DATETIME":
                self.logger.info("🕒 Using datetime tool...")
                try:
                    result = self.tools[1].invoke({"query": resolved_message})
                    self.logger.info(f"✅ Datetime result: {result}")
                    response = self._format_response(query_type, resolved_message, result, formatted_history)
                except Exception as time_error:
                    self.logger.error(f"❌ Datetime error: {str(time_error)}")
                    response = f"I apologize, but I couldn't process that time query. Error: {str(time_error)}"
            else:
                # Handle regular queries
                self.logger.info(f"🤖 Generating response using {query_type} template...")
                response = self._format_response(query_type, resolved_message, None, formatted_history)
                self.logger.info("✅ Response generated successfully")
            
            # Update shared chat history with the new exchange
            self.__class__._shared_history.append([message, response])
            
            # Apply LRU-like behavior - keep only the most recent exchanges
            if len(self.__class__._shared_history) > self.__class__._max_history_length:
                self.__class__._shared_history = self.__class__._shared_history[-self.__class__._max_history_length:]
                self.logger.info(f"📚 Trimmed chat history to {self.__class__._max_history_length} most recent exchanges")
            
            self.logger.info(f"📚 Updated shared history: {self.__class__._shared_history}")
            
            # Return messages in correct Gradio format
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            
        except Exception as e:
            self.logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)
            
            # Return error in correct Gradio format
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"I apologize, but I encountered an error: {str(e)}"}
            ]
    
    def _resolve_calculation_context(self, message: str) -> str:
        """Resolve context for calculations using chat history."""
        import re
        
        for prev_msg in reversed(self.chat_history):
            prev_human = prev_msg.get('human', '')
            prev_assistant = prev_msg.get('assistant', '')
            
            # Handle percentage follow-ups
            if "%" in message:
                new_percent = re.findall(r'(\d+)%', message)
                if new_percent and '$' in prev_human:
                    base_amounts = re.findall(r'\$(\d+)', prev_human)
                    if base_amounts:
                        return f"what is {new_percent[0]}% of {base_amounts[0]}"
            
            # Handle "it" or "that" references
            if any(word in message.lower() for word in ["it", "that"]):
                numbers = re.findall(r'\d+', prev_assistant)
                if numbers:
                    return message.lower().replace("it", numbers[-1]).replace("that", numbers[-1])
        
        return message
    
    def _format_response(self, query_type: str, message: str, result: Optional[str], chat_history: str) -> str:
        """Format response based on query type and context."""
        response_prompt = self.response_prompts[query_type]
        
        if query_type == "CALCULATOR":
            response = response_prompt.format_messages(
                chat_history=chat_history,
                query=message,
                result=result
            )
        elif query_type == "DATETIME":
            response = response_prompt.format_messages(
                chat_history=chat_history,
                result=result
            )
        else:
            response = response_prompt.format_messages(
                chat_history=chat_history,
                query=message
            )
        
        raw_response = self.llm.predict_messages(response)
        return raw_response.content.strip()
    
    def _resolve_context(self, message: str) -> dict:
        """Resolve context from conversation history using LLM."""
        if not self.__class__._shared_history:
            return {"resolved_query": message, "context_type": None, "reasoning": "No conversation history"}
            
        try:
            # Create a context analysis prompt
            context_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a context resolution assistant. Analyze the conversation history 
                and current query to understand the full context and type of query.
                
                Identify:
                1. If the query references previous conversation
                2. The type of query (CALCULATOR, DATETIME, FACTUAL, etc.)
                3. Any specific values, names, or references that need resolution
                
                Return a JSON object with:
                {
                    "context_type": "The query category",
                    "resolved_query": "The fully resolved query with context",
                    "reasoning": "Brief explanation of the resolution"
                }
                
                Examples:
                1. History: "my name is John" -> "Nice to meet you, John"
                   Query: "what's my name?"
                   Response: {
                     "context_type": "FACTUAL",
                     "resolved_query": "what is John's name based on previous introduction",
                     "reasoning": "User previously introduced themselves as John"
                   }
                
                2. History: "What's 15% of $120?" -> "That's $18"
                   Query: "what about 20%?"
                   Response: {
                     "context_type": "CALCULATOR",
                     "resolved_query": "calculate 20% of 120",
                     "reasoning": "Using previous base amount of $120 with new percentage"
                   }
                """),
                ("human", """Previous conversation:
                {chat_history}
                
                Current query: {query}
                
                Resolve the context and return the JSON response:""")
            ])
            
            # Format recent chat history
            formatted_history = "\n".join([
                f"Human: {msg[0]}\nAssistant: {msg[1]}"
                for msg in self.__class__._shared_history[-3:]  # Use last 3 messages for recent context
                if isinstance(msg, list) and len(msg) == 2  # Only include complete exchanges
            ])
            
            if not formatted_history:
                return {"resolved_query": message, "context_type": None, "reasoning": "No valid conversation history"}
            
            # Get context resolution from LLM
            context_messages = context_prompt.format_messages(
                chat_history=formatted_history,
                query=message
            )
            resolution = self.llm.predict_messages(context_messages).content.strip()
            
            # Parse the JSON response
            import json
            resolution_dict = json.loads(resolution)
            
            self.logger.info(f"🔄 Original query: '{message}'")
            self.logger.info(f"📜 Using history: {formatted_history}")
            self.logger.info(f"✨ Context resolution: {resolution_dict}")
            
            return resolution_dict
            
        except Exception as e:
            self.logger.warning(f"⚠️ Context resolution failed: {str(e)}")
            return {"resolved_query": message, "context_type": None, "reasoning": f"Error: {str(e)}"} 