"""
Gemini 2.5 Pro Client with temperature presets for different agent types
Using LangChain's ChatGoogleGenerativeAI for better integration

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
from typing import Optional, Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """
    Gemini 2.5 Pro client with different temperature settings for different agent types
    """
    
    # Temperature presets for different agent types (WEEK 2: Further optimized)
    TEMPERATURE_PRESETS = {
        "planning": 0.7,          # Higher creativity for architecture
        "architecture": 0.6,      # Moderate for design decisions
        "coding": 0.4,            # WEEK 2: Lowered from 0.5 (eliminate incomplete patterns)
        "code_generation": 0.3,   # WEEK 2: Lowered from 0.4 (maximum precision)
        "code_refactoring": 0.3,  # Minimal for safety
        "review": 0.2,            # Very precise for reviews
        "refactoring": 0.3,       # Precise for refactoring
        "documentation": 0.6,     # More creative for docs
        "chat": 0.7,              # High creativity for interaction
        "debugging": 0.3,         # Precise for bug finding
        "testing": 0.4,           # Balanced for test generation
        "general": 0.6,           # Default fallback
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.model_name = "gemini-2.5-pro"
        os.environ["GOOGLE_API_KEY"] = self.api_key  # Ensure it's set for langchain
        logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def _get_generation_config(
        self, 
        agent_type: str = "coding",
        temperature: Optional[float] = None,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> Dict[str, Any]:
        """Get generation config based on agent type"""
        temp = temperature if temperature is not None else self.TEMPERATURE_PRESETS.get(agent_type, 0.5)
        
        return {
            "temperature": temp,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_async(
        self,
        prompt: str,
        agent_type: str = "coding",
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response asynchronously with retry logic
        
        Args:
            prompt: User prompt
            agent_type: Type of agent (affects temperature)
            temperature: Override temperature
            system_instruction: System instruction for the model
            **kwargs: Additional generation config
        
        Returns:
            Generated text response
        """
        try:
            generation_config = self._get_generation_config(
                agent_type=agent_type,
                temperature=temperature,
                **kwargs
            )
            
            # Create LangChain ChatGoogleGenerativeAI model
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=generation_config["temperature"],
                max_output_tokens=generation_config["max_output_tokens"],
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"]
            )
            
            # Combine system instruction and prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
                logger.debug("System instruction added to prompt")
            
            logger.debug(f"Generating with agent_type={agent_type}, temp={generation_config['temperature']}")
            logger.debug(f"Prompt length: {len(full_prompt)} chars")
            
            response = await llm.ainvoke(full_prompt)
            
            logger.debug(f"Response object type: {type(response)}")
            logger.debug(f"Response content type: {type(response.content)}")
            
            # Ensure content is a string (LangChain may return list in some cases)
            content = response.content
            if isinstance(content, list):
                # Join list items into a single string
                content = "\n".join(str(item) for item in content)
                logger.debug(f"Content was list, converted to string. Length: {len(content)}")
            else:
                logger.debug(f"Content is string. Length: {len(content) if content else 0}")
            
            if not content or not content.strip():
                logger.error("⚠️ LLM returned empty content!")
                logger.error(f"Response object: {response}")
                return ""
            
            logger.debug(f"Content preview (first 200 chars): {content[:200]}")
            return content
        
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
    
    def generate_sync(
        self,
        prompt: str,
        agent_type: str = "coding",
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synchronous version of generate"""
        try:
            generation_config = self._get_generation_config(
                agent_type=agent_type,
                temperature=temperature,
                **kwargs
            )
            
            # Create LangChain ChatGoogleGenerativeAI model
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=generation_config["temperature"],
                max_output_tokens=generation_config["max_output_tokens"],
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"]
            )
            
            # Combine system instruction and prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            
            response = llm.invoke(full_prompt)
            
            # Ensure content is a string (LangChain may return list in some cases)
            content = response.content
            if isinstance(content, list):
                # Join list items into a single string
                content = "\n".join(str(item) for item in content)
            
            return content
        
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
    
    async def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        agent_type: str = "coding",
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate with conversation context
        
        Args:
            messages: List of {"role": "user/model", "parts": "content"}
            agent_type: Type of agent
            temperature: Override temperature
            system_instruction: System instruction
        
        Returns:
            Generated response
        """
        try:
            generation_config = self._get_generation_config(
                agent_type=agent_type,
                temperature=temperature,
                **kwargs
            )
            
            # Create LangChain ChatGoogleGenerativeAI model
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=generation_config["temperature"],
                max_output_tokens=generation_config["max_output_tokens"],
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"]
            )
            
            # Convert messages to LangChain format
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            langchain_messages = []
            if system_instruction:
                langchain_messages.append(SystemMessage(content=system_instruction))
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("parts", "")
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                else:
                    langchain_messages.append(AIMessage(content=content))
            
            response = await llm.ainvoke(langchain_messages)
            
            # Ensure content is a string (LangChain may return list in some cases)
            content = response.content
            if isinstance(content, list):
                # Join list items into a single string
                content = "\n".join(str(item) for item in content)
            
            return content
        
        except Exception as e:
            logger.error(f"Error generating with context: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate using LangChain)"""
        llm = ChatGoogleGenerativeAI(model=self.model_name)
        # LangChain's get_num_tokens
        return llm.get_num_tokens(text)
    
    async def stream_generate(
        self,
        prompt: str,
        agent_type: str = "coding",
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Stream generation for large outputs (prevents timeouts).
        
        This method streams the response from Gemini, yielding chunks as they arrive.
        No timeout limits - perfect for generating large files (500+ lines).
        
        Args:
            prompt: User prompt
            agent_type: Type of agent (affects temperature)
            temperature: Override temperature
            system_instruction: System instruction for the model
            **kwargs: Additional generation config
        
        Returns:
            Complete generated text (assembled from stream)
        """
        try:
            generation_config = self._get_generation_config(
                agent_type=agent_type,
                temperature=temperature,
                **kwargs
            )
            
            # Create LangChain ChatGoogleGenerativeAI model
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=generation_config["temperature"],
                max_output_tokens=generation_config["max_output_tokens"],
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"],
                streaming=True  # Enable streaming
            )
            
            logger.debug(f"Streaming with agent_type={agent_type}, temp={generation_config['temperature']}")
            
            # Combine system instruction and prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            
            # Stream the response
            full_response = ""
            chunk_count = 0
            
            async for chunk in llm.astream(full_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    chunk_count += 1
                    
                    # Log progress every 10 chunks
                    if chunk_count % 10 == 0:
                        logger.debug(f"Streamed {chunk_count} chunks, {len(full_response)} chars so far...")
            
            logger.success(f"Streaming complete: {chunk_count} chunks, {len(full_response)} chars total")
            return full_response
        
        except Exception as e:
            logger.error(f"Error streaming content: {str(e)}")
            raise
    
    async def stream_generate_with_callback(
        self,
        prompt: str,
        callback,
        agent_type: str = "coding",
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Stream generation with callback for real-time updates.
        
        Useful for showing progress to users during long generations.
        
        Args:
            prompt: User prompt
            callback: Function called with each chunk: callback(chunk_text, total_so_far)
            agent_type: Type of agent
            temperature: Override temperature
            system_instruction: System instruction
            **kwargs: Additional generation config
        
        Returns:
            Complete generated text
        """
        try:
            generation_config = self._get_generation_config(
                agent_type=agent_type,
                temperature=temperature,
                **kwargs
            )
            
            # Create LangChain ChatGoogleGenerativeAI model
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=generation_config["temperature"],
                max_output_tokens=generation_config["max_output_tokens"],
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"],
                streaming=True
            )
            
            # Combine system instruction and prompt
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            
            # Stream with callback
            full_response = ""
            async for chunk in llm.astream(full_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    # Call the callback with chunk and current total
                    if callback:
                        await callback(chunk.content, full_response)
            
            return full_response
        
        except Exception as e:
            logger.error(f"Error streaming with callback: {str(e)}")
            raise
