"""
Base Agent - Foundation class for all agents in the system

Every agent inherits from this base class and gets:
- LLM client integration
- Prompt management
- Logging capabilities
- Memory access
- Communication abilities

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from loguru import logger

from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class BaseAgent(ABC):
    """
    Base class for all agents in the AutoCoder system
    
    Each agent has:
    - A unique name and role
    - Access to LLM (Gemini)
    - Access to prompts
    - Agent-specific temperature setting
    - Logging capabilities
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        agent_type: str,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize base agent
        
        Args:
            name: Unique agent name (e.g., "planning_agent_1")
            role: Agent role description
            agent_type: Type for temperature preset (planning, coding, review, etc.)
            llm_client: Gemini client instance (creates new if None)
            prompt_manager: Prompt manager instance (creates new if None)
        """
        self.name = name
        self.role = role
        self.agent_type = agent_type
        
        # Initialize LLM and prompts
        self.llm = llm_client or GeminiClient()
        self.prompt_manager = prompt_manager or PromptManager()
        
        # Agent state
        self.is_active = True
        self.task_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {self.name} with role: {role}")
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent
        
        Args:
            task: Task dictionary with details
        
        Returns:
            Result dictionary with output
        """
        pass
    
    async def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate LLM response using agent's settings
        
        Args:
            prompt: User prompt
            temperature: Override temperature (uses agent_type default if None)
            system_instruction: Override system instruction
        
        Returns:
            Generated response
        """
        try:
            # Use agent's system instruction if not provided
            if system_instruction is None:
                system_instruction = self.get_system_instruction()
            
            logger.debug(f"{self.name} generating response")
            
            response = await self.llm.generate_async(
                prompt=prompt,
                agent_type=self.agent_type,
                temperature=temperature,
                system_instruction=system_instruction
            )
            
            # Log task in history
            self.task_history.append({
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "agent_type": self.agent_type
            })
            
            return response
        
        except Exception as e:
            logger.error(f"Error in {self.name} generating response: {str(e)}")
            raise
    
    def generate_response_sync(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """Synchronous version of generate_response"""
        try:
            if system_instruction is None:
                system_instruction = self.get_system_instruction()
            
            response = self.llm.generate_sync(
                prompt=prompt,
                agent_type=self.agent_type,
                temperature=temperature,
                system_instruction=system_instruction
            )
            
            self.task_history.append({
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "agent_type": self.agent_type
            })
            
            return response
        
        except Exception as e:
            logger.error(f"Error in {self.name} generating response: {str(e)}")
            raise
    
    def get_system_instruction(self) -> str:
        """Get system instruction for this agent type"""
        return self.prompt_manager.get_system_instruction(self.agent_type)
    
    def get_prompt(
        self,
        category: str,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get a prompt template
        
        Args:
            category: Prompt category (e.g., 'planning_prompts')
            prompt_name: Name of the prompt
            variables: Variables to fill in the template
        
        Returns:
            Formatted prompt
        """
        return self.prompt_manager.get_prompt(category, prompt_name, variables)
    
    def get_temperature(self) -> float:
        """Get the temperature setting for this agent type"""
        return self.llm.TEMPERATURE_PRESETS.get(self.agent_type, 0.5)
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get agent's task history"""
        return self.task_history
    
    def clear_history(self):
        """Clear task history"""
        self.task_history = []
        logger.info(f"Cleared history for {self.name}")
    
    def deactivate(self):
        """Deactivate this agent"""
        self.is_active = False
        logger.info(f"Deactivated {self.name}")
    
    def activate(self):
        """Activate this agent"""
        self.is_active = True
        logger.info(f"Activated {self.name}")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role}) - Type: {self.agent_type}"
    
    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', role='{self.role}', agent_type='{self.agent_type}')"
