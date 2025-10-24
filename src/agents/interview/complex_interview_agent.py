"""
Complex Interview Questions Agent

This agent generates in-depth, multi-part interview questions for various
technical roles and topics.

Workflow:
1. Receives a job role, technical topic, and difficulty level.
2. Generates a list of complex, well-structured interview questions.
3. For each question, it also provides:
    - Key points for the expected answer.
    - Potential follow-up questions.
    - A rubric for evaluating the candidate's response.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager

from typing import Dict, Any, List, Optional
from loguru import logger
import json
import re

class ComplexInterviewAgent(BaseAgent):
    """
    Generates complex, in-depth interview questions.
    """

    def __init__(
        self, 
        llm_client: Optional[GeminiClient] = None, 
        prompt_manager: Optional[PromptManager] = None
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        super().__init__(
            name="complex_interview_agent",
            agent_type="interview",
            role="Expert Technical Interviewer and Question Designer",
            llm_client=llm_client,
            prompt_manager=prompt_manager,
        )

    async def generate_interview_questions(self, role: str, topic: str, difficulty: str, num_questions: int = 100, tech_stack: str = "general") -> List[Dict[str, Any]]:
        """
        Generates a list of complex interview questions.

        Args:
            role: The job role (e.g., "Senior Python Developer").
            topic: The technical topic (e.g., "System Design").
            difficulty: The desired difficulty level (e.g., "Expert").
            num_questions: Minimum number of questions to generate.
            tech_stack: The technology stack to focus on.

        Returns:
            A list of dictionaries, where each dictionary represents a question
            with its answer key, follow-ups, and evaluation rubric.
        """
        if not self.llm_client or not self.prompt_manager:
            logger.error("InterviewAgent not initialized with LLMClient or PromptManager.")
            return [{"question": "Error: Agent not initialized.", "category": "Error"}]
        
        prompt_data = {
            "role": role,
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "tech_stack": tech_stack
        }
        try:
            prompt = self.prompt_manager.get_prompt(
                "interview_prompts", 
                "generate_interview_questions", 
                prompt_data
            )
        except ValueError as e:
            logger.error(f"Error getting prompt: {e}. Using fallback.")
            prompt = f"Generate {num_questions} interview questions for {role} on {topic}. Difficulty: {difficulty}. Format as JSON."
        # --- END FIX ---
        response_text = await self.generate_response(prompt)
        
        try:
            # First attempt to parse the JSON
            return self._parse_json_response(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}. Attempting to repair.")
            # If parsing fails, try to repair the JSON using the LLM
            repaired_json_str = await self._repair_json(response_text)
            try:
                return self._parse_json_response(repaired_json_str)
            except json.JSONDecodeError as final_e:
                logger.error(f"Failed to parse JSON even after repair: {final_e}")
                logger.debug(f"Original Response: {response_text}")
                logger.debug(f"Repaired Response: {repaired_json_str}")
                return [{
                    "question": "Error: Could not parse the generated questions from LLM response.",
                    "category": "Error",
                    "difficulty": "N/A",
                    "answer_key": [f"LLM Error: {final_e}", f"Response: {response_text[:100]}..."],
                    "follow_ups": []
                }]

    def _parse_json_response(self, json_str: str) -> List[Dict[str, Any]]:
        """Extracts and parses JSON from a string."""
        # Use a more robust regex to find the JSON array
        json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if json_match:
            json_data = json_match.group(0)
            return json.loads(json_data)
        else:
            # Fallback for malformed start/end
            if not json_str.strip().startswith('['):
                json_str = '[' + json_str
            if not json_str.strip().endswith(']'):
                json_str = json_str + ']'
            return json.loads(json_str)

    async def _repair_json(self, broken_json: str) -> str:
        """Asks the LLM to repair a broken JSON string."""
        logger.info("Attempting to repair JSON with LLM...")
        prompt = self.prompt_manager.get_prompt(
            "interview_prompts",
            "repair_json",
            {"broken_json": broken_json}
        )
        repaired_response = await self.generate_response(prompt)
        return repaired_response

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a task to generate interview questions.
        This is the implementation of the abstract method from BaseAgent.
        """
        role = task.get("role", "Software Engineer")
        topic = task.get("topic", "Python")
        difficulty = task.get("difficulty", "Intermediate")
        num_questions = task.get("num_questions", 100)
        
        # For now, we'll just call the async method.
        # A more robust implementation might handle different task types.
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        questions = loop.run_until_complete(
            self.generate_interview_questions(role, topic, difficulty, num_questions)
        )
        return {"status": "success", "questions": questions}
