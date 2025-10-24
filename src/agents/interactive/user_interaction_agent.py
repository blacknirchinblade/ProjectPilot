
"""
User Interaction Agent - Handle Mid-Project Changes and Feedback (Streamlit Compatible)

This agent:
- Is instantiated by the Streamlit chat UI.
- Receives user chat messages via `handle_user_request`.
- Detects user intent (Question, Bug Fix, or Change Request).
- Delegates to the appropriate specialist agent (ErrorFixingAgent, ContextualChangeAgent).
- Formats the specialist's response back to the user.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
import traceback
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

# Core LLM and Agent imports
from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.agents.interactive.conversation_manager import Conversation

# Specialist agents for delegation
from src.agents.interactive.error_fixing_agent import ErrorFixingAgent
from src.agents.interactive.contextual_change_agent import ContextualChangeAgent
from src.agents.interactive.modification_agent import InteractiveModificationAgent
import enum

class ApprovalResult(enum.Enum):
    """Approval decision results."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFY = "modify"



# Dataclasses (still useful for internal logic)
class ChangeType(Enum):
    """Types of change requests."""
    MODIFY_CODE = "modify_code"
    ADD_FEATURE = "add_feature"
    REMOVE_FEATURE = "remove_feature"
    FIX_BUG = "fix_bug"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    ADD_TEST = "add_test"
    UPDATE_DOCS = "update_docs"

@dataclass
class ChangeRequest:
    """Represents a user change request."""
    change_type: ChangeType
    description: str
    target_file: Optional[Path] = None
    target_function: Optional[str] = None
    priority: str = "medium"

@dataclass
class UserFeedback:
    """User feedback on generated code."""
    file_path: Path
    feedback_text: str
    rating: Optional[int] = None  # 1-5 scale
    specific_issues: List[str] = None
    suggestions: List[str] = None

@dataclass
class ApprovalRequest:
    """Request for user approval."""
    component_name: str
    component_type: str  # file, function, class, etc.
    preview: str
    reason: str  # Why approval is needed


class UserInteractionAgent(BaseAgent): # <-- **** FIX 1: INHERIT FROM BASEAGENT ****
    """
    Handles user interactions during code generation.
    Enables feedback, changes, and iterative refinement.
    """
    
    def __init__(
        self,
        llm_client: GeminiClient,
        prompt_manager: PromptManager,
        project_path: Path,
        conversation: Conversation
    ):
        """
        Initialize the user interaction agent for a Streamlit chat session.
        
        Args:
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
            project_path: The path to the generated project directory.
            conversation: The active Conversation object for this chat.
        """
        # --- **** FIX 2: CALL SUPER() WITH LLM_CLIENT AND PROMPT_MANAGER **** ---
        super().__init__(
            name="user_interaction_agent",
            role="User-facing Chat Coordinator",
            agent_type="chat", # Use 'chat' temperature settings
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.project_path = project_path
        self.conversation = conversation
        
        # --- FIX 3: CORRECTLY INITIALIZE AND INJECT DEPENDENCIES ---
        # 1. Create ONE modification agent
        self.modification_agent = InteractiveModificationAgent(
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            project_root=str(project_path)
        )

        # 2. Initialize specialist agents FOR DELEGATION
        self.error_fixer = ErrorFixingAgent(
            project_root=str(project_path),
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            modification_agent=self.modification_agent # <-- Pass it in
        )
        self.contextual_changer = ContextualChangeAgent(
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            project_root=str(project_path),
            modification_agent=self.modification_agent # <-- Pass it in
        )
        # --- END FIX ---
        
        logger.info(f"Initialized UserInteractionAgent for project: {project_path}")

    async def execute_task(self, task: Dict) -> str:
        """
        Main entry point for the agent, fulfilling the BaseAgent contract.
        """
        user_request = task.get("user_request")
        if not user_request:
            return "Error: No user_request provided in the task."
        return await self.handle_user_request(user_request)

    async def handle_user_request(self, prompt: str) -> str:
        """
        Main entry point called by the Streamlit chat UI.
        
        Analyzes the user's prompt and delegates to the correct workflow.
        
        Args:
            prompt: The user's chat message.
            
        Returns:
            A formatted string response for the chat UI.
        """
        logger.info(f"Handling user request: {prompt[:100]}...")
        
        try:
            # Step 1: Detect Intent
            intent_analysis = await self._detect_intent(prompt)
            intent = intent_analysis.get("intent", "question")
            
            logger.info(f"Detected intent: {intent}")
            
            # Step 2: Delegate based on intent
            if intent == "bug_report":
                error_message = intent_analysis.get("error_message", prompt)
                return await self._run_error_fixing_workflow(error_message, prompt)
                
            elif intent == "change_request":
                change_description = intent_analysis.get("change_description", prompt)
                return await self._run_change_request_workflow(change_description)
                
            else: # Default to "question"
                return await self._run_question_answering_workflow(prompt)
                
        except Exception as e:
            logger.error(f"Failed to handle user request: {e}\n{traceback.format_exc()}")
            return f"Sorry, I encountered an error while processing your request: {e}"

    async def _detect_intent(self, prompt: str) -> Dict[str, str]:
        """
        Uses the LLM to classify the user's intent.
        """
        # We can get the system prompt from the prompt_manager
        # NOTE: This assumes you have a 'chat_prompts.yaml' with these keys
        try:
            system_prompt = self.get_prompt("chat_prompts", "intent_detection_system")
        except ValueError:
            logger.warning("No 'intent_detection_system' prompt found, using fallback.")
            system_prompt = "You are an intent classification system."

        # Get the last 5 messages as history
        history = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in self.conversation.messages[-5:]]
        )
        
        try:
            user_prompt = self.get_prompt(
                "chat_prompts",
                "intent_detection_user",
                {
                    "conversation_history": history,
                    "user_request": prompt
                }
            )
        except ValueError:
             logger.warning("No 'intent_detection_user' prompt found, using fallback.")
             user_prompt = f"History:\n{history}\n\nRequest: {prompt}\n\nClassify as JSON: {{\"intent\": \"question|bug_report|change_request\", ...}}"
        
        try:
            response = await self.generate_response(
                prompt=user_prompt,
                system_instruction=system_prompt
            )
            
            # Parse the JSON response
            intent_json = self._extract_json(response)
            if intent_json:
                return intent_json
                
            logger.warning("Could not parse intent JSON, defaulting to 'question'")
            return {"intent": "question"}
            
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return {"intent": "question"} # Default to question on error

    async def _run_error_fixing_workflow(self, error_message: str, original_prompt: str) -> str:
        """
        Orchestrates the ErrorFixingAgent when a bug is reported.
        """
        logger.info("Running Error Fixing Workflow...")
        
        try:
            fix_result = await self.error_fixer.fix_error(
                error_message=error_message,
                error_context={"user_report": original_prompt},
                auto_apply=True # Apply fixes automatically in this workflow
            )
            
            if fix_result["status"] == "success":
                fixes_applied = fix_result.get("applied_fixes", [])
                if fixes_applied:
                    response = f"I've analyzed the error and applied {len(fixes_applied)} fix(es):\n\n"
                    for i, fix in enumerate(fixes_applied, 1):
                        fix_details = fix.get('fix', {})
                        response += f"**Fix {i}: {fix_details.get('description')}**\n"
                        response += f"* **File:** `{fix_details.get('filepath')}`\n"
                        response += f"* **Line:** `{fix_details.get('line_number')}`\n"
                        response += f"* **Reason:** {fix_details.get('reasoning')}\n\n"
                    response += "Please try running your code again."
                else:
                    response = "I analyzed the error but couldn't find a fix to apply. Here's my analysis:\n\n"
                    response += fix_result["error_analysis"].get("root_cause", "No root cause found.")
                
                return response
            else:
                return f"I tried to fix the error but encountered a problem: {fix_result.get('error')}"
                
        except Exception as e:
            logger.error(f"Error fixing workflow failed: {e}\n{traceback.format_exc()}")
            return f"Sorry, the error fixing agent failed: {e}"

    async def _run_change_request_workflow(self, change_description: str) -> str:
        """
        Orchestrates the ContextualChangeAgent when a change is requested.
        """
        logger.info("Running Change Request Workflow...")
        
        try:
            change_result = await self.contextual_changer.understand_and_plan_changes(
                user_request=change_description,
                auto_apply=True # Apply changes automatically
            )
            
            if change_result["status"] == "success":
                changes_applied = change_result.get("applied_changes", [])
                if changes_applied:
                    response = f"I've processed your change request and applied {len(changes_applied)} modification(s):\n\n"
                    for i, change in enumerate(changes_applied, 1):
                        change_details = change.get('change', {})
                        response += f"**Change {i}: {change_details.get('description')}**\n"
                        response += f"* **File:** `{change_details.get('filepath')}`\n"
                        response += f"* **Change:** Replaced `{change_details.get('old_code', '...').strip()}` with `{change_details.get('new_code', '...').strip()}`\n\n"
                    response += "The project code has been updated."
                else:
                    response = "I understood your request, but I didn't find any code to modify. Here is my analysis:\n\n"
                    response += f"**Intent:** {change_result['understood_intent'].get('action')}\n"
                    response += f"**Target:** {change_result['understood_intent'].get('target')}\n"
                    response += f"**Value:** {change_result['understood_intent'].get('value')}\n"
                
                return response
            else:
                return f"I tried to make the change but encountered a problem: {change_result.get('message')}"
                
        except Exception as e:
            logger.error(f"Change request workflow failed: {e}\n{traceback.format_exc()}")
            return f"Sorry, the change request agent failed: {e}"

    async def _run_question_answering_workflow(self, prompt: str) -> str:
        """
        Handles general questions about the project.
        """
        logger.info("Running Question Answering Workflow...")
        
        # Get system prompt
        try:
            system_prompt = self.get_prompt("chat_prompts", "qa_system")
        except ValueError:
            logger.warning("No 'qa_system' prompt found, using fallback.")
            system_prompt = "You are a helpful AI assistant."
        
        # Get conversation history
        history = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in self.conversation.messages[-10:]] # Get last 10 messages
        )
        
        # Get project context
        project_context = self.conversation.context
        
        try:
            user_prompt = self.get_prompt(
                "chat_prompts",
                "qa_user",
                {
                    "conversation_history": history,
                    "project_context": json.dumps(project_context, indent=2, default=str),
                    "user_question": prompt
                }
            )
        except ValueError:
            logger.warning("No 'qa_user' prompt found, using fallback.")
            user_prompt = f"History:\n{history}\n\nContext:\n{json.dumps(project_context, indent=2, default=str)}\n\nQuestion: {prompt}"

        
        try:
            response = await self.generate_response(
                prompt=user_prompt,
                system_instruction=system_prompt
            )
            return response
            
        except Exception as e:
            logger.error(f"Question answering workflow failed: {e}\n{traceback.format_exc()}")
            return f"Sorry, I encountered an error answering your question: {e}"

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Safely extracts a JSON object from a string, even if it's wrapped in markdown.
        """
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to find the first '{' and last '}'
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
            else:
                logger.warning("No JSON object found in response.")
                return None
                
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON: {e}")
            logger.debug(f"JSON string that failed: {json_str}")
            return None

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        import re
        
        # Try to find code blocks
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Try without language specifier
        code_pattern = r'```\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Return as-is if no code blocks found
        return response.strip()

# --- REMOVED ALL CLI-SPECIFIC METHODS ---
# (handle_change_request, collect_feedback, refine_based_on_feedback, etc.)