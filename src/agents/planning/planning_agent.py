"""
Planning Agent - Handles project planning, requirement analysis, and task breakdown

This agent is responsible for:
- Analyzing project requirements
- Asking clarifying questions
- Breaking down projects into tasks
- Designing system architecture
- Identifying dependencies

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class PlanningAgent(BaseAgent):
    """
    Planning Agent for project analysis and task decomposition
    
    Uses low temperature (0.3) for precise, methodical planning
    """
    
    def __init__(
        self,
        name: str = "planning_agent",
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """Initialize Planning Agent"""
        super().__init__(
            name=name,
            role="Project Planner and Architect",
            agent_type="planning",  # Uses temperature 0.3
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        logger.info(f"{self.name} ready for planning tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a planning task
        
        Args:
            task: Task dictionary with:
                - task_type: Type of planning task
                - data: Task-specific data
        
        Returns:
            Result dictionary with planning output
        """
        task_type = task.get("task_type")
        
        if task_type == "analyze_requirements":
            return await self.analyze_requirements(task.get("data", {}))
        elif task_type == "ask_clarifications":
            return await self.generate_clarifying_questions(task.get("data", {}))
        elif task_type == "design_architecture":
            return await self.design_architecture(task.get("data", {}))
        elif task_type == "breakdown_tasks":
            return await self.breakdown_tasks(task.get("data", {}))
        elif task_type == "analyze_dependencies":
            return await self.analyze_dependencies(task.get("data", {}))
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    async def analyze_requirements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze project requirements
        
        Args:
            data: Dictionary with 'project_description'
        
        Returns:
            Structured requirement analysis
        """
        try:
            project_description = data.get("project_description", "")
            
            if not project_description:
                return {
                    "status": "error",
                    "message": "No project description provided"
                }
            
            # Get prompt template
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="requirement_analysis",
                variables={"project_description": project_description}
            )
            
            logger.info(f"{self.name} analyzing requirements")
            
            # Generate analysis
            response = await self.generate_response(prompt)
            
            return {
                "status": "success",
                "task": "requirement_analysis",
                "input": project_description,
                "output": response
            }
        
        except Exception as e:
            logger.error(f"Error in requirement analysis: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_clarifying_questions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate clarifying questions for ambiguous requirements
        
        Args:
            data: Dictionary with 'project_description'
        
        Returns:
            List of clarifying questions
        """
        try:
            project_description = data.get("project_description", "")
            
            if not project_description:
                return {
                    "status": "error",
                    "message": "No project description provided"
                }
            
            # Get prompt template
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="clarification_questions",
                variables={"project_description": project_description}
            )
            
            logger.info(f"{self.name} generating clarifying questions")
            
            # Generate questions
            response = await self.generate_response(prompt)
            
            return {
                "status": "success",
                "task": "clarification_questions",
                "input": project_description,
                "output": response,
                "questions": self._parse_questions(response)
            }
        
        except Exception as e:
            logger.error(f"Error generating clarifying questions: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def design_dynamic_architecture(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design system architecture dynamically based on specifications
        (NEW: LLM decides structure, not hardcoded)
        
        Args:
            specification: Enhanced specification from PromptEngineerAgent
        
        Returns:
            {
                "status": "success",
                "architecture": {
                    "structure_type": "modular|monolithic|microservices",
                    "files": [
                        {
                            "path": "src/data/dataset.py",
                            "purpose": "Data loading and preprocessing",
                            "estimated_lines": 150,
                            "classes": ["MNISTDataset", "DataTransform"],
                            "functions": ["load_data", "preprocess"],
                            "dependencies": ["torch", "torchvision"]
                        },
                        ...
                    ],
                    "config_files": [...],
                    "total_estimated_files": 8,
                    "total_estimated_lines": 1500
                }
            }
        """
        try:
            import json
            
            logger.info(f"{self.name} designing dynamic architecture")
            
            # Get prompt for dynamic architecture design
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="dynamic_architecture_design",
                variables={
                    "specification": json.dumps(specification, indent=2)
                }
            )
            
            # Generate architecture with LLM
            response = await self.generate_response(prompt)
            
            # Extract JSON from response (handle markdown wrapping)
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]  # Remove ```json
            if json_str.startswith("```"):
                json_str = json_str[3:]  # Remove ```
            if json_str.endswith("```"):
                json_str = json_str[:-3]  # Remove trailing ```
            json_str = json_str.strip()
            
            # Parse JSON response
            try:
                architecture = json.loads(json_str)
                logger.info(f"Designed architecture with {len(architecture.get('files', []))} files")
                
                return architecture  # Return architecture directly, not wrapped
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse architecture JSON: {e}")
                logger.debug(f"Raw response: {response[:200]}...")
                # Fallback to minimal structure
                return self._get_fallback_architecture(specification)
                
        except Exception as e:
            logger.error(f"Error designing dynamic architecture: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _get_fallback_architecture(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback architecture if LLM fails"""
        return {
            "status": "success",
            "architecture": {
                "structure_type": "modular",
                "files": [
                    {
                        "path": "src/main.py",
                        "purpose": "Main entry point",
                        "estimated_lines": 100,
                        "classes": [],
                        "functions": ["main"],
                        "dependencies": []
                    }
                ],
                "config_files": [
                    {"path": "config.yaml", "purpose": "Configuration"},
                    {"path": "requirements.txt", "purpose": "Dependencies"}
                ],
                "total_estimated_files": 3,
                "total_estimated_lines": 200
            }
        }
    
    async def optimize_module_structure(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize module structure for maintainability and scalability
        
        Args:
            architecture: Initial architecture design
        
        Returns:
            Optimized architecture with improvements
        """
        try:
            import json
            
            logger.info(f"{self.name} optimizing module structure")
            
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="optimize_module_structure",
                variables={
                    "architecture": json.dumps(architecture, indent=2)
                }
            )
            
            response = await self.generate_response(prompt)
            
            try:
                optimized = json.loads(response)
                logger.info("Architecture optimized successfully")
                return {
                    "status": "success",
                    "optimized_architecture": optimized,
                    "improvements": optimized.get("improvements", [])
                }
            except json.JSONDecodeError:
                return {
                    "status": "partial",
                    "message": "Could not parse optimization, returning original",
                    "optimized_architecture": architecture
                }
                
        except Exception as e:
            logger.error(f"Error optimizing module structure: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "optimized_architecture": architecture
            }
    
    async def design_architecture(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design system architecture (LEGACY METHOD - keeping for backward compatibility)
        
        Args:
            data: Dictionary with 'project_type' and 'requirements'
        
        Returns:
            Architecture design
        """
        try:
            project_type = data.get("project_type", "")
            requirements = data.get("requirements", "")
            
            if not project_type or not requirements:
                return {
                    "status": "error",
                    "message": "Missing project_type or requirements"
                }
            
            # Get prompt template
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="architecture_design",
                variables={
                    "project_type": project_type,
                    "requirements": requirements
                }
            )
            
            logger.info(f"{self.name} designing architecture")
            
            # Generate architecture
            response = await self.generate_response(prompt)
            
            return {
                "status": "success",
                "task": "architecture_design",
                "input": {
                    "project_type": project_type,
                    "requirements": requirements
                },
                "output": response
            }
        
        except Exception as e:
            logger.error(f"Error designing architecture: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def breakdown_tasks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down project into specific tasks
        
        Args:
            data: Dictionary with 'project_name' and 'architecture'
        
        Returns:
            Task breakdown
        """
        try:
            project_name = data.get("project_name", "")
            architecture = data.get("architecture", "")
            
            if not project_name or not architecture:
                return {
                    "status": "error",
                    "message": "Missing project_name or architecture"
                }
            
            # Get prompt template
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="task_breakdown",
                variables={
                    "project_name": project_name,
                    "architecture": architecture
                }
            )
            
            logger.info(f"{self.name} breaking down tasks")
            
            # Generate task breakdown
            response = await self.generate_response(prompt)
            
            return {
                "status": "success",
                "task": "task_breakdown",
                "input": {
                    "project_name": project_name,
                    "architecture": architecture
                },
                "output": response,
                "tasks": self._parse_tasks(response)
            }
        
        except Exception as e:
            logger.error(f"Error breaking down tasks: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def analyze_dependencies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dependencies between components
        
        Args:
            data: Dictionary with 'components'
        
        Returns:
            Dependency analysis
        """
        try:
            components = data.get("components", "")
            
            if not components:
                return {
                    "status": "error",
                    "message": "No components provided"
                }
            
            # Get prompt template
            prompt = self.get_prompt(
                category="planning_prompts",
                prompt_name="dependency_analysis",
                variables={"components": components}
            )
            
            logger.info(f"{self.name} analyzing dependencies")
            
            # Generate dependency analysis
            response = await self.generate_response(prompt)
            
            return {
                "status": "success",
                "task": "dependency_analysis",
                "input": components,
                "output": response
            }
        
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from response text"""
        questions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions or lines ending with ?
            if line and (line[0].isdigit() or line.endswith('?')):
                # Remove numbering like "1. " or "1) "
                question = line.lstrip('0123456789.)- ').strip()
                if question:
                    questions.append(question)
        
        return questions
    
    def _parse_tasks(self, response: str) -> List[Dict[str, Any]]:
        """Parse tasks from response text (basic parsing)"""
        tasks = []
        lines = response.split('\n')
        
        current_task = None
        for line in lines:
            line = line.strip()
            # Simple task detection (can be improved)
            if line.startswith(('Task ', 'TASK ', '- Task', '* Task')):
                if current_task:
                    tasks.append(current_task)
                current_task = {"description": line}
        
        if current_task:
            tasks.append(current_task)
        
        return tasks
