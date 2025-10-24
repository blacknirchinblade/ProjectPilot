
"""
Enhanced Advanced Documentation Agent

This agent generates a comprehensive suite of documentation for a given project,
including a detailed README, a workflow guide, a usage manual, and a data guide.

Workflow:
1. Receives the project's configuration and file structure.
2. Generates a comprehensive README.md with dynamic sections.
3. Generates a WORKFLOW.md explaining the agent pipeline.
4. Generates a USAGE.md for end-users.
5. Generates a DATA_GUIDE.md for using custom datasets.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import json
from typing import Dict, Any, List
from loguru import logger
from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager

class AdvancedDocumentationAgent(BaseAgent):
    """
    Generates a full suite of project documentation.
    """

    def __init__(self, llm_client: GeminiClient, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager    
        super().__init__(
            name="advanced_documentation_agent",
            role="Technical Writer and Documentation Specialist",
            agent_type="documentation",
            llm_client=llm_client,
            prompt_manager=prompt_manager,
        )

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a task for the AdvancedDocumentationAgent.
        """
        if task_data.get("task_type") == "generate_docs":
            docs = await self.generate_documentation_suite(
                task_data.get("project_config"),
                task_data.get("file_structure")
            )
            return {"status": "success", "docs": docs}
        else:
            return {"status": "error", "message": "Unknown task type"}

    async def generate_documentation_suite(self, project_config: Dict[str, Any], file_structure: List[str]) -> Dict[str, str]:
        """
        Generates a complete suite of documentation files.

        Args:
            project_config: The configuration of the generated project.
            file_structure: A list of file paths in the project.

        Returns:
            A dictionary where keys are filenames (e.g., "README.md")
            and values are the file content.
        """
        
        docs = {}

        try:
            # Generate each document
            readme_content = await self._generate_readme(project_config, file_structure)
            workflow_content = await self._generate_workflow_doc(project_config)
            usage_content = await self._generate_usage_guide(project_config)
            data_guide_content = await self._generate_data_guide(project_config)

            docs["README.md"] = readme_content
            docs["WORKFLOW.md"] = workflow_content
            docs["USAGE.md"] = usage_content
            docs["DATA_GUIDE.md"] = data_guide_content
            
        except Exception as e:
            logger.error(f"Error generating documentation suite: {e}")
            # Provide fallback documentation
            docs["README.md"] = self._create_fallback_readme(project_config, file_structure)
            docs["WORKFLOW.md"] = "# Workflow Documentation\n\nDocumentation generation failed. Please check the logs."
        
        return docs

    async def _generate_readme(self, config: Dict[str, Any], file_structure: List[str]) -> str:
        """Generate comprehensive README.md"""
        prompt_data = {
            "project_config": config, 
            "file_structure": file_structure,
            "project_name": config.get("project_name", "Project"),
            "project_description": config.get("project_description", ""),
            "project_type": config.get("project_type", "Software Project")
        }
        try:
            prompt = self.prompt_manager.get_prompt("documentation_prompts", "generate_readme", prompt_data)
            readme = await self.llm_client.generate_async(prompt, agent_type="documentation")
            return readme
        except Exception as e:
            logger.error(f"Error generating README: {e}")
            return self._create_fallback_readme(config, file_structure)

    async def _generate_workflow_doc(self, config: Dict[str, Any]) -> str:
        """Generate WORKFLOW.md"""
        prompt_data = {
            "config": config,
            "project_name": config.get("project_name", "Project")
        }
        try:
            prompt = self.prompt_manager.get_prompt("documentation_prompts", "generate_workflow_doc", prompt_data)
            workflow = await self.llm_client.generate_async(prompt, agent_type="documentation")
            return workflow
        except Exception as e:
            logger.error(f"Error generating workflow doc: {e}")
            return self._create_fallback_workflow_doc(config)

    async def _generate_usage_guide(self, config: Dict[str, Any]) -> str:
        """Generate USAGE.md"""
        prompt_data = {
            "config": config,
            "project_name": config.get("project_name", "Project")
        }
        try:
            prompt = self.prompt_manager.get_prompt("documentation_prompts", "generate_usage_guide", prompt_data)
            usage = await self.llm_client.generate_async(prompt, agent_type="documentation")
            return usage
        except Exception as e:
            logger.error(f"Error generating usage guide: {e}")
            return self._create_fallback_usage_guide(config)

    async def _generate_data_guide(self, config: Dict[str, Any]) -> str:
        """Generate DATA_GUIDE.md"""
        prompt_data = {
            "config": config,
            "project_name": config.get("project_name", "Project")
        }
        try:
            prompt = self.prompt_manager.get_prompt("documentation_prompts", "generate_data_guide", prompt_data)
            data_guide = await self.llm_client.generate_async(prompt, agent_type="documentation")
            return data_guide
        except Exception as e:
            logger.error(f"Error generating data guide: {e}")
            return self._create_fallback_data_guide(config)

    def _create_fallback_readme(self, config: Dict[str, Any], file_structure: List[str]) -> str:
        """Create a fallback README when generation fails"""
        project_name = config.get("project_name", "Project")
        return f"""# {project_name}

        > Auto-generated by ProjectPilot AI

        ## Description
        {config.get('project_description', 'A project generated by ProjectPilot AI.')}

        ## Project Structure
        
## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py

Features
AI-powered functionality

Automated project generation

Production-ready codebase

License
This project is auto-generated by Ganesh Islavath.

"""
    
    def _create_fallback_workflow_doc(self, config: Dict[str, Any]) -> str:
        """Create a fallback WORKFLOW.md when generation fails"""
        project_name = config.get("project_name", "Project")
        return f"""# Workflow Documentation for {project_name}

        Agent Pipeline
Planning Phase - Project analysis and architecture design

Implementation Phase - Code generation and component development

Documentation Phase - Comprehensive documentation generation

Testing Phase - Quality assurance and validation

Development Process
AI-driven code generation

Iterative refinement

Quality validation

Documentation automation
"""
    def _create_fallback_usage_guide(self, config: Dict[str, Any]) -> str:
        """Create a fallback USAGE.md when generation fails"""
        project_name = config.get("project_name", "Project")
        return f"""# Usage Guide for {project_name}

        ## Getting Started
        Project: {config.get('project_name', 'Project')}
Getting Started
Install dependencies from requirements.txt

Configure environment variables

Run the main application

Basic Usage
Follow the setup instructions in README.md

Use the provided entry points

Refer to individual component documentation
"""
    
    def _create_fallback_data_guide(self, config: Dict[str, Any]) -> str:
        """Create a fallback DATA_GUIDE.md when generation fails"""
        project_name = config.get("project_name", "Project")
        return f"""# Data Guide for {project_name}

        ## Using Custom Datasets
        Data Preparation
        Data Sources
        Custom datasets can be integrated

        Follow the data loading patterns in the code

        Ensure proper data formatting

        Data Processing
        Preprocessing scripts provided

        Data validation included

        Format conversion utilities available
        """ 