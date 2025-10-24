"""
Documentation Agent - Comprehensive Documentation Generation

This agent generates various types of documentation for Python code.
Uses temperature=0.6 for clear, creative explanations.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
from typing import Dict, List, Any, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent


class DocumentationAgent(BaseAgent):
    """
    Documentation Agent for generating comprehensive documentation.
    
    Responsibilities:
    - Generate docstrings for functions, classes, modules
    - Create README files
    - Generate API documentation
    - Add inline code comments
    - Create usage examples
    - Generate change logs
    
    Uses temperature=0.6 for clear, natural documentation.
    """
    
    def __init__(self, name: str = "documentation_agent"):
        """
        Initialize Documentation Agent.
        
        Args:
            name: Agent name (default: "documentation_agent")
        """
        role = "Expert Technical Writer for Python ML/DL Projects"
        super().__init__(
            name=name,
            role=role,
            agent_type="documentation"  # Uses temperature 0.6
        )
        logger.info(f"{self.name} ready for documentation tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a documentation task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of documentation task
                - data: Task-specific parameters
        
        Returns:
            Dictionary with documentation results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        try:
            if task_type == "generate_docstrings":
                return await self.generate_docstrings(
                    code=data.get("code"),
                    style=data.get("style", "google")
                )
            
            elif task_type == "generate_readme":
                return await self.generate_readme(
                    project_info=data.get("project_info", {})
                )
            
            elif task_type == "generate_api_docs":
                return await self.generate_api_docs(
                    code=data.get("code"),
                    format=data.get("format", "markdown")
                )
            
            elif task_type == "add_comments":
                return await self.add_comments(
                    code=data.get("code")
                )
            
            elif task_type == "generate_usage_examples":
                return await self.generate_usage_examples(
                    code=data.get("code"),
                    num_examples=data.get("num_examples", 3)
                )
            
            elif task_type == "generate_changelog":
                return await self.generate_changelog(
                    changes=data.get("changes", []),
                    version=data.get("version", "1.0.0")
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing documentation task '{task_type}': {e}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def generate_docstrings(
        self,
        code: str,
        style: str = "google"
    ) -> Dict[str, Any]:
        """
        Generate docstrings for code.
        
        Args:
            code: Code to generate docstrings for
            style: Docstring style (google, numpy, sphinx)
        
        Returns:
            Dictionary with documented code
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for docstring generation"
            }
        
        logger.info(f"{self.name} generating {style} docstrings ({len(code)} chars)")
        
        prompt_data = {
            "code": code,
            "style": style
        }
        
        prompt = self.get_prompt("documentation_prompts", "generate_docstrings", prompt_data)
        response = await self.generate_response(prompt)
        
        # Extract documented code
        documented_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "generate_docstrings",
            "style": style,
            "documented_code": documented_code,
            "functions_documented": self._count_docstrings(documented_code)
        }
    
    async def generate_readme(
        self,
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate README file.
        
        Args:
            project_info: Dictionary with project details
                - name: Project name
                - description: Project description
                - features: List of features
                - installation: Installation instructions
                - usage: Usage examples
        
        Returns:
            Dictionary with README content
        """
        if not project_info:
            return {
                "status": "error",
                "message": "Project info is required for README generation"
            }
        
        logger.info(f"{self.name} generating README for {project_info.get('name', 'project')}")
        
        # Format project info for prompt
        name = project_info.get("name", "Project")
        description = project_info.get("description", "A Python project")
        features = project_info.get("features", [])
        technologies = project_info.get("technologies", "Python, ML/DL frameworks")
        
        features_text = "\n".join([f"- {f}" for f in features]) if features else "- Feature 1\n- Feature 2"
        
        prompt_data = {
            "project_name": name,
            "description": description,
            "technologies": technologies,
            "features": features_text
        }
        
        prompt = self.get_prompt("documentation_prompts", "generate_readme", prompt_data)
        response = await self.generate_response(prompt)
        
        # Ensure response is a string
        if not isinstance(response, str):
            logger.error(f"generate_response returned non-string type: {type(response)}")
            if isinstance(response, list):
                response = "\n".join(str(item) for item in response)
            else:
                response = str(response) if response else "# README\n\nError generating documentation."
        
        # Additional validation
        if not response or len(response.strip()) == 0:
            logger.warning("Empty README response, using fallback template")
            response = f"""# {name}

## Overview
{description}

## Technologies
{technologies}

## Features
{features_text}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Please refer to the source code for usage instructions.
"""
        
        return {
            "status": "success",
            "task": "generate_readme",
            "readme_content": response,
            "sections": self._extract_sections(response)
        }
    
    async def generate_api_docs(
        self,
        code: str,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate API documentation.
        
        Args:
            code: Code to document
            format: Output format (markdown, html, rst)
        
        Returns:
            Dictionary with API documentation
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for API documentation"
            }
        
        logger.info(f"{self.name} generating API docs in {format} format")
        
        # Use generate_api_documentation prompt (note: different name in yaml)
        prompt_data = {
            "modules": f"Code to document:\n```python\n{code}\n```"
        }
        
        prompt = self.get_prompt("documentation_prompts", "generate_api_documentation", prompt_data)
        response = await self.generate_response(prompt)
        
        return {
            "status": "success",
            "task": "generate_api_docs",
            "format": format,
            "api_docs": response,
            "endpoints": self._extract_api_endpoints(response)
        }
    
    async def add_comments(
        self,
        code: str
    ) -> Dict[str, Any]:
        """
        Add inline comments to code.
        
        Args:
            code: Code to add comments to
        
        Returns:
            Dictionary with commented code
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for comment addition"
            }
        
        logger.info(f"{self.name} adding comments to code")
        
        # Use docstrings prompt with instruction to also add inline comments
        code_with_instruction = f"{code}\n\n# Add clear inline comments explaining the logic and flow"
        
        prompt_data = {
            "code": code_with_instruction
        }
        
        prompt = self.get_prompt("documentation_prompts", "generate_docstrings", prompt_data)
        response = await self.generate_response(prompt)
        
        commented_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "add_comments",
            "commented_code": commented_code,
            "comments_added": self._count_comments(commented_code)
        }
    
    async def generate_usage_examples(
        self,
        code: str,
        num_examples: int = 3
    ) -> Dict[str, Any]:
        """
        Generate usage examples.
        
        Args:
            code: Code to generate examples for
            num_examples: Number of examples to generate
        
        Returns:
            Dictionary with usage examples
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for usage example generation"
            }
        
        logger.info(f"{self.name} generating {num_examples} usage examples")
        
        # Create custom prompt for examples
        examples_prompt = f"""Generate {num_examples} practical usage examples for this code:

```python
{code}
```

For each example, provide:
1. A clear title/scenario
2. Complete working code
3. Expected output/result
4. Brief explanation

Make examples progressively more complex (basic → intermediate → advanced).
Format with markdown headers and code blocks.
"""
        
        response = await self.generate_response(examples_prompt)
        
        examples = self._extract_examples(response)
        
        return {
            "status": "success",
            "task": "generate_usage_examples",
            "examples": examples,
            "num_examples": len(examples)
        }
    
    async def generate_changelog(
        self,
        changes: List[str],
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Generate changelog entry.
        
        Args:
            changes: List of changes
            version: Version number
        
        Returns:
            Dictionary with changelog content
        """
        if not changes:
            return {
                "status": "error",
                "message": "Changes list is required for changelog generation"
            }
        
        logger.info(f"{self.name} generating changelog for version {version}")
        
        # Create custom prompt for changelog
        changes_text = "\n".join([f"- {c}" for c in changes])
        
        changelog_prompt = f"""Generate a professional changelog entry for version {version}:

Changes:
{changes_text}

Format following Keep a Changelog standard:
- Start with version number and date
- Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security
- Use clear, actionable language
- Include brief descriptions

Output clean markdown format.
"""
        
        response = await self.generate_response(changelog_prompt)
        
        return {
            "status": "success",
            "task": "generate_changelog",
            "version": version,
            "changelog": response,
            "change_categories": self._extract_change_categories(response)
        }
    
    # ==================== Helper Methods ====================
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response.
        
        Args:
            response: LLM response text
        
        Returns:
            Extracted code
        """
        # Try to find code in markdown blocks
        code_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            # Return the last code block (usually the complete file)
            return matches[-1].strip()
        
        # If no markdown blocks, return the whole response
        return response.strip()
    
    def _count_docstrings(self, code: str) -> int:
        """
        Count docstrings in code.
        
        Args:
            code: Code to analyze
        
        Returns:
            Number of docstrings
        """
        # Count triple-quoted strings that follow function/class definitions
        pattern = r'(?:def|class)\s+\w+[^:]*:\s*(?:"""|\'\'\').*?(?:"""|\'\'\')'
        matches = re.findall(pattern, code, re.DOTALL)
        return len(matches)
    
    def _extract_sections(self, readme: str) -> List[str]:
        """
        Extract section headers from README.
        
        Args:
            readme: README content
        
        Returns:
            List of section names
        """
        # Find markdown headers
        headers = re.findall(r'^#+\s+(.+)$', readme, re.MULTILINE)
        return headers
    
    def _extract_api_endpoints(self, docs: str) -> List[str]:
        """
        Extract API endpoints from documentation.
        
        Args:
            docs: API documentation
        
        Returns:
            List of endpoints/functions
        """
        endpoints = []
        
        # Look for function/method signatures
        patterns = [
            r'`(\w+\([^)]*\))`',  # Inline code functions
            r'###?\s+(\w+)',       # Headers with function names
            r'^\s*-\s*`?(\w+)\(?'  # List items with function names
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, docs, re.MULTILINE)
            endpoints.extend(matches)
        
        return list(set(endpoints))[:20]  # Return unique, limit to 20
    
    def _count_comments(self, code: str) -> int:
        """
        Count comments in code.
        
        Args:
            code: Code to analyze
        
        Returns:
            Number of comments
        """
        # Count single-line comments
        single_line = len(re.findall(r'^\s*#[^#]', code, re.MULTILINE))
        
        # Count multi-line comments/docstrings
        multi_line = len(re.findall(r'(?:"""|\'\'\')[^\'\"]*(?:"""|\'\'\')', code, re.DOTALL))
        
        return single_line + multi_line
    
    def _extract_examples(self, response: str) -> List[Dict[str, str]]:
        """
        Extract usage examples from response.
        
        Args:
            response: LLM response
        
        Returns:
            List of examples with titles and code
        """
        examples = []
        
        # Split response into sections
        lines = response.split('\n')
        current_example = None
        current_code = []
        in_code_block = False
        
        for line in lines:
            # Check for example headers
            if re.match(r'^#+\s+Example\s+\d+', line, re.IGNORECASE) or \
               re.match(r'^\*\*Example\s+\d+', line, re.IGNORECASE):
                # Save previous example
                if current_example and current_code:
                    examples.append({
                        "title": current_example,
                        "code": '\n'.join(current_code).strip()
                    })
                
                current_example = line.strip('#* ').strip()
                current_code = []
                in_code_block = False
            
            # Track code blocks
            elif '```' in line:
                in_code_block = not in_code_block
            elif in_code_block:
                current_code.append(line)
        
        # Save last example
        if current_example and current_code:
            examples.append({
                "title": current_example,
                "code": '\n'.join(current_code).strip()
            })
        
        return examples if examples else [{"title": "Example", "code": self._extract_code(response)}]
    
    def _extract_change_categories(self, changelog: str) -> List[str]:
        """
        Extract change categories from changelog.
        
        Args:
            changelog: Changelog content
        
        Returns:
            List of change categories
        """
        categories = []
        
        # Common changelog categories
        category_keywords = [
            'added', 'changed', 'deprecated', 'removed', 
            'fixed', 'security', 'features', 'improvements'
        ]
        
        lines = changelog.lower().split('\n')
        for line in lines:
            for keyword in category_keywords:
                if keyword in line and (line.strip().startswith('#') or line.strip().startswith('**')):
                    categories.append(keyword.title())
        
        return list(set(categories))
