"""
Coding Agent - Main code generation agent

This is the PRIMARY agent responsible for:
- Generating Python modules, classes, and functions
- Creating ML/DL pipelines
- Building data loaders and preprocessors
- Generating configuration managers
- Writing production-ready, well-documented code

Features:
- Uses temperature 0.4 for balanced creativity and precision
- Integrated ProblemChecker for automatic code validation
- Comprehensive error detection (syntax, imports, style)
- Fix suggestions for common issues
- Validation results included in generation output

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import traceback
import json
import re
import asyncio
from typing import Dict, Any, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.tools.problem_checker import ProblemChecker, ProblemSeverity


class CodingAgent(BaseAgent):
    """
    Main Coding Agent for production-grade code generation
    
    Generates clean, well-documented, PEP 8 compliant code with:
    - Type hints
    - Comprehensive docstrings
    - Error handling
    - Input validation
    - Example usage
    - **Iterative refinement from feedback**
    """
    
    def __init__(
        self,
        name: str = "coding_agent",
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        enable_validation: bool = True
    ):
        """
        Initialize Coding Agent
        
        Args:
            name: Agent name
            llm_client: LLM client for generation
            prompt_manager: Prompt manager for templates
            enable_validation: Enable ProblemChecker validation (default True)
        """
        super().__init__(
            name=name,
            role="Expert Python Developer for ML/DL/NLP/CV Projects",
            agent_type="coding",  # Uses temperature 0.4
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.enable_validation = enable_validation
        if enable_validation:
            self.problem_checker = ProblemChecker(
                enable_flake8=False,
                enable_pylint=False,
                max_line_length=1000
            )
            logger.info(f"{self.name} initialized with ProblemChecker validation")
        else:
            self.problem_checker = None
            logger.info(f"{self.name} initialized without validation")
        
        logger.info(f"{self.name} ready for code generation tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a coding task
        
        Args:
            task: Task dictionary with:
                - task_type: Type of coding task
                - data: Task-specific data
                - **feedback: Optional feedback from previous iteration**
        
        Returns:
            Result dictionary with generated code
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        feedback = task.get("feedback", None) # <-- NEW: Get feedback

        try:
            # --- THIS IS THE NEW, PRIMARY TASK TYPE ---
            if task_type == "generate_from_spec":
                return await self.generate_from_spec(data, feedback)
            # -------------------------------------------
            elif task_type == "generate_module":
                return await self.generate_module(data)
            elif task_type == "generate_class":
                return await self.generate_class(data)
            elif task_type == "generate_function":
                return await self.generate_function(data)
            elif task_type == "create_ml_pipeline":
                return await self.create_ml_pipeline(data)
            elif task_type == "create_data_loader":
                return await self.create_data_loader(data)
            elif task_type == "generate_config_manager":
                return await self.generate_config_manager(data)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown task type: {task_type}"
                }
        except Exception as e:
            logger.error(f"Error executing coding task '{task_type}': {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

    
    async def generate_module(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete Python module
        (Legacy method for simple generation)
        """
        try:
            module_name = data.get("module_name", "")
            purpose = data.get("purpose", "")
            requirements = data.get("requirements", "")
            
            if not module_name or not purpose:
                return {
                    "status": "error",
                    "message": "Missing module_name or purpose"
                }
            
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="generate_module",
                variables={
                    "module_name": module_name,
                    "purpose": purpose,
                    "requirements": requirements
                }
            )
            
            logger.info(f"{self.name} generating module: {module_name}")
            
            estimated_size = len(requirements) * 100 + len(purpose) * 10
            
            code, attempts = await self._generate_with_retry(
                prompt=prompt,
                task_name=f"module_{module_name}",
                max_retries=3,
                expected_size_chars=estimated_size,
                is_python_file=True
            )
            
            validation = self.validate_code_structure(code, is_python_file=True)
            
            if validation["is_valid"]:
                logger.info(f"✓ Generated code passed validation (attempts: {attempts})")
            else:
                logger.warning(
                    f"⚠ Generated code has {validation['error_count']} errors, "
                    f"{validation['warning_count']} warnings (attempts: {attempts})"
                )
                if validation.get("suggestions"):
                    logger.info(f"Fix suggestions available: {len(validation['suggestions'])}")
            
            return {
                "status": "success",
                "task": "generate_module",
                "module_name": module_name,
                "code": code,
                "language": "python",
                "validation": validation,
                "attempts": attempts
            }
        
        except Exception as e:
            logger.error(f"Error generating module: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_from_spec(
        self, 
        data: Dict[str, Any], 
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate code from detailed file specification (NEW: Intelligent Architecture & Iteration)
        
        Args:
            data: Dictionary with:
                - file_spec: Detailed specification from DynamicArchitectAgent
                - import_map: Map of all imports for correct paths
                - project_context: Overall project information
            feedback: (Optional) Feedback from review and test agents
        
        Returns:
            Generated code with correct imports and minimal docstrings
        """
        try:
            file_spec = data.get("file_spec")
            import_map = data.get("import_map", {})
            project_context = data.get("project_context", {})
            
            if not file_spec:
                return {"status": "error", "message": "Missing file_spec"}
            
            file_path = file_spec.get("path", "")
            file_name = file_spec.get("name", "")
            purpose = file_spec.get("purpose", "")
            components = file_spec.get("components", [])
            dependencies = file_spec.get("dependencies", [])


            # --- THIS IS THE FIX ---
            # Check if the file is a Python file
            is_python_file = file_path.endswith(".py")
            # --- END FIX ---
            
            # Build import list from import_map
            imports_list = []
            for dep in dependencies:
                if dep in import_map:
                    imports_list.append(f"from {import_map[dep]['module']} import {import_map[dep]['symbol']}")
            
            imports_text = "\n".join(imports_list) if imports_list else "# No internal project imports needed."
            
            components_text = "\n".join([
                f"- {comp.get('name', 'unknown')}: {comp.get('description', 'no description')}"
                for comp in components
            ])
            
            # --- NEW: Build feedback block ---
            feedback_text = ""
            if feedback:
                logger.info(f"Applying feedback to {file_path}...")
                feedback_text = "--- PREVIOUS ATTEMPT FEEDBACK ---\n"
                if feedback.get("review_report"):
                    review_summary = feedback["review_report"].get("summary", {})
                    feedback_text += f"Review Score: {review_summary.get('overall_score', 'N/A')}\n"
                    feedback_text += "Review Issues:\n"
                    for issue in feedback["review_report"].get("issues", []):
                        if issue.get("file_path") == file_path: # Only show feedback for this file
                            feedback_text += f"- (Line {issue.get('line', 'N/A')}) {issue.get('message')}\n"
                
                if feedback.get("test_report"):
                    feedback_text += "\nTest Failures:\n"
                    for failure in feedback["test_report"].get("failures", []):
                        if file_path in failure.get("traceback", ""): # Check if file is in traceback
                            feedback_text += f"- {failure.get('test_name')}: {failure.get('error_message')}\n"
                
                feedback_text += "\n--- PLEASE FIX THESE ISSUES IN THE NEW CODE ---\n"
            # --- END NEW BLOCK ---

            # --- MODIFIED: Use PromptManager ---
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="generate_from_spec",
                variables={
                    "project_name": project_context.get("name", "Unknown"),
                    "project_type": project_context.get("type", "Unknown"),
                    "technologies": ", ".join(project_context.get("technologies", [])),
                    "file_path": file_path,
                    "purpose": purpose,
                    "components": components_text,
                    "imports": imports_text,
                    "feedback": feedback_text  # Pass the feedback
                }
            )
            # --- END MODIFICATION ---
            
            logger.info(f"{self.name} generating from spec: {file_path}")
            
            estimated_size = len(components) * 50 * 80
            
            code, attempts = await self._generate_with_retry(
                prompt=prompt,
                task_name=f"spec_{file_name}",
                max_retries=3,
                expected_size_chars=estimated_size,
                is_python_file=is_python_file
            )
            
            validation = self.validate_code_structure(code, is_python_file=is_python_file)
            
            if validation["is_valid"]:
                logger.info(f"✓ Generated {file_path} (attempts: {attempts})")
            else:
                logger.warning(f"⚠ {file_path} has {validation['error_count']} issues (attempts: {attempts})")
            
            return {
                "status": "success",
                "task": "generate_from_spec",
                "file_path": file_path,
                "code": code,
                "language": "python" if is_python_file else file_path.split('.')[-1],
                "validation": validation,
                "attempts": attempts,
                "imports_used": len(dependencies)
            }
        
        except Exception as e:
            logger.error(f"Error generating from spec: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_class(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Python class
        """
        try:
            class_name = data.get("class_name", "")
            purpose = data.get("purpose", "")
            attributes = data.get("attributes", "")
            methods = data.get("methods", "")
            
            if not class_name or not purpose:
                return {
                    "status": "error",
                    "message": "Missing class_name or purpose"
                }
            
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="generate_class",
                variables={
                    "class_name": class_name,
                    "purpose": purpose,
                    "attributes": attributes or "To be determined",
                    "methods": methods or "To be determined"
                }
            )
            
            logger.info(f"{self.name} generating class: {class_name}")
            code, attempts = await self._generate_with_retry(prompt, f"class_{class_name}", is_python_file=True)
            
            return {
                "status": "success",
                "task": "generate_class",
                "class_name": class_name,
                "code": code,
                "language": "python"
            }
        
        except Exception as e:
            logger.error(f"Error generating class: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_function(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Python function
        """
        try:
            function_name = data.get("function_name", "")
            purpose = data.get("purpose", "")
            parameters = data.get("parameters", "")
            return_type = data.get("return_type", "")
            
            if not function_name or not purpose:
                return {
                    "status": "error",
                    "message": "Missing function_name or purpose"
                }
            
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="generate_function",
                variables={
                    "function_name": function_name,
                    "purpose": purpose,
                    "parameters": parameters or "To be determined",
                    "return_type": return_type or "To be determined"
                }
            )
            
            logger.info(f"{self.name} generating function: {function_name}")
            code, attempts = await self._generate_with_retry(prompt, f"func_{function_name}", is_python_file=True)
            
            return {
                "status": "success",
                "task": "generate_function",
                "function_name": function_name,
                "code": code,
                "language": "python"
            }
        
        except Exception as e:
            logger.error(f"Error generating function: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def create_ml_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete ML pipeline
        """
        try:
            task_type = data.get("task_type", "")
            dataset_info = data.get("dataset_info", "")
            model_type = data.get("model_type", "")
            
            if not task_type:
                return {
                    "status": "error",
                    "message": "Missing task_type"
                }
            
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="create_ml_pipeline",
                variables={
                    "task_type": task_type,
                    "dataset_info": dataset_info or "General dataset",
                    "model_type": model_type or "To be determined"
                }
            )
            
            logger.info(f"{self.name} creating ML pipeline for: {task_type}")
            code, attempts = await self._generate_with_retry(prompt, f"pipeline_{task_type}", is_python_file=True)
            
            return {
                "status": "success",
                "task": "create_ml_pipeline",
                "ml_task": task_type,
                "code": code,
                "language": "python"
            }
        
        except Exception as e:
            logger.error(f"Error creating ML pipeline: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def create_data_loader(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a data loader
        """
        try:
            data_type = data.get("data_type", "")
            data_source = data.get("data_source", "")
            format_type = data.get("format", "")
            
            if not data_type:
                return {
                    "status": "error",
                    "message": "Missing data_type"
                }
            
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="create_data_loader",
                variables={
                    "data_type": data_type,
                    "data_source": data_source or "General source",
                    "format": format_type or "Standard format"
                }
            )
            
            logger.info(f"{self.name} creating data loader for: {data_type}")
            code, attempts = await self._generate_with_retry(prompt, f"loader_{data_type}", is_python_file=True)
            
            return {
                "status": "success",
                "task": "create_data_loader",
                "data_type": data_type,
                "code": code,
                "language": "python"
            }
        
        except Exception as e:
            logger.error(f"Error creating data loader: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_config_manager(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate configuration manager
        """
        try:
            project_name = data.get("project_name", "")
            parameters = data.get("parameters", "")
            
            if not project_name:
                return {
                    "status": "error",
                    "message": "Missing project_name"
                }
            
            prompt = self.get_prompt(
                category="coding_prompts",
                prompt_name="generate_config_manager",
                variables={
                    "project_name": project_name,
                    "parameters": parameters or "Standard parameters"
                }
            )
            
            logger.info(f"{self.name} generating config manager for: {project_name}")
            code, attempts = await self._generate_with_retry(prompt, f"config_{project_name}", is_python_file=True)

            return {
                "status": "success",
                "task": "generate_config_manager",
                "project_name": project_name,
                "code": code,
                "language": "python"
            }
        
        except Exception as e:
            logger.error(f"Error generating config manager: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract clean code from markdown-formatted response
        """
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        code_pattern = r'```(?:python)?\n(.*?)```'
        if matches:
            return matches[-1].strip()
        
        if "```" in response:
            code_pattern = r'```\n(.*?)\n```'
            matches = re.findall(code_pattern, response, re.DOTALL)
            if matches:
                return matches[-1].strip()
        
        lines = response.split('\n')
        code_start_idx = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                (stripped.startswith('# ') and i < 10)):
                code_start_idx = i
                break
        
        if code_start_idx != -1:
            return '\n'.join(lines[code_start_idx:]).strip()
        
        return response.strip()
    
    def _validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        """
        try:
            compile(code, '<generated>', 'exec')
            return True, None
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n   {e.text.rstrip()}"
                if e.offset:
                    error_msg += f"\n   {' ' * (e.offset - 1)}^"
            return False, error_msg
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def _check_completeness(self, code: str) -> dict[str, Any]:
        """
        Check code completeness beyond syntax.
        """
        issues = []
        
        if '# CODE_GENERATION_COMPLETE' not in code:
            issues.append("Missing completion marker '# CODE_GENERATION_COMPLETE'")
        
        lines = code.split('\n')
        last_lines = '\n'.join(lines[-5:]) if len(lines) > 5 else code
        
        incomplete_patterns = [
            (r'\.\s*$', 'Incomplete method call (line ends with .)'),
            (r'=\s*$', 'Incomplete assignment (line ends with =)'),
            (r',\s*\n\s*\)(?!\s*\n)', 'Trailing comma before close paren in last lines'),
        ]
        
        for pattern, message in incomplete_patterns:
            if re.search(pattern, last_lines):
                issues.append(message)
        
        function_count = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
        docstring_count = len(re.findall(r'"""[\s\S]*?"""', code)) + len(re.findall(r"'''[\s\S]*?'''", code))
        
        if function_count > 0 and docstring_count < function_count:
             issues.append(f"Missing docstrings: {docstring_count}/{function_count} functions have docstrings")
        
        return {
            'is_complete': len(issues) == 0,
            'issues': issues,
            'function_count': function_count,
            'docstring_count': docstring_count
        }
    
    def _auto_fix_common_issues(self, code: str) -> str:
        """
        Attempt to auto-fix common syntax issues in generated code.
        """
        return code # Auto-fix disabled
    
    async def _generate_with_retry(
        self,
        prompt: str,
        task_name: str,
        max_retries: int = 3,
        expected_size_chars: int = 0,
        is_python_file: bool = True
    ) -> tuple[str, int]:
        """
        Generate code with retry logic and validation.

        Args:
            is_python_file: If True, run Python validation.
        """
        is_large_file = expected_size_chars > 15000
        use_streaming = expected_size_chars > 20000 
        
        if is_large_file and is_python_file:
            logger.warning(f"⚠️ Generating LARGE file (expected {expected_size_chars} chars) - using enhanced validation")
            if use_streaming:
                logger.info(f"🌊 Using STREAMING mode to prevent timeouts")
            
            prompt = f"""{prompt}

⚠️ SPECIAL INSTRUCTIONS FOR LARGE FILE GENERATION ⚠️
This is a COMPLEX file (>{expected_size_chars // 1000}K chars). Extra care required:
1. Generate code in logical sections
2. DOUBLE-CHECK every line for completeness
3. Pay special attention to:
   - String closing quotes in EVERY line
   - Parenthesis pairing in EVERY line
   - No trailing commas/operators
4. Verify syntax multiple times
5. MUST end with '# CODE_GENERATION_COMPLETE'
"""
        
        attempt = 0
        last_code = ""
        original_prompt = prompt
        
        while attempt < max_retries:
            attempt += 1
            logger.info(f"Code generation attempt {attempt}/{max_retries} for {task_name}")
            
            if use_streaming:
                logger.info(f"🌊 Streaming generation for large file (~{expected_size_chars:,} chars expected)")
                response = await self.llm.stream_generate(
                    prompt,
                    agent_type=self.agent_type
                )
            else:
                response = await self.generate_response(prompt)
            
            code = self.extract_code_from_response(response)
            last_code = code

            # --- THIS IS THE FIX ---
            # If it's not a Python file, we don't validate it.
            # We just accept the LLM's output and return immediately.
            if not is_python_file:
                logger.success(f"✓ Generated non-Python file on attempt {attempt}. Skipping validation.")
                # We still want the marker for non-python files to check for completeness
                if '# CODE_GENERATION_COMPLETE' in code:
                    return code, attempt
                else:
                    logger.warning(f"Non-Python file missing completion marker on attempt {attempt}.")
                    # Add the marker ourselves
                    last_code += "\n# CODE_GENERATION_COMPLETE"
                    # Continue to retry logic to get a complete file
                    prompt = f"""{original_prompt}\n\n⚠️ PREVIOUS ATTEMPT WAS INCOMPLETE ⚠️\nIt was missing the final '# CODE_GENERATION_COMPLETE' marker. Please generate the full file and include the marker at the end."""
                    continue # Go to next retry
            # --- END FIX ---
            
            code = self._auto_fix_common_issues(code)
            
            is_valid, error = self._validate_syntax(code)
            
            if is_valid:
                completeness = self._check_completeness(code)
                
                if completeness['is_complete']:
                    logger.success(f"✓ Generated valid, complete code on attempt {attempt}")
                    return code, attempt
                else:
                    logger.warning(f"Code incomplete on attempt {attempt}: {completeness['issues']}")
                    
                    if attempt < max_retries:
                        fix_hints = [issue for issue in completeness['issues']]
                        
                        prompt = f"""{original_prompt}

⚠️ PREVIOUS ATTEMPT WAS INCOMPLETE ⚠️

Issues found:
{chr(10).join('- ' + issue for issue in completeness['issues'])}

STRICT REQUIREMENTS:
{chr(10).join('- ' + hint for hint in fix_hints)}
- Generate COMPLETE code from scratch
- Every line must be syntactically complete
- NO trailing commas, dots, or operators
- ALL strings must have closing quotes
- ALL brackets/parentheses must be paired
- MUST end with: # CODE_GENERATION_COMPLETE

This is attempt {attempt}. Generate PERFECT, COMPLETE code now.
"""
            else:
                logger.error(f"Syntax error in generated code (attempt {attempt}): {error}")
                
                if attempt < max_retries:
                    prompt = f"""{original_prompt}

⚠️ PREVIOUS ATTEMPT HAD SYNTAX ERROR ⚠️

Error details:
{error}

REGENERATION INSTRUCTIONS:
1. Start fresh - ignore previous broken code
2. Be EXTRA careful with syntax:
   - Close ALL strings with matching quotes
   - Close ALL parentheses/brackets
   - NO incomplete lines
3. Double-check every line is complete
4. Don't forget error handling where needed
5. No indentation errors
6. No incomplete code lines
7. End with: # CODE_GENERATION_COMPLETE

This is attempt {attempt}/{max_retries}. Generate COMPLETE, VALID Python code.
"""
            
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        logger.warning(f"All {max_retries} generation attempts had issues, returning last attempt")
        return last_code, max_retries
    
    def validate_code_structure(self, code: str, is_python_file: bool = True) -> Dict[str, Any]:
        """
        Comprehensive validation of generated code structure using ProblemChecker
        Only runs Python validation if is_python_file is True.
        """
        validation = {
            "has_imports": "import " in code or "from " in code,
            "has_docstrings": '"""' in code or "'''" in code,
            "has_type_hints": ": " in code and "->" in code,
            "has_functions": "def " in code,
            "has_classes": "class " in code,
            "is_valid": True,
            "problems": [],
            "error_count": 0,
            "warning_count": 0,
            "suggestions": []
        }

        if not is_python_file:
            validation["syntax_valid"] = True
            return validation
        # --- END FIX ---
        
        if self.enable_validation and self.problem_checker:
            try:
                check_result = self.problem_checker.check_code(code)
                validation["syntax_valid"] = check_result.is_valid
                
                errors = check_result.get_problems_by_severity(ProblemSeverity.ERROR)
                warnings = check_result.get_problems_by_severity(ProblemSeverity.WARNING)
                
                validation["error_count"] = len(errors)
                validation["warning_count"] = len(warnings)
                validation["problems"] = [
                    {
                        "line": p.line, "severity": p.severity.value,
                        "category": p.category.value, "message": p.message,
                        "code": p.code
                    }
                    for p in (errors + warnings)
                ]
                
                for error in errors:
                    if error.suggestion:
                        validation["suggestions"].append({
                            "line": error.line,
                            "message": error.message,
                            "fix": error.suggestion
                        })
                
                if errors:
                    validation["is_valid"] = False
                    logger.warning(f"Code validation found {len(errors)} errors")
                
            except Exception as e:
                logger.warning(f"ProblemChecker validation failed: {e}, falling back to basic check")
                is_valid, error_msg = self._validate_syntax(code)
                validation["syntax_valid"] = is_valid
                validation["is_valid"] = is_valid
                if not is_valid:
                    validation["error_count"] = 1
                    validation["problems"].append({"line": None, "severity": "error", "category": "syntax", "message": error_msg, "code": "E0001"})
        else:
            is_valid, error_msg = self._validate_syntax(code)
            validation["syntax_valid"] = is_valid
            validation["is_valid"] = is_valid
            if not is_valid:
                validation["error_count"] = 1
                validation["problems"].append({"line": None, "severity": "error", "category": "syntax", "message": error_msg, "code": "E0001"})
        
        return validation
    
# CODE_GENERATION_COMPLETE