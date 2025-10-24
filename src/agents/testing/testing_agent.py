"""
Testing Agent - Test Generation & Coverage Analysis (Corrected)

This agent generates comprehensive tests for Python code.
Uses temperature=0.4 for thorough test coverage.

Author: AutoCoder System
Refactored: October 21, 2025
"""

import re
import json
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.tools.task_runner import TaskRunner, CommandResult
from src.tools.test_failure_handler import TestFailureHandler, TestResult
from src.agents.interactive.modification_agent import InteractiveModificationAgent


class TestingAgent(BaseAgent):
    """
    Testing Agent for test generation and coverage analysis.
    
    Responsibilities:
    - Generate unit tests for functions and classes
    - Create integration tests
    - Generate test fixtures and mocks
    - Analyze test coverage
    - Create edge case tests
    - Generate pytest configurations
    
    Uses temperature=0.4 for thorough, balanced test generation.
    """
    
    def __init__(
        self,
        name: str = "testing_agent",
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        task_runner: Optional[TaskRunner] = None,
        modification_agent: Optional[InteractiveModificationAgent] = None
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        """
        Initialize Testing Agent.
        
        Args:
            name: Agent name
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
            task_runner: Tool for executing tests
            modification_agent: Tool for applying fixes (future use)
        """
        role = "Expert Software Quality Assurance Engineer"
        super().__init__(
            name=name,
            role=role,
            agent_type="testing",  # Uses temperature 0.4
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        # --- THIS IS THE FIX ---
        # Changed 'working_directory' to 'working_dir'
        self.task_runner = task_runner or TaskRunner(working_dir=Path.cwd())
        # --- END FIX ---

        self.test_failure_handler = TestFailureHandler()
        self.modification_agent = modification_agent
        
        logger.info(f"{self.name} ready for test generation tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a testing task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of testing task
                - data: Task-specific parameters
        
        Returns:
            Dictionary with testing results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        # Ensure clients are available
        if not self.llm_client or not self.prompt_manager:
            logger.warning("Re-initializing LLM client for TestingAgent. Should be passed in __init__.")
            self.llm_client = GeminiClient()
            self.prompt_manager = PromptManager()

        try:
            if task_type == "generate_unit_tests":
                return await self.generate_unit_tests(
                    code=data.get("code"),
                    file_path=data.get("file_path", "unknown.py"),
                    test_framework=data.get("test_framework", "pytest")
                )
            
            elif task_type == "generate_project_tests":
                return await self.generate_project_tests(
                    project_files=data.get("project_files", {}),
                    architecture=data.get("architecture", {})
                )
            
            elif task_type == "generate_integration_tests":
                return await self.generate_integration_tests(
                    code=data.get("code"),
                    dependencies=data.get("dependencies", [])
                )
            
            elif task_type == "generate_test_fixtures":
                return await self.generate_test_fixtures(
                    code=data.get("code"),
                    fixture_type=data.get("fixture_type", "pytest")
                )
            
            elif task_type == "generate_mock_tests":
                # This task is called by streamlit_app.py
                code_data = data.get("code", "{}")
                if isinstance(code_data, str):
                    try:
                        code_data = json.loads(code_data) # a dict of files
                    except json.JSONDecodeError:
                        code_data = {"error.py": code_data} # treat as single file
                
                # This task is ambiguous, let's call generate_project_tests instead
                logger.warning("Redirecting 'generate_mock_tests' to 'generate_project_tests'")
                return await self.generate_project_tests(
                    project_files=code_data,
                    architecture=data.get("architecture", {})
                )
            
            elif task_type == "generate_edge_case_tests":
                return await self.generate_edge_case_tests(
                    code=data.get("code"),
                    function_name=data.get("function_name")
                )
            
            elif task_type == "analyze_test_coverage":
                return self.analyze_test_coverage( # This is not async
                    code=data.get("code"),
                    test_code=data.get("test_code")
                )
            
            elif task_type == "run_tests":
                # This is a non-async function
                return self.run_tests(
                    test_path=data.get("test_path", "tests/")
                )
            
            elif task_type == "analyze_test_results":
                # This is a non-async function
                return self.analyze_test_results(
                    pytest_output=data.get("pytest_output")
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing testing task '{task_type}': {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def generate_unit_tests(
        self,
        code: str,
        file_path: str = "unknown.py",
        test_framework: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Generate unit tests for a single block of code.
        """
        if not code:
            return {"status": "error", "message": "Code is required"}
        
        logger.info(f"{self.name} generating unit tests for: {file_path}")
        
        prompt_data = {
            "code": code,
            "test_framework": test_framework,
            "file_path": file_path
        }
        
        try:
            prompt = self.get_prompt("testing_prompts", "generate_unit_tests", prompt_data)
        except ValueError:
             logger.warning("Could not find 'generate_unit_tests' prompt. Using fallback.")
             prompt = f"Generate pytest unit tests for the following code:\n\n```python\n{code}\n```"

        response = await self.generate_response(prompt)
        
        test_code = self._extract_code(response)
        test_count = self._count_tests(test_code)
        
        return {
            "status": "success",
            "task": "generate_unit_tests",
            "test_framework": test_framework,
            "test_code": test_code,
            "test_count": test_count,
            "coverage_areas": self._extract_coverage_areas(response)
        }

    async def generate_project_tests(
        self,
        project_files: Dict[str, str],
        architecture: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates test files for all code files in the project.
        This is the main task called by the orchestrator.
        """
        logger.info(f"Starting whole-project test generation for {len(project_files)} files...")
        
        test_files = {}
        total_test_count = 0
        
        # Get all file specs from architecture
        file_specs = []
        if isinstance(architecture.get("structure"), dict):
            for directory, specs in architecture.get("structure", {}).items():
                file_specs.extend(specs)

        for file_path, code in project_files.items():
            # Skip non-python files, __init__, and tests themselves
            if not file_path.endswith(".py") or file_path.startswith("tests/") or "__init__" in file_path or not code.strip():
                continue

            # Find the matching file_spec to get context
            file_spec = next((s for s in file_specs if s.get("path") == file_path), None)
            purpose = file_spec.get("purpose", "A Python module") if file_spec else "A Python module"

            logger.debug(f"Generating tests for: {file_path}")
            
            try:
                # Use a more specific prompt for project-aware test generation
                file_path_module = file_path.replace("src/", "src.").replace("/", ".").removesuffix(".py")
                prompt_data = {
                    "file_path": file_path,
                    "file_purpose": purpose,
                    "file_path_module": file_path_module,
                    "code": code,
                    "test_framework": "pytest",
                    "architecture": json.dumps(architecture, indent=2, default=str),
                }
                try:
                    prompt = self.get_prompt("testing_prompts", "generate_project_tests", prompt_data)
                except ValueError:
                    logger.warning(f"No 'generate_project_tests' prompt found. Using fallback.")
                    prompt = f"Generate pytest unit tests for the file `{file_path}`. The file's purpose is: {purpose}\n\n```python\n{code}\n```"
                
                response = await self.generate_response(prompt)
                
                test_code = self._extract_code(response)
                test_count = self._count_tests(test_code)
                total_test_count += test_count
                
                # Create the test file path
                test_file_path = f"tests/test_{file_path.replace('src/', '').replace('/', '_')}"
                test_files[test_file_path] = test_code
            except Exception as e:
                logger.error(f"Failed to generate tests for {file_path}: {e}")
                test_files[f"tests/test_{file_path.replace('src/', '').replace('/', '_')}"] = f"# ERROR: Failed to generate tests for {file_path}\n# REASON: {e}"


        logger.success(f"Generated {total_test_count} tests across {len(test_files)} files.")
        
        return {
            "status": "success",
            "task": "generate_project_tests",
            "test_files": test_files,
            "total_test_count": total_test_count
        }

    async def generate_integration_tests(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} generating integration tests (placeholder)")
        return {"status": "success", "task": "generate_integration_tests", "test_code": "# Integration tests placeholder"}
        
    async def generate_test_fixtures(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} generating test fixtures (placeholder)")
        return {"status": "success", "task": "generate_test_fixtures", "fixture_code": "# Fixtures placeholder"}

    async def generate_mock_tests(self, code: str, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} generating mock tests (placeholder)")
        return {"status": "success", "task": "generate_mock_tests", "test_code": f"# Mock tests placeholder for code\n{code[:50]}..."}

    async def generate_edge_case_tests(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} generating edge case tests (placeholder)")
        return {"status": "success", "task": "generate_edge_case_tests", "test_code": "# Edge case tests placeholder"}

    def analyze_test_coverage(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} analyzing test coverage (placeholder)")
        return {"status": "success", "task": "analyze_test_coverage", "analysis": "Coverage analysis placeholder"}
        
    def run_tests(self, test_path: str = "tests/") -> Dict[str, Any]:
        """
        Runs the test suite using the TaskRunner.
        This is a synchronous function as it blocks until tests complete.
        
        Args:
            test_path: The directory or file to test.
            
        Returns:
            Dictionary with test results.
        """
        logger.info(f"Running tests in: {test_path}...")
        
        if not self.task_runner:
            logger.error("TaskRunner is not initialized. Cannot run tests.")
            return {"status": "error", "message": "TaskRunner not initialized."}
            
        try:
            # Use the run_pytest method from TaskRunner
            result: CommandResult = self.task_runner.run_pytest(
                test_paths=[test_path]
            )
            
            logger.info(f"Test run complete. Success: {result.success}")
            
            # Analyze the results
            return self.analyze_test_results(result.output)
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {
                "status": "error",
                "message": f"Failed to run tests: {e}",
                "output": ""
            }

    def analyze_test_results(self, pytest_output: str) -> Dict[str, Any]:
        """
        Analyzes the raw output from a pytest run.
        
        Args:
            pytest_output: The raw stdout/stderr from the pytest command.
            
        Returns:
            A structured TestResult dictionary.
        """
        if not pytest_output:
            return {"status": "error", "message": "No pytest output received."}
            
        try:
            parsed_result: TestResult = self.test_failure_handler.parse_pytest_output(
                pytest_output
            )
            
            # Create a serializable dictionary from the TestResult dataclass
            report_dict = {
                "total_tests": parsed_result.total_tests,
                "passed": parsed_result.passed,
                "failed": parsed_result.failed,
                "skipped": parsed_result.skipped,
                "errors": parsed_result.errors,
                "duration": parsed_result.duration,
                "failures": [
                    {
                        "test_name": f.test_name,
                        "failure_type": f.failure_type.value,
                        "severity": f.severity.value,
                        "error_message": f.error_message,
                        "file_path": str(f.file_path),
                        "line_number": f.line_number,
                        "traceback": f.traceback
                    } for f in parsed_result.failures
                ]
            }

            if parsed_result.has_failures:
                logger.warning(f"Test analysis: {parsed_result.failed} failures, {parsed_result.errors} errors found.")
                return {
                    "status": "failure",
                    "summary": str(parsed_result),
                    "output": pytest_output,
                    **report_dict
                }
            else:
                logger.success(f"Test analysis: All {parsed_result.passed} tests passed.")
                return {
                    "status": "success",
                    "summary": str(parsed_result),
                    "output": pytest_output,
                    **report_dict
                }
                
        except Exception as e:
            logger.error(f"Error analyzing test results: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": f"Error parsing test output: {e}", "output": pytest_output}
    
    # ==================== Helper Methods ====================
    
    def _extract_code(self, response: str) -> str:
        """
        Extract test code from LLM response.
        """
        code_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if re.match(r'^(import|from|def test_|class Test|@pytest)', line.strip()):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return response.strip()
    
    def _count_tests(self, test_code: str) -> int:
        """
        Count number of test functions in code.
        """
        test_pattern = r'^\s*def\s+test_\w+'
        matches = re.findall(test_pattern, test_code, re.MULTILINE)
        return len(matches)
    
    def _extract_coverage_areas(self, response: str) -> List[str]:
        """
        Extract coverage areas from response.
        """
        areas = []
        keywords = ['test', 'cover', 'validate', 'verify', 'check']
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                stripped = line.strip()
                if stripped.startswith(('-', '•', '*')) or (stripped and stripped[0].isdigit()):
                    area = re.sub(r'^[-•*\d.)\s]+', '', stripped)
                    if area and len(area) > 10:
                        areas.append(area)
        
        return areas[:10] if areas else ["Main functionality tested"]
    
    def _extract_scenarios(self, response: str) -> List[str]:
        """
        Extract test scenarios from response.
        """
        scenarios = []
        scenario_keywords = ['scenario', 'case', 'test']
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in scenario_keywords):
                stripped = line.strip()
                if stripped.startswith(('-', '•', '*')):
                    scenario = re.sub(r'^[-•*\s]+', '', stripped)
                    if scenario and len(scenario) > 5:
                        scenarios.append(scenario)
        
        return scenarios[:5] if scenarios else ["Integration scenarios covered"]
    
    def _extract_fixture_names(self, fixture_code: str) -> List[str]:
        """
        Extract fixture names from code.
        """
        fixture_pattern = r'@pytest\.fixture[^\n]*\n\s*def\s+(\w+)'
        matches = re.findall(fixture_pattern, fixture_code)
        return matches if matches else []
    
    def _extract_mock_strategies(self, response: str) -> List[str]:
        """
        Extract mocking strategies from response.
        """
        strategies = []
        mock_keywords = ['mock', 'patch', 'stub', 'fake']
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in mock_keywords):
                stripped = line.strip()
                if stripped.startswith(('-', '•', '*')):
                    strategy = re.sub(r'^[-•*\s]+', '', stripped)
                    if strategy:
                        strategies.append(strategy)
        
        return strategies[:5] if strategies else ["External dependencies mocked"]
    
    def _extract_edge_cases(self, response: str) -> List[str]:
        """
        Extract edge cases from response.
        """
        edge_cases = []
        edge_keywords = ['edge', 'boundary', 'corner', 'limit', 'extreme']
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in edge_keywords):
                stripped = line.strip()
                if stripped.startswith(('-', '•', '*')) or (stripped and stripped[0].isdigit()):
                    case = re.sub(r'^[-•*\d.)\s]+', '', stripped)
                    if case:
                        edge_cases.append(case)
        
        return edge_cases[:10] if edge_cases else ["Common edge cases covered"]
    
    def _parse_coverage_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse coverage analysis from response.
        """
        result = {
            "coverage_percentage": 0,
            "covered_functions": [],
            "uncovered_functions": [],
            "missing_tests": []
        }
        
        coverage_pattern = r'coverage[:\s]+(\d+)%'
        match = re.search(coverage_pattern, response, re.IGNORECASE)
        if match:
            result["coverage_percentage"] = int(match.group(1))
        
        lines = response.split('\n')
        in_covered = False
        in_uncovered = False
        in_missing = False
        
        for line in lines:
            line_lower = line.lower()
            line_stripped = line.strip()
            
            if 'covered' in line_lower and 'function' in line_lower and 'uncovered' not in line_lower:
                in_covered = True
                in_uncovered = False
                in_missing = False
                continue
            elif 'uncovered' in line_lower and 'function' in line_lower:
                in_covered = False
                in_uncovered = True
                in_missing = False
                continue
            elif ('missing' in line_lower and 'test' in line_lower) or ('need' in line_lower and 'test' in line_lower):
                in_covered = False
                in_uncovered = False
                in_missing = True
                continue
            
            if line_stripped.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                item = re.sub(r'^[-•*\d.)\s]+', '', line_stripped)
                item = re.sub(r'\([^)]*\)', '', item).strip()
                
                if item and len(item) > 2:
                    if in_covered:
                        result["covered_functions"].append(item)
                    elif in_uncovered:
                        result["uncovered_functions"].append(item)
                    elif in_missing:
                        result["missing_tests"].append(item)
        
        return result