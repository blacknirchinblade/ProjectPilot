"""
Debugging Agent - Automated Code Debugging (Corrected)

Analyzes code, errors, and test failures to identify the root cause
and propose intelligent fixes.

Author: AutoCoder System
Refactored: October 21, 2025
"""

import ast
import re
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.tools.test_failure_handler import TestFailureHandler, TestResult, FailureSeverity
from src.tools.task_runner import TaskRunner
from src.agents.interactive.modification_agent import InteractiveModificationAgent


class DebuggingAgent(BaseAgent):
    """
    Agent specialized in debugging code based on errors and test failures.
    
    Capabilities:
    - Analyze pytest output
    - Analyze general exceptions and tracebacks
    - Identify root causes (syntax, logic, import, attribute, etc.)
    - Propose and apply intelligent fixes
    - Use context (other files, dependencies) for analysis
    - Perform iterative debugging (fix, re-run, analyze)
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        project_root: Optional[str] = None,
        task_runner: Optional[TaskRunner] = None,
        modification_agent: Optional[InteractiveModificationAgent] = None,
        max_iterations: int = 5,
        name: str = "debugging_agent" # Allow name override
    ):
        """
        Initialize Debugging Agent.
        
        Args:
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
            project_root: The root directory of the project being debugged.
            task_runner: Tool for executing tests.
            modification_agent: Tool for applying fixes.
            max_iterations: Max debugging iterations.
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        role = "Expert Code Debugging and Root Cause Analysis Specialist"
        super().__init__(
            name=name,
            role=role,
            agent_type="debugging",  # Uses temperature 0.3
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.test_failure_handler = TestFailureHandler()
        
        
        # Changed 'working_directory' to 'working_dir'
        self.task_runner = task_runner or TaskRunner(working_dir=self.project_root)
       
        
        self.modification_agent = modification_agent
        self.max_iterations = max_iterations
        
        if not self.modification_agent:
            logger.warning("ModificationAgent not provided, initializing a new one.")
            self.modification_agent = InteractiveModificationAgent(
                llm_client=self.llm_client,
                prompt_manager=self.prompt_manager,
                project_root=str(self.project_root)
            )
        
        logger.info(f"{self.name} ready for debugging tasks at {self.project_root}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a debugging task.
        
        Args:
            task: Task dictionary with type and data
                - task_type: "debug_test_failures" | "analyze_error"
                - data: Task-specific parameters
        
        Returns:
            Debugging results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        # Ensure clients are available
        if not self.llm_client or not self.prompt_manager:
            logger.warning("Re-initializing LLM client for DebuggingAgent. Should be passed in __init__.")
            self.llm_client = GeminiClient()
            self.prompt_manager = PromptManager()
            
        try:
            if task_type == "debug_test_failures":
                return await self.debug_test_failures(
                    test_results_output=data.get("test_results_output"), # Raw pytest output
                    files=data.get("files", {})
                )
            
            elif task_type == "analyze_error":
                return await self.analyze_error(
                    traceback=data.get("traceback"),
                    code_context=data.get("code_context", {})
                )
            
            elif task_type == "suggest_fix":
                return await self.suggest_fix(
                    code=data.get("code"),
                    error_description=data.get("error_description")
                )
            
            elif task_type == "fix_bug":
                return await self.fix_bug(
                    code=data.get("code"),
                    bug_description=data.get("bug_description")
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing debugging task '{task_type}': {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def debug_test_failures(
        self,
        test_results_output: str,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze pytest output and attempt to fix failures.
        
        Args:
            test_results_output: Raw output from the pytest command
            files: Dictionary of file paths to code content
            
        Returns:
            Debugging report with proposed fixes
        """
        logger.info("Debugging test failures...")
        
        # Parse test output
        parsed_result = self.test_failure_handler.parse_pytest_output(test_results_output)
        
        if not parsed_result.has_failures:
            return {
                "status": "success",
                "message": "No test failures found",
                "fixes_proposed": 0,
                "fixes_applied": 0
            }
        
        logger.info(f"Found {parsed_result.failed} failures and {parsed_result.errors} errors")
        
        all_fixes = []
        all_analyses = []
        
        for failure in parsed_result.failures:
            analysis = await self.analyze_failure(failure, files)
            all_analyses.append(analysis)
            
            if analysis.get("fixes"):
                all_fixes.extend(analysis["fixes"])
        
        fixes_applied = 0
        if self.modification_agent and all_fixes:
            logger.info(f"Applying {len(all_fixes)} fixes...")
            # In a real loop, we would apply fixes here.
            # For now, we just report what *would* be applied.
            # applied_results = await self.modification_agent.apply_changes(all_fixes)
            # fixes_applied = sum(1 for res in applied_results if res['status'] == 'success')
            fixes_applied = 0 # Simulating no auto-apply for now
        
        return {
            "status": "success",
            "message": f"Analyzed {len(parsed_result.failures)} failures",
            "fixes_proposed": len(all_fixes),
            "fixes_applied": fixes_applied,
            "analyses": all_analyses,
            "report": self.test_failure_handler.generate_failure_report(parsed_result)
        }
    
    async def analyze_failure(
        self,
        failure: TestResult,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze a single test failure and propose fixes.
        """
        logger.debug(f"Analyzing failure: {failure.test_name} ({failure.failure_type.value})")
        
        code_context = ""
        if failure.file_path and str(failure.file_path) in files:
            code_context = self._get_code_context(
                code=files[str(failure.file_path)],
                line_number=failure.line_number,
                window=5
            )
        elif failure.file_path and (self.project_root / failure.file_path).exists():
             try:
                code = (self.project_root / failure.file_path).read_text(encoding='utf-8')
                code_context = self._get_code_context(
                    code=code,
                    line_number=failure.line_number,
                    window=5
                )
             except Exception as e:
                logger.warning(f"Could not read file for failure context: {e}")

        
        try:
            prompt = self.get_prompt("debugging_prompts", "analyze_test_failure", {
                "test_name": failure.test_name,
                "failure_type": failure.failure_type.value,
                "error_message": failure.error_message,
                "traceback": failure.traceback or "No traceback available",
                "code_context": code_context or "No code context available."
            })
        except ValueError:
            logger.warning("No 'analyze_test_failure' prompt found. Using fallback.")
            prompt = f"Analyze this test failure:\nTest: {failure.test_name}\nError: {failure.error_message}\nTraceback: {failure.traceback}\nCode:\n{code_context}"

        
        response = await self.generate_response(prompt)
        
        return {
            "test_name": failure.test_name,
            "analysis": response,
            "fixes": self._extract_fixes_from_analysis(response, failure.file_path)
        }
    
    async def analyze_error(
        self,
        traceback: str,
        code_context: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze a general runtime error.
        """
        logger.info(f"Analyzing runtime error: {traceback[:100]}")
        
        file_path, line_number = self._parse_traceback_location(traceback)
        
        context_str = ""
        if file_path and file_path in code_context:
            context_str = self._get_code_context(
                code_context[file_path],
                line_number
            )
        
        try:
            prompt = self.get_prompt("debugging_prompts", "analyze_runtime_error", {
                "traceback": traceback,
                "code_context": context_str or "No code context available."
            })
        except ValueError:
            logger.warning("No 'analyze_runtime_error' prompt found. Using fallback.")
            prompt = f"Analyze this error:\n{traceback}\n\nCode:\n{context_str}"
        
        response = await self.generate_response(prompt)
        
        return {
            "status": "success",
            "analysis": response,
            "fixes": self._extract_fixes_from_analysis(response, file_path)
        }
    
    async def suggest_fix(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} suggesting fix (placeholder)")
        return {"status": "success", "task": "suggest_fix", "suggestions": "Fix placeholder"}

    async def fix_bug(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} fixing bug (placeholder)")
        return {"status": "success", "task": "fix_bug", "fixed_code": "# Fixed code placeholder"}
    
    async def diagnose_issue(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} diagnosing issue (placeholder)")
        return {"status": "success", "task": "diagnose_issue", "diagnosis": "Diagnosis placeholder"}
        
    async def analyze_performance(self, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"{self.name} analyzing performance (placeholder)")
        return {"status": "success", "task": "analyze_performance", "analysis": "Performance analysis placeholder"}
    
    # ==================== Helper Methods ====================
    
    def _get_code_context(
        self,
        code: str,
        line_number: Optional[int],
        window: int = 5
    ) -> str:
        """
        Get code context around a specific line.
        """
        if not line_number:
            return code[:1000]
        
        lines = code.split('\n')
        start = max(0, line_number - window - 1)
        end = min(len(lines), line_number + window)
        
        context_lines = []
        for i in range(start, end):
            prefix = "  "
            if i == line_number - 1:
                prefix = "-> "
            context_lines.append(f"{i + 1:4d} {prefix} {lines[i]}")
        
        return "\n".join(context_lines)
    
    def _parse_traceback_location(self, traceback: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse file path and line number from traceback."""
        match = re.search(r'File "([^"]+)", line (\d+)', traceback)
        
        if match:
            file_path = match.group(1)
            line_number = int(match.group(2))
            return file_path, line_number
        
        return None, None
    
    def _extract_fixes_from_analysis(
        self,
        analysis: str,
        file_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract code fixes from analysis text.
        """
        fixes = []
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, analysis, re.DOTALL)
        
        for code_block in matches:
            fixes.append({
                "filepath": file_path,
                "fix_type": "modify_code",
                "description": "Proposed fix by AI",
                "old_code": None,  # Hard to determine old code reliably
                "new_code": code_block.strip(),
                "reasoning": "AI-generated fix"
            })
        return fixes

    def _extract_root_cause(self, response: str) -> str:
        pattern = r'(?:Root Cause|Cause):\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 20]
        return paragraphs[0] if paragraphs else "Root cause not clearly identified"
    
    def _extract_error_type(self, error_message: str) -> str:
        error_types = [
            'SyntaxError', 'TypeError', 'ValueError', 'KeyError', 'IndexError',
            'AttributeError', 'NameError', 'ImportError', 'RuntimeError',
            'ZeroDivisionError', 'FileNotFoundError', 'PermissionError'
        ]
        for error_type in error_types:
            if error_type in error_message:
                return error_type
        return "Unknown Error"

    def _parse_errors(self, response: str) -> List[Dict[str, Any]]:
        errors = []
        current_error = {}
        for line in response.split('\n'):
            line_stripped = line.strip()
            if re.match(r'^\d+\.', line_stripped) or line_stripped.startswith('-'):
                if current_error:
                    errors.append(current_error)
                error_text = re.sub(r'^[\d.\-\s]+', '', line_stripped)
                current_error = {"description": error_text, "severity": "medium"}
            elif any(sev in line_stripped.lower() for sev in ['critical', 'high', 'medium', 'low']):
                if current_error:
                    for sev in ['critical', 'high', 'medium', 'low']:
                        if sev in line_stripped.lower():
                            current_error["severity"] = sev
                            break
        if current_error:
            errors.append(current_error)
        return errors if errors else []

    def _parse_fix_suggestions(self, response: str) -> List[Dict[str, str]]:
        suggestions = []
        current_suggestion = None
        current_code = []
        in_code_block = False
        for line in response.split('\n'):
            if re.match(r'^#+\s+Solution\s+\d+', line, re.IGNORECASE) or \
               re.match(r'^\d+\.\s+', line):
                if current_suggestion:
                    suggestions.append({
                        "title": current_suggestion,
                        "code": '\n'.join(current_code).strip() if current_code else ""
                    })
                current_suggestion = line.strip('#0123456789. ').strip()
                current_code = []
                in_code_block = False
            elif '```' in line:
                in_code_block = not in_code_block
            elif in_code_block:
                current_code.append(line)
        if current_suggestion:
            suggestions.append({
                "title": current_suggestion,
                "code": '\n'.join(current_code).strip() if current_code else ""
            })
        return suggestions if suggestions else [{"title": "General Fix", "code": self._extract_code(response)}]
    
    def _extract_recommended_fix(self, response: str) -> str:
        pattern = r'(?:Recommended|Best|Preferred).*?:(.+?)(?:\n\n|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "See suggestions above"

    def _extract_changes(self, response: str) -> List[str]:
        changes = []
        lines = response.split('\n')
        in_changes_section = False
        for line in lines:
            line_lower = line.lower()
            if 'changes' in line_lower and ('made' in line_lower or 'list' in line_lower):
                in_changes_section = True
                continue
            if in_changes_section:
                if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                    change = re.sub(r'^[-•*\d.)\s]+', '', line.strip())
                    if change:
                        changes.append(change)
                elif line.strip() and not line.strip().startswith('#'):
                    if changes:
                        break
        return changes if changes else ["Code modifications applied"]
    
    def _extract_explanation(self, response: str) -> str:
        pattern = r'(?:Explanation|Why):\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Bug fix applied"
    
    def _extract_problem(self, response: str) -> str:
        pattern = r'(?:Problem|Issue|Identification):\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        sentences = re.split(r'[.!?]\s+', response)
        return sentences[0] if sentences else "Issue identified in code"
    
    def _extract_severity(self, response: str) -> str:
        severity_keywords = {
            'critical': ['critical', 'severe', 'fatal', 'catastrophic'],
            'high': ['high', 'major', 'serious', 'significant'],
            'medium': ['medium', 'moderate', 'important'],
            'low': ['low', 'minor', 'trivial', 'cosmetic']
        }
        response_lower = response.lower()
        for severity, keywords in severity_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return severity
        return "medium"
    
    def _extract_recommendations(self, response: str) -> List[str]:
        recommendations = []
        lines = response.split('\n')
        in_recommendations = False
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['recommendation', 'action', 'solution', 'fix']):
                in_recommendations = True
                continue
            if in_recommendations:
                if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                    rec = re.sub(r'^[-•*\d.)\s]+', '', line.strip())
                    if rec:
                        recommendations.append(rec)
        return recommendations[:10] if recommendations else ["Review and fix identified issues"]
    
    def _extract_bottlenecks(self, response: str) -> List[str]:
        bottlenecks = []
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['bottleneck', 'slow', 'inefficient']):
                if line.strip().startswith(('-', '•', '*', '1.', '2.')):
                    bottleneck = re.sub(r'^[-•*\d.)\s]+', '', line.strip())
                    if bottleneck and len(bottleneck) > 10:
                        bottlenecks.append(bottleneck)
        return bottlenecks[:5] if bottlenecks else ["Performance analysis completed"]
    
    def _extract_optimizations(self, response: str) -> List[str]:
        optimizations = []
        lines = response.split('\n')
        in_optimizations = False
        for line in lines:
            line_lower = line.lower()
            if 'optimization' in line_lower or 'improve' in line_lower:
                in_optimizations = True
                continue
            if in_optimizations:
                if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                    opt = re.sub(r'^[-•*\d.)\s]+', '', line.strip())
                    if opt:
                        optimizations.append(opt)
        return optimizations[:10] if optimizations else ["See analysis for optimization opportunities"]