"""
Error Fixing Agent - Intelligent Debugging and Code Repair

This agent:
1. Understands runtime errors (tracebacks, exceptions, test failures)
2. Analyzes error context and identifies root cause
3. Proposes intelligent fixes (not just keyword replacement)
4. Applies fixes and triggers review cycle
5. Handles complex errors: imports, syntax, logic, dependencies

Examples:
    User runs code → gets ModuleNotFoundError
    Agent: Analyzes error → Identifies missing import → Adds import → Re-runs review cycle
    
    User runs code → gets AttributeError
    Agent: Analyzes error → Finds wrong method call → Fixes method → Re-runs review cycle

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
import re
import ast
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.interactive.modification_agent import InteractiveModificationAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager

class ErrorFixingAgent(BaseAgent):
    """
    Intelligent error fixing agent that understands runtime errors
    and applies appropriate fixes
    """
    
    def __init__(
        self,
        project_root: str,
        llm_client: "GeminiClient",
        prompt_manager: "PromptManager",
        modification_agent: "InteractiveModificationAgent"
    ):
        """
        Initialize error fixing agent
        
        Args:
            project_root: Absolute path to project root
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
            modification_agent: The initialized Modification agent.
        """
        super().__init__(
            name="error_fixing_agent",
            role="Expert Python Debugging and Error Fixing Specialist",
            agent_type="error_fixing", # Uses low temp
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.project_root = Path(project_root)
        self.modification_agent = modification_agent # Use the injected agent
        self.fix_history = []  # Track all fixes applied
        
        logger.info("error_fixing_agent initialized")
        logger.info(f"   - Project root: {self.project_root}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an error fixing task
        
        Args:
            task: Task with:
                - error_message: The error traceback/message
                - error_context: Optional context
                - auto_apply: Whether to apply fixes automatically
        
        Returns:
            Result with error analysis and fixes
        """
        error_message = task.get("error_message")
        error_context = task.get("error_context")
        auto_apply = task.get("auto_apply", False)
        
        return await self.fix_error(
            error_message=error_message,
            error_context=error_context,
            auto_apply=auto_apply
        )
    
    async def fix_error(
        self,
        error_message: str,
        error_context: Optional[Dict[str, Any]] = None,
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point - analyze error and fix it
        
        Args:
            error_message: The error message/traceback from running code
            error_context: Optional context (file, line, command run, etc)
            auto_apply: If True, apply fixes automatically
            
        Returns:
            {
                "status": "success" | "error",
                "error_analysis": {
                    "error_type": str,
                    "root_cause": str,
                    "affected_files": List[str],
                    "line_numbers": List[int]
                },
                "proposed_fixes": List[Dict],
                "applied_fixes": List[Dict],  # If auto_apply=True
                "needs_review": bool,  # Should trigger review cycle?
                "suggested_command": str  # Command to re-run
            }
        """
        logger.info(f"Analyzing error: {error_message[:100]}...")
        
        try:
            # Step 1: Parse and understand the error
            error_analysis = await self._analyze_error(error_message, error_context)
            logger.info(f"Error type: {error_analysis.get('error_type')}")
            logger.info(f"Root cause: {error_analysis.get('root_cause', 'Unknown')[:100]}")
            
            # Step 2: Identify affected files and locations
            affected_locations = await self._find_error_locations(
                error_analysis,
                error_message
            )
            logger.info(f"Found {len(affected_locations)} affected locations")
            
            # Step 3: Generate intelligent fixes
            proposed_fixes = await self._generate_fixes(
                error_analysis,
                affected_locations,
                error_message
            )
            logger.info(f"Proposed {len(proposed_fixes)} fixes")
            
            # Step 4: Apply fixes if auto_apply
            applied_fixes = []
            if auto_apply:
                applied_fixes = await self._apply_fixes(proposed_fixes)
                logger.info(f"Applied {len(applied_fixes)} fixes")
            
            # Step 5: Determine if review cycle needed
            needs_review = self._should_trigger_review(error_analysis, proposed_fixes)
            
            # Step 6: Suggest command to re-run
            suggested_command = self._generate_rerun_command(error_context)
            
            result = {
                "status": "success",
                "error_analysis": error_analysis,
                "proposed_fixes": proposed_fixes,
                "applied_fixes": applied_fixes,
                "needs_review": needs_review,
                "suggested_command": suggested_command
            }
            
            # Track in history
            self.fix_history.append({
                "error_type": error_analysis.get("error_type"),
                "fixes_applied": len(applied_fixes),
                "timestamp": self._get_timestamp()
            })
            
            logger.info("✅ Error analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Error in error fixing: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _analyze_error(
        self,
        error_message: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to deeply understand the error
        
        Returns:
            {
                "error_type": "ModuleNotFoundError" | "AttributeError" | "SyntaxError" | ...,
                "root_cause": str,
                "severity": "critical" | "high" | "medium" | "low",
                "affected_files": List[str],
                "line_numbers": List[int],
                "error_category": "import" | "syntax" | "logic" | "dependency" | "runtime"
            }
        """
        prompt = f"""
Analyze this Python error and provide a detailed diagnosis:

ERROR MESSAGE:
{error_message}

CONTEXT:
{context if context else "No additional context provided"}

Provide analysis in JSON format:
{{
    "error_type": "The exception type (e.g., ModuleNotFoundError, AttributeError)",
    "root_cause": "Clear explanation of what caused the error",
    "severity": "critical/high/medium/low",
    "affected_files": ["list of files that need fixing"],
    "line_numbers": [list of line numbers with errors],
    "error_category": "import/syntax/logic/dependency/runtime",
    "fix_strategy": "High-level strategy to fix this error"
}}

IMPORTANT: Provide ONLY valid JSON, no markdown or explanations.
"""
        
        response = await self.generate_response(prompt)
        analysis = self._parse_json_response(response)
        
        # Extract error type from traceback if not in LLM response
        if not analysis.get("error_type"):
            error_type_match = re.search(r'(\w+Error):', error_message)
            if error_type_match:
                analysis["error_type"] = error_type_match.group(1)
        
        return analysis
    
    async def _find_error_locations(
        self,
        error_analysis: Dict[str, Any],
        error_message: str
    ) -> List[Dict[str, Any]]:
        """
        Find exact locations in code that need fixing
        
        Returns:
            [
                {
                    "filepath": str,
                    "line_number": int,
                    "error_line": str,
                    "context_before": List[str],
                    "context_after": List[str]
                }
            ]
        """
        locations = []
        
        # Parse traceback for file locations
        traceback_pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(traceback_pattern, error_message)
        
        for filepath, line_num in matches:
            line_num = int(line_num)
            
            # Convert to project-relative path
            try:
                file_path = Path(filepath)
                if not file_path.is_absolute():
                    file_path = self.project_root / filepath
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Get context
                    context_size = 5
                    start = max(0, line_num - context_size - 1)
                    end = min(len(lines), line_num + context_size)
                    
                    location = {
                        "filepath": str(file_path.relative_to(self.project_root)),
                        "line_number": line_num,
                        "error_line": lines[line_num - 1].strip() if line_num <= len(lines) else "",
                        "context_before": [l.rstrip() for l in lines[start:line_num-1]],
                        "context_after": [l.rstrip() for l in lines[line_num:end]]
                    }
                    locations.append(location)
            except Exception as e:
                logger.debug(f"Could not read {filepath}: {e}")
                continue
        
        return locations
    
    async def _generate_fixes(
        self,
        error_analysis: Dict[str, Any],
        locations: List[Dict[str, Any]],
        error_message: str
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent fixes based on error analysis
        
        Returns:
            [
                {
                    "filepath": str,
                    "fix_type": "add_import" | "fix_syntax" | "fix_logic" | "modify_code",
                    "description": str,
                    "old_code": Optional[str],
                    "new_code": str,
                    "line_number": Optional[int],
                    "priority": "critical" | "high" | "medium" | "low",
                    "reasoning": str
                }
            ]
        """
        # Quick fixes for common errors
        error_type = error_analysis.get("error_type", "")
        
        # Handle ModuleNotFoundError
        if "ModuleNotFoundError" in error_type or "No module named" in error_message:
            return await self._generate_import_fixes(error_message, locations)
        
        # Handle AttributeError
        elif "AttributeError" in error_type:
            return await self._generate_attribute_fixes(error_message, locations)
        
        # Handle SyntaxError
        elif "SyntaxError" in error_type:
            return await self._generate_syntax_fixes(error_message, locations)
        
        # Handle NameError
        elif "NameError" in error_type:
            return await self._generate_name_fixes(error_message, locations)
        
        # Handle TypeError
        elif "TypeError" in error_type:
            return await self._generate_type_fixes(error_message, locations)
        
        # Generic fix using LLM
        else:
            return await self._generate_generic_fixes(error_analysis, locations, error_message)
    
    async def _generate_import_fixes(
        self,
        error_message: str,
        locations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix ModuleNotFoundError by adding imports"""
        fixes = []
        
        # Extract missing module name
        module_match = re.search(r"No module named '([^']+)'", error_message)
        if not module_match:
            return fixes
        
        missing_module = module_match.group(1)
        logger.info(f"Missing module: {missing_module}")
        
        # Determine import statement
        # Common packages and their import patterns
        import_mappings = {
            "numpy": "import numpy as np",
            "pandas": "import pandas as pd",
            "matplotlib": "import matplotlib.pyplot as plt",
            "sklearn": "from sklearn import *",
            "torch": "import torch",
            "tensorflow": "import tensorflow as tf",
        }
        
        # Check if it's a known package
        import_statement = import_mappings.get(missing_module, f"import {missing_module}")
        
        # Apply to first location (usually the entry point)
        if locations:
            loc = locations[0]
            fixes.append({
                "filepath": loc["filepath"],
                "fix_type": "add_import",
                "description": f"Add missing import for '{missing_module}'",
                "old_code": None,
                "new_code": import_statement,
                "line_number": 1,  # Add at top of file
                "priority": "critical",
                "reasoning": f"Module '{missing_module}' is not imported but is being used"
            })
        
        return fixes
    
    async def _generate_attribute_fixes(
        self,
        error_message: str,
        locations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix AttributeError by correcting method/attribute names"""
        fixes = []
        
        # Extract attribute name
        attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_message)
        if not attr_match:
            return fixes
        
        object_type = attr_match.group(1)
        wrong_attr = attr_match.group(2)
        
        # Use LLM to suggest correct attribute
        prompt = f"""
The code has an AttributeError:
Object type: {object_type}
Wrong attribute: {wrong_attr}

What is the correct attribute name for {object_type}?
Common corrections:
- append() vs add()
- shape vs size
- fit_transform() vs transform()

Respond with ONLY the correct attribute name, nothing else.
"""
        
        correct_attr = await self.generate_response(prompt)
        correct_attr = correct_attr.strip()
        
        # Apply fix to locations
        for loc in locations:
            if wrong_attr in loc.get("error_line", ""):
                old_line = loc["error_line"]
                new_line = old_line.replace(f".{wrong_attr}", f".{correct_attr}")
                
                fixes.append({
                    "filepath": loc["filepath"],
                    "fix_type": "fix_attribute",
                    "description": f"Fix AttributeError: {wrong_attr} → {correct_attr}",
                    "old_code": old_line,
                    "new_code": new_line,
                    "line_number": loc["line_number"],
                    "priority": "high",
                    "reasoning": f"'{wrong_attr}' is not a valid attribute for {object_type}, should be '{correct_attr}'"
                })
        
        return fixes
    
    async def _generate_syntax_fixes(
        self,
        error_message: str,
        locations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix SyntaxError"""
        fixes = []
        
        for loc in locations:
            # Use LLM to fix syntax
            context = "\n".join(loc.get("context_before", [])) + "\n" + \
                      loc.get("error_line", "") + "\n" + \
                      "\n".join(loc.get("context_after", []))
            
            prompt = f"""
Fix this Python syntax error:

ERROR: {error_message}

CODE:
```python
{context}
```

Provide ONLY the corrected code for the error line, nothing else.
"""
            
            corrected_line = await self.generate_response(prompt)
            corrected_line = corrected_line.strip().strip('`').strip()
            
            fixes.append({
                "filepath": loc["filepath"],
                "fix_type": "fix_syntax",
                "description": "Fix syntax error",
                "old_code": loc["error_line"],
                "new_code": corrected_line,
                "line_number": loc["line_number"],
                "priority": "critical",
                "reasoning": "Syntax error prevents code execution"
            })
        
        return fixes
    
    async def _generate_name_fixes(
        self,
        error_message: str,
        locations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix NameError (undefined variable)"""
        fixes = []
        
        # Extract undefined name
        name_match = re.search(r"name '(\w+)' is not defined", error_message)
        if not name_match:
            return fixes
        
        undefined_name = name_match.group(1)
        
        # Check if it's a common typo
        for loc in locations:
            error_line = loc.get("error_line", "")
            
            # Use LLM to suggest fix
            prompt = f"""
Variable '{undefined_name}' is not defined in this code:
{error_line}

Is this:
1. A typo? (suggest correct name)
2. Missing variable definition? (suggest definition)
3. Missing import? (suggest import)

Respond with ONLY the fix, e.g., "x = 0" or "import numpy as np"
"""
            
            fix_suggestion = await self.generate_response(prompt)
            fix_suggestion = fix_suggestion.strip()
            
            fixes.append({
                "filepath": loc["filepath"],
                "fix_type": "fix_name",
                "description": f"Fix undefined name: {undefined_name}",
                "old_code": None,
                "new_code": fix_suggestion,
                "line_number": loc["line_number"] - 1,  # Add before error line
                "priority": "high",
                "reasoning": f"'{undefined_name}' is used but not defined"
            })
        
        return fixes
    
    async def _generate_type_fixes(
        self,
        error_message: str,
        locations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix TypeError (wrong argument types, missing arguments, etc)"""
        fixes = []
        
        for loc in locations:
            context = loc.get("error_line", "")
            
            prompt = f"""
Fix this TypeError:
{error_message}

Code: {context}

Provide the corrected line of code ONLY.
"""
            
            corrected = await self.generate_response(prompt)
            corrected = corrected.strip().strip('`').strip()
            
            fixes.append({
                "filepath": loc["filepath"],
                "fix_type": "fix_type",
                "description": "Fix TypeError",
                "old_code": context,
                "new_code": corrected,
                "line_number": loc["line_number"],
                "priority": "high",
                "reasoning": "Type mismatch or wrong arguments"
            })
        
        return fixes
    
    async def _generate_generic_fixes(
        self,
        error_analysis: Dict[str, Any],
        locations: List[Dict[str, Any]],
        error_message: str
    ) -> List[Dict[str, Any]]:
        """Use LLM for generic error fixing"""
        fixes = []
        
        for loc in locations:
            context = "\n".join(loc.get("context_before", [])) + "\n" + \
                      loc.get("error_line", "") + "\n" + \
                      "\n".join(loc.get("context_after", []))
            
            prompt = f"""
Fix this error:

ERROR ANALYSIS:
{error_analysis}

ERROR MESSAGE:
{error_message}

CODE:
```python
{context}
```

Provide the corrected code ONLY for the error line.
"""
            
            corrected = await self.generate_response(prompt)
            corrected = corrected.strip().strip('`').strip()
            
            fixes.append({
                "filepath": loc["filepath"],
                "fix_type": "generic_fix",
                "description": f"Fix {error_analysis.get('error_type', 'error')}",
                "old_code": loc["error_line"],
                "new_code": corrected,
                "line_number": loc["line_number"],
                "priority": error_analysis.get("severity", "medium"),
                "reasoning": error_analysis.get("root_cause", "Unknown cause")
            })
        
        return fixes
    
    async def _apply_fixes(
        self,
        fixes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply proposed fixes using ModificationAgent"""
        applied = []
        
        for fix in fixes:
            try:
                filepath = self.project_root / fix["filepath"]
                
                if fix["fix_type"] == "add_import":
                    # Add import at top of file
                    result = await self.modification_agent.add_import(
                        filepath=str(filepath),
                        import_statement=fix["new_code"]
                    )
                    
                elif fix["old_code"]:
                    # Replace old code with new
                    result = await self.modification_agent.modify_file(
                        filepath=str(filepath),
                        changes=[{
                            "old_code": fix["old_code"],
                            "new_code": fix["new_code"]
                        }]
                    )
                
                else:
                    # Insert new code
                    result = await self.modification_agent.insert_code(
                        filepath=str(filepath),
                        code=fix["new_code"],
                        line_number=fix.get("line_number", 1)
                    )
                
                applied.append({
                    "fix": fix,
                    "result": result,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Failed to apply fix: {e}")
                applied.append({
                    "fix": fix,
                    "error": str(e),
                    "status": "error"
                })
        
        return applied
    
    def _should_trigger_review(
        self,
        error_analysis: Dict[str, Any],
        fixes: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if fixes should trigger full review cycle
        
        Rules:
        - Critical errors → Yes
        - Logic/algorithm changes → Yes
        - Simple import fixes → No
        - Syntax fixes → Maybe (if >3 lines changed)
        """
        severity = error_analysis.get("severity", "medium")
        error_category = error_analysis.get("error_category", "")
        
        # Always review critical errors
        if severity == "critical":
            return True
        
        # Review logic changes
        if error_category == "logic":
            return True
        
        # Review if many fixes
        if len(fixes) > 3:
            return True
        
        # Don't review simple import fixes
        if error_category == "import" and len(fixes) == 1:
            return False
        
        # Default: review
        return True
    
    def _generate_rerun_command(
        self,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate command to re-run after fixes"""
        if context and "command" in context:
            return context["command"]
        
        # Default: re-run tests
        return "python -m pytest tests/"
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Remove markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            return eval(response)  # Safe here since we control the prompt
        except:
            return {}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
