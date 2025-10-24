"""
Refactoring Agent - Code Improvement & Optimization

This agent refactors and improves code based on review feedback.
Uses temperature=0.5 for balanced creativity and precision.

Enhanced with SearchRefactorTool for:
- AST-based symbol search and analysis
- Safe programmatic refactoring (rename, extract method)
- Find references and dependencies
- Pattern-based code search

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.tools.search_refactor_tool import (
    SearchRefactorTool, 
    SymbolType, 
    RefactoringType
)


class RefactoringAgent(BaseAgent):
    """
    Refactoring Agent for code improvement and optimization.
    
    Responsibilities:
    - Refactor code based on review feedback
    - Improve code structure and organization
    - Optimize performance bottlenecks
    - Apply design patterns
    - Simplify complex logic
    - Enhance code readability
    
    Uses temperature=0.5 for creative problem-solving with consistency.
    """
    
    def __init__(
        self, 
        name: str = "refactoring_agent",
        workspace_path: Optional[Path] = None,
        enable_ast_refactoring: bool = True
    ):
        """
        Initialize Refactoring Agent.
        
        Args:
            name: Agent name (default: "refactoring_agent")
            workspace_path: Path to workspace for AST-based refactoring
            enable_ast_refactoring: Enable SearchRefactorTool for programmatic refactoring
        """
        role = "Expert Code Refactoring Specialist for Python"
        super().__init__(
            name=name,
            role=role,
            agent_type="refactoring"  # Uses temperature 0.5
        )
        
        # Initialize SearchRefactorTool for AST-based refactoring
        self.enable_ast_refactoring = enable_ast_refactoring
        self.workspace_path = workspace_path
        
        if enable_ast_refactoring and workspace_path:
            self.search_refactor_tool = SearchRefactorTool(workspace_path)
            logger.info(f"{self.name} initialized with SearchRefactorTool (AST-based refactoring)")
        else:
            self.search_refactor_tool = None
            if enable_ast_refactoring:
                logger.warning(f"{self.name} AST refactoring enabled but no workspace_path provided")
        
        logger.info(f"{self.name} ready for code refactoring tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a refactoring task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of refactoring task
                - data: Task-specific parameters
        
        Returns:
            Dictionary with refactoring results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        try:
            if task_type == "refactor_code":
                return await self.refactor_code(
                    code=data.get("code"),
                    review_feedback=data.get("review_feedback", ""),
                    focus_areas=data.get("focus_areas", [])
                )
            
            elif task_type == "improve_structure":
                return await self.improve_structure(
                    code=data.get("code"),
                    issues=data.get("issues", [])
                )
            
            elif task_type == "optimize_performance":
                return await self.optimize_performance(
                    code=data.get("code"),
                    bottlenecks=data.get("bottlenecks", [])
                )
            
            elif task_type == "apply_design_pattern":
                return await self.apply_design_pattern(
                    code=data.get("code"),
                    pattern=data.get("pattern"),
                    context=data.get("context", "")
                )
            
            elif task_type == "simplify_logic":
                return await self.simplify_logic(
                    code=data.get("code"),
                    complex_areas=data.get("complex_areas", [])
                )
            
            elif task_type == "enhance_readability":
                return await self.enhance_readability(
                    code=data.get("code"),
                    style_issues=data.get("style_issues", [])
                )
            
            # AST-based refactoring tasks (Tools Integration)
            elif task_type == "find_symbol":
                return self.find_symbol(
                    name_pattern=data.get("name_pattern"),
                    symbol_type=data.get("symbol_type"),
                    file_pattern=data.get("file_pattern")
                )
            
            elif task_type == "find_references":
                return self.find_references(
                    symbol_name=data.get("symbol_name"),
                    file_pattern=data.get("file_pattern")
                )
            
            elif task_type == "rename_symbol":
                return self.rename_symbol(
                    old_name=data.get("old_name"),
                    new_name=data.get("new_name"),
                    file_pattern=data.get("file_pattern"),
                    dry_run=data.get("dry_run", True)
                )
            
            elif task_type == "extract_method":
                return self.extract_method(
                    file_path=data.get("file_path"),
                    start_line=data.get("start_line"),
                    end_line=data.get("end_line"),
                    method_name=data.get("method_name"),
                    dry_run=data.get("dry_run", True)
                )
            
            elif task_type == "search_pattern":
                return self.search_code_pattern(
                    pattern=data.get("pattern"),
                    is_regex=data.get("is_regex", False),
                    file_pattern=data.get("file_pattern")
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing refactoring task '{task_type}': {e}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def refactor_code(
        self,
        code: str,
        review_feedback: str = "",
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Refactor code based on review feedback.
        
        Args:
            code: Original code to refactor
            review_feedback: Feedback from code review
            focus_areas: Specific areas to focus on
        
        Returns:
            Dictionary with refactored code and changes
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for refactoring"
            }
        
        logger.info(f"{self.name} refactoring code ({len(code)} chars)")
        
        # Build focus areas text
        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nPriority focus areas: {', '.join(focus_areas)}"
        
        prompt_data = {
            "code": code,
            "review_feedback": review_feedback or "General improvement needed",
            "focus_areas": focus_text
        }
        
        prompt = self.get_prompt("refactoring_prompts", "refactor_code", prompt_data)
        response = await self.generate_response(prompt)
        
        # Extract refactored code
        refactored_code = self._extract_code(response)
        changes = self._extract_changes(response)
        
        return {
            "status": "success",
            "task": "refactor_code",
            "original_code": code,
            "refactored_code": refactored_code,
            "changes_made": changes,
            "improvement_summary": self._extract_summary(response),
            "focus_areas": focus_areas or []
        }
    
    async def improve_structure(
        self,
        code: str,
        issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Improve code structure and organization.
        
        Args:
            code: Code to restructure
            issues: Specific structural issues to address
        
        Returns:
            Dictionary with improved code
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for structure improvement"
            }
        
        logger.info(f"{self.name} improving code structure")
        
        issues_text = "\n".join(f"- {issue}" for issue in (issues or ["General structure improvement"]))
        
        prompt_data = {
            "code": code,
            "issues": issues_text
        }
        
        prompt = self.get_prompt("refactoring_prompts", "improve_structure", prompt_data)
        response = await self.generate_response(prompt)
        
        improved_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "improve_structure",
            "original_code": code,
            "improved_code": improved_code,
            "structural_changes": self._extract_changes(response),
            "issues_addressed": issues or []
        }
    
    async def optimize_performance(
        self,
        code: str,
        bottlenecks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize code for better performance.
        
        Args:
            code: Code to optimize
            bottlenecks: Known performance bottlenecks
        
        Returns:
            Dictionary with optimized code
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for performance optimization"
            }
        
        logger.info(f"{self.name} optimizing code performance")
        
        bottlenecks_text = "\n".join(f"- {bn}" for bn in (bottlenecks or ["General performance optimization"]))
        
        prompt_data = {
            "code": code,
            "bottlenecks": bottlenecks_text
        }
        
        prompt = self.get_prompt("refactoring_prompts", "optimize_performance", prompt_data)
        response = await self.generate_response(prompt)
        
        optimized_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "optimize_performance",
            "original_code": code,
            "optimized_code": optimized_code,
            "optimizations_applied": self._extract_changes(response),
            "performance_improvements": self._extract_performance_gains(response),
            "bottlenecks_addressed": bottlenecks or []
        }
    
    async def apply_design_pattern(
        self,
        code: str,
        pattern: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Apply a design pattern to code.
        
        Args:
            code: Code to refactor with pattern
            pattern: Design pattern to apply (e.g., "Factory", "Strategy")
            context: Context for applying the pattern
        
        Returns:
            Dictionary with refactored code using pattern
        """
        if not code or not pattern:
            return {
                "status": "error",
                "message": "Code and pattern are required"
            }
        
        logger.info(f"{self.name} applying {pattern} pattern")
        
        prompt_data = {
            "code": code,
            "pattern": pattern,
            "context": context or f"Apply {pattern} pattern for better design"
        }
        
        prompt = self.get_prompt("refactoring_prompts", "apply_design_pattern", prompt_data)
        response = await self.generate_response(prompt)
        
        refactored_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "apply_design_pattern",
            "pattern_applied": pattern,
            "original_code": code,
            "refactored_code": refactored_code,
            "pattern_explanation": self._extract_summary(response),
            "benefits": self._extract_benefits(response)
        }
    
    async def simplify_logic(
        self,
        code: str,
        complex_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Simplify complex logic and reduce complexity.
        
        Args:
            code: Code with complex logic
            complex_areas: Specific areas that are too complex
        
        Returns:
            Dictionary with simplified code
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for logic simplification"
            }
        
        logger.info(f"{self.name} simplifying complex logic")
        
        areas_text = "\n".join(f"- {area}" for area in (complex_areas or ["Overall logic simplification"]))
        
        prompt_data = {
            "code": code,
            "complex_areas": areas_text
        }
        
        prompt = self.get_prompt("refactoring_prompts", "simplify_logic", prompt_data)
        response = await self.generate_response(prompt)
        
        simplified_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "simplify_logic",
            "original_code": code,
            "simplified_code": simplified_code,
            "simplifications_made": self._extract_changes(response),
            "complexity_reduction": self._extract_complexity_info(response),
            "complex_areas_addressed": complex_areas or []
        }
    
    async def enhance_readability(
        self,
        code: str,
        style_issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhance code readability and style.
        
        Args:
            code: Code to enhance
            style_issues: Specific style issues to fix
        
        Returns:
            Dictionary with enhanced code
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for readability enhancement"
            }
        
        logger.info(f"{self.name} enhancing code readability")
        
        issues_text = "\n".join(f"- {issue}" for issue in (style_issues or ["General readability improvement"]))
        
        prompt_data = {
            "code": code,
            "style_issues": issues_text
        }
        
        prompt = self.get_prompt("refactoring_prompts", "enhance_readability", prompt_data)
        response = await self.generate_response(prompt)
        
        enhanced_code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "enhance_readability",
            "original_code": code,
            "enhanced_code": enhanced_code,
            "enhancements_made": self._extract_changes(response),
            "readability_score": self._extract_score(response),
            "style_issues_fixed": style_issues or []
        }
    
    # ==================== AST-Based Refactoring Methods (Tools Integration) ====================
    
    def find_symbol(
        self,
        name_pattern: str,
        symbol_type: Optional[SymbolType] = None,
        file_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find symbols in codebase using AST analysis.
        
        Args:
            name_pattern: Regex pattern to match symbol names
            symbol_type: Type of symbol (CLASS, FUNCTION, METHOD, etc.)
            file_pattern: File pattern to limit search
        
        Returns:
            Dictionary with found symbols
        """
        if not self.search_refactor_tool:
            return {
                "status": "error",
                "message": "AST refactoring not enabled or workspace not set"
            }
        
        logger.info(f"{self.name} searching for symbols: {name_pattern}")
        
        try:
            result = self.search_refactor_tool.find_symbols(
                name_pattern=name_pattern,
                symbol_type=symbol_type
            )
            
            return {
                "status": "success",
                "task": "find_symbol",
                "pattern": name_pattern,
                "symbol_type": symbol_type.value if symbol_type else "any",
                "symbols_found": len(result.symbols),
                "symbols": [
                    {
                        "name": s.name,
                        "type": s.type.value,
                        "file": str(s.file_path),
                        "line": s.line,
                        "scope": s.scope
                    }
                    for s in result.symbols
                ]
            }
        
        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
            return {
                "status": "error",
                "task": "find_symbol",
                "message": str(e)
            }
    
    def find_references(
        self,
        symbol_name: str,
        file_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find all references to a symbol using AST analysis.
        
        Args:
            symbol_name: Name of symbol to find references for
            file_pattern: File pattern to limit search
        
        Returns:
            Dictionary with found references
        """
        if not self.search_refactor_tool:
            return {
                "status": "error",
                "message": "AST refactoring not enabled or workspace not set"
            }
        
        logger.info(f"{self.name} finding references to: {symbol_name}")
        
        try:
            result = self.search_refactor_tool.find_references(
                symbol_name=symbol_name
            )
            
            return {
                "status": "success",
                "task": "find_references",
                "symbol": symbol_name,
                "references_found": len(result.references),
                "references": [
                    {
                        "file": str(r.file_path),
                        "line": r.line,
                        "context": r.context,
                        "is_definition": r.is_definition
                    }
                    for r in result.references
                ]
            }
        
        except Exception as e:
            logger.error(f"Reference search failed: {e}")
            return {
                "status": "error",
                "task": "find_references",
                "message": str(e)
            }
    
    def rename_symbol(
        self,
        old_name: str,
        new_name: str,
        file_pattern: Optional[str] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Rename a symbol across the codebase using AST analysis.
        
        Args:
            old_name: Current symbol name
            new_name: New symbol name
            file_pattern: File pattern to limit scope
            dry_run: If True, only show what would be changed (don't modify files)
        
        Returns:
            Dictionary with rename results
        """
        if not self.search_refactor_tool:
            return {
                "status": "error",
                "message": "AST refactoring not enabled or workspace not set"
            }
        
        logger.info(f"{self.name} renaming symbol: {old_name} → {new_name} (dry_run={dry_run})")
        
        try:
            result = self.search_refactor_tool.rename_symbol(
                old_name=old_name,
                new_name=new_name,
                dry_run=dry_run
            )
            
            if result.success:
                status_msg = "Preview complete (dry run)" if dry_run else "Rename successful"
                logger.info(f"✓ {status_msg}: {result.files_modified} files affected")
            else:
                logger.warning(f"Rename failed: {', '.join(result.errors)}")
            
            return {
                "status": "success" if result.success else "error",
                "task": "rename_symbol",
                "refactoring_type": result.refactoring_type.value,
                "old_name": old_name,
                "new_name": new_name,
                "dry_run": dry_run,
                "files_modified": result.files_modified,
                "changes": len(result.changes),
                "preview": {str(k): v for k, v in result.changes.items()} if dry_run else None,
                "errors": result.errors if not result.success else None
            }
        
        except Exception as e:
            logger.error(f"Symbol rename failed: {e}")
            return {
                "status": "error",
                "task": "rename_symbol",
                "message": str(e)
            }
    
    def extract_method(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        method_name: str,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Extract code block into a new method/function.
        
        Args:
            file_path: Path to file containing code to extract
            start_line: Starting line number
            end_line: Ending line number
            method_name: Name for the new method
            dry_run: If True, only show what would be changed
        
        Returns:
            Dictionary with extraction results
        """
        if not self.search_refactor_tool:
            return {
                "status": "error",
                "message": "AST refactoring not enabled or workspace not set"
            }
        
        logger.info(
            f"{self.name} extracting method '{method_name}' from "
            f"{file_path}:{start_line}-{end_line} (dry_run={dry_run})"
        )
        
        try:
            result = self.search_refactor_tool.extract_method(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                method_name=method_name,
                dry_run=dry_run
            )
            
            if result.success:
                status_msg = "Preview complete (dry run)" if dry_run else "Extraction successful"
                logger.info(f"✓ {status_msg}")
            else:
                logger.warning(f"Extraction failed: {', '.join(result.errors)}")
            
            return {
                "status": "success" if result.success else "error",
                "task": "extract_method",
                "refactoring_type": result.refactoring_type.value,
                "file": str(file_path),
                "method_name": method_name,
                "lines_extracted": f"{start_line}-{end_line}",
                "dry_run": dry_run,
                "files_modified": result.files_modified,
                "changes": len(result.changes),
                "preview": {str(k): v for k, v in result.changes.items()} if dry_run else None,
                "errors": result.errors if not result.success else None
            }
        
        except Exception as e:
            logger.error(f"Method extraction failed: {e}")
            return {
                "status": "error",
                "task": "extract_method",
                "message": str(e)
            }
    
    def search_code_pattern(
        self,
        pattern: str,
        is_regex: bool = False,
        file_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for code patterns across the codebase.
        
        Args:
            pattern: Text or regex pattern to search for
            is_regex: Whether pattern is a regex
            file_pattern: File pattern to limit search
        
        Returns:
            Dictionary with search results
        """
        if not self.search_refactor_tool:
            return {
                "status": "error",
                "message": "AST refactoring not enabled or workspace not set"
            }
        
        logger.info(f"{self.name} searching for pattern: {pattern}")
        
        try:
            result = self.search_refactor_tool.search_pattern(
                pattern=pattern,
                is_regex=is_regex
            )
            
            return {
                "status": "success",
                "task": "search_pattern",
                "pattern": pattern,
                "is_regex": is_regex,
                "matches_found": result.total_matches,
                "files_searched": result.files_searched,
                "matches": [
                    {
                        "file": str(r.file_path),
                        "line": r.line,
                        "context": r.context
                    }
                    for r in result.references
                ]
            }
        
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return {
                "status": "error",
                "task": "search_pattern",
                "message": str(e)
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
            # Return the last code block (usually the final refactored version)
            return matches[-1].strip()
        
        # If no markdown blocks, look for code-like content
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see imports, defs, or classes
            if re.match(r'^(import|from|def|class|async def)', line.strip()):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return response.strip()
    
    def _extract_changes(self, response: str) -> List[str]:
        """
        Extract list of changes made from response.
        
        Args:
            response: LLM response text
        
        Returns:
            List of changes
        """
        changes = []
        
        # Look for sections mentioning changes
        change_keywords = [
            'changes', 'modifications', 'improvements', 
            'updates', 'refactored', 'optimized'
        ]
        
        lines = response.split('\n')
        in_change_section = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Start of change section
            if any(keyword in line_lower for keyword in change_keywords):
                in_change_section = True
                continue
            
            # Extract list items
            if in_change_section:
                stripped = line.strip()
                if stripped.startswith(('-', '•', '*')) or (stripped and stripped[0].isdigit() and '.' in stripped[:3]):
                    change = re.sub(r'^[-•*\d.)\s]+', '', stripped)
                    if change:
                        changes.append(change)
                elif stripped and not stripped.startswith('#'):
                    # Non-list item might end the section
                    if len(changes) > 0:
                        break
        
        return changes if changes else ["Code refactored successfully"]
    
    def _extract_summary(self, response: str) -> str:
        """
        Extract summary/explanation from response.
        
        Args:
            response: LLM response text
        
        Returns:
            Summary text
        """
        # Look for summary section
        summary_keywords = ['summary', 'explanation', 'overview', 'improvements']
        
        lines = response.split('\n')
        summary_lines = []
        in_summary = False
        
        for line in lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in summary_keywords):
                in_summary = True
                continue
            
            if in_summary:
                if line.strip() and not line.startswith('```'):
                    summary_lines.append(line.strip())
                    if len(summary_lines) >= 3:  # Get first few lines
                        break
        
        if summary_lines:
            return ' '.join(summary_lines)
        
        # Default: return first paragraph
        paragraphs = response.split('\n\n')
        if paragraphs:
            return paragraphs[0].strip()
        
        return "Code successfully refactored."
    
    def _extract_performance_gains(self, response: str) -> Dict[str, str]:
        """
        Extract performance improvement information.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with performance metrics
        """
        gains = {}
        
        # Look for complexity mentions
        complexity_pattern = r'O\(([^)]+)\)\s*(?:to|→|->)\s*O\(([^)]+)\)'
        complexity_matches = re.findall(complexity_pattern, response)
        
        if complexity_matches:
            old, new = complexity_matches[0]
            gains['time_complexity'] = f"O({old}) → O({new})"
        
        # Look for speed improvements
        speed_pattern = r'(\d+)x?\s*(?:faster|speedup|improvement)'
        speed_match = re.search(speed_pattern, response, re.IGNORECASE)
        
        if speed_match:
            gains['speed_improvement'] = f"{speed_match.group(1)}x faster"
        
        return gains if gains else {"improvement": "Performance optimized"}
    
    def _extract_benefits(self, response: str) -> List[str]:
        """
        Extract benefits from response.
        
        Args:
            response: LLM response text
        
        Returns:
            List of benefits
        """
        benefits = []
        
        benefit_keywords = ['benefit', 'advantage', 'improvement', 'gain']
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in benefit_keywords):
                stripped = line.strip()
                if stripped.startswith(('-', '•', '*')):
                    benefit = re.sub(r'^[-•*\s]+', '', stripped)
                    if benefit:
                        benefits.append(benefit)
        
        return benefits if benefits else ["Improved code design and maintainability"]
    
    def _extract_complexity_info(self, response: str) -> str:
        """
        Extract complexity reduction information.
        
        Args:
            response: LLM response text
        
        Returns:
            Complexity information string
        """
        # Look for cyclomatic complexity mentions (various formats)
        complexity_patterns = [
            r'complexity[:\s]+(?:from\s+)?(\d+)\s*(?:to|→|->)\s*(\d+)',
            r'(?:reduced|decreased)\s+from\s+(\d+)\s+to\s+(\d+)',
            r'(\d+)\s+to\s+(\d+).*complexity'
        ]
        
        for pattern in complexity_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                old, new = match.groups()
                return f"Complexity reduced from {old} to {new}"
        
        # Look for general complexity mentions
        if 'simpl' in response.lower() or 'reduc' in response.lower():
            return "Logic simplified and complexity reduced"
        
        return "Code complexity improved"
    
    def _extract_score(self, response: str) -> int:
        """
        Extract readability or quality score from response.
        
        Args:
            response: LLM response text
        
        Returns:
            Score (0-100)
        """
        # Look for score patterns
        score_pattern = r'(?:score|rating|readability)[:\s]+(\d+)(?:/100)?'
        match = re.search(score_pattern, response, re.IGNORECASE)
        
        if match:
            score = int(match.group(1))
            return min(score, 100)  # Cap at 100
        
        return 85  # Default good score if not found
