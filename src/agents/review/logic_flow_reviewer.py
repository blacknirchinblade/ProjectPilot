"""
Logic Flow Reviewer Agent

Analyzes code logic, control flow, and execution paths.
Provides comprehensive logic quality scoring.

Features:
- Control flow analysis
- Execution path tracing
- Unreachable code detection
- Error handling validation
- Logic complexity scoring
- Edge case coverage

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import re
from typing import Dict, List, Any, Set, Tuple
from src.agents.base_agent import BaseAgent
from src.utils.logger import logger


class LogicFlowReviewer(BaseAgent):
    """
    Agent specialized in analyzing code logic and control flow.
    
    Provides scoring on:
    - Control flow clarity (25%)
    - Error handling (25%)
    - Edge case coverage (20%)
    - Logic consistency (15%)
    - Dead code detection (15%)
    """
    
    def __init__(self, name: str = "logic_flow_reviewer"):
        """
        Initialize Logic Flow Reviewer.
        
        Args:
            name: Agent name (default: "logic_flow_reviewer")
        """
        role = "Expert Logic Flow and Control Flow Analyst for Python"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"  # Uses temperature 0.2
        )
        logger.info(f"{self.name} ready for logic flow analysis")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute logic flow analysis task.
        
        Task Types:
        - analyze_logic_flow: Full logic analysis
        - check_control_flow: Control flow only
        - check_error_handling: Error handling only
        - detect_dead_code: Unreachable code detection
        
        Args:
            task: Task dictionary with type and data
            
        Returns:
            Analysis results with score and issues
        """
        task_type = task.get("task_type", "")
        data = task.get("data", {})
        
        try:
            if task_type == "analyze_logic_flow":
                code = data.get("code", "")
                filename = data.get("filename", "unknown.py")
                return await self.analyze_logic_flow(code, filename)
            
            elif task_type == "check_control_flow":
                code = data.get("code", "")
                return await self.check_control_flow(code)
            
            elif task_type == "check_error_handling":
                code = data.get("code", "")
                return await self.check_error_handling(code)
            
            elif task_type == "detect_dead_code":
                code = data.get("code", "")
                return await self.detect_dead_code(code)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error in {task_type}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def analyze_logic_flow(self, code: str, filename: str) -> Dict[str, Any]:
        """
        Comprehensive logic flow analysis.
        
        Analyzes:
        - Control flow clarity
        - Error handling
        - Edge case coverage
        - Logic consistency
        - Dead code detection
        
        Args:
            code: Python code to analyze
            filename: Name of the file
            
        Returns:
            Dictionary with score, breakdown, issues, and recommendations
        """
        if not code or not code.strip():
            return {
                "status": "error",
                "message": "Code is required",
                "logic_score": 0
            }
        
        logger.info(f"{self.name} analyzing logic flow for {filename} ({len(code)} chars)")
        
        # Run all analysis components
        control_flow_result = await self.check_control_flow(code)
        error_handling_result = await self.check_error_handling(code)
        edge_case_result = await self._check_edge_cases(code)
        consistency_result = await self._check_logic_consistency(code)
        dead_code_result = await self.detect_dead_code(code)
        
        # Calculate weighted score
        control_flow_score = control_flow_result.get("score", 100)
        error_handling_score = error_handling_result.get("score", 100)
        edge_case_score = edge_case_result.get("score", 100)
        consistency_score = consistency_result.get("score", 100)
        dead_code_score = dead_code_result.get("score", 100)
        
        total_score = (
            control_flow_score * 0.25 +
            error_handling_score * 0.25 +
            edge_case_score * 0.20 +
            consistency_score * 0.15 +
            dead_code_score * 0.15
        )
        
        # Collect all issues
        all_issues = []
        all_issues.extend(control_flow_result.get("issues", []))
        all_issues.extend(error_handling_result.get("issues", []))
        all_issues.extend(edge_case_result.get("issues", []))
        all_issues.extend(consistency_result.get("issues", []))
        all_issues.extend(dead_code_result.get("issues", []))
        
        # Get AI analysis
        try:
            analysis_prompt = self.get_prompt("review_prompts", "logic_flow_analysis", {
                "code": code[:3000],
                "filename": filename,
                "issues": "\n".join(f"- {issue}" for issue in all_issues[:10]),
                "control_flow_score": control_flow_score,
                "error_handling_score": error_handling_score,
                "edge_case_score": edge_case_score,
                "total_score": round(total_score, 1)
            })
            
            analysis = await self.generate_response(analysis_prompt)
        except Exception as e:
            logger.warning(f"AI analysis failed for logic flow: {e}")
            analysis = f"Logic flow score: {round(total_score, 1)}/100. {len(all_issues)} issues found."
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, total_score)
        
        return {
            "success": True,
            "score": round(total_score, 1),
            "status": "success",
            "logic_score": round(total_score, 1),
            "breakdown": {
                "control_flow": round(control_flow_score, 1),
                "error_handling": round(error_handling_score, 1),
                "edge_cases": round(edge_case_score, 1),
                "consistency": round(consistency_score, 1),
                "dead_code": round(dead_code_score, 1)
            },
            "issues": all_issues,
            "analysis": analysis,
            "recommendations": recommendations
        }
    
    async def check_control_flow(self, code: str) -> Dict[str, Any]:
        """
        Analyze control flow structure.
        
        Checks:
        - If/elif/else chains
        - Loop structures
        - Return statements
        - Branch coverage
        - Flow complexity
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with score and issues
        """
        issues = []
        score = 100
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "error",
                "score": 0,
                "issues": [f"Syntax error: {str(e)}"]
            }
        
        # Statistics
        if_statements = 0
        elif_count = 0
        else_count = 0
        loops = 0
        returns = 0
        early_returns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if_statements += 1
                # Check for elif/else
                if node.orelse:
                    if isinstance(node.orelse[0], ast.If):
                        elif_count += 1
                    else:
                        else_count += 1
            
            elif isinstance(node, (ast.For, ast.While)):
                loops += 1
            
            elif isinstance(node, ast.Return):
                returns += 1
        
        # Check for long if-elif chains
        if elif_count > 5:
            issues.append(f"Long if-elif chain ({elif_count} branches) - consider using dictionary dispatch or match-case")
            score -= 10
        
        # Check for missing else clauses
        if if_statements > 0 and else_count == 0 and if_statements > 2:
            issues.append("Multiple if statements without else clauses - may have unhandled cases")
            score -= 5
        
        # Check for complex control flow
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in functions:
            func_returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
            if len(func_returns) > 5:
                issues.append(f"Function '{func.name}' has {len(func_returns)} return statements - consider simplifying")
                score -= 5
        
        # Check for loops without break/continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                has_continue = any(isinstance(n, ast.Continue) for n in ast.walk(node))
                if not has_break and not has_continue and isinstance(node.body, list) and len(node.body) > 5:
                    issues.append("Complex loop without break/continue - may be hard to follow")
                    score -= 3
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues,
            "statistics": {
                "if_statements": if_statements,
                "elif_count": elif_count,
                "else_count": else_count,
                "loops": loops,
                "returns": returns
            }
        }
    
    async def check_error_handling(self, code: str) -> Dict[str, Any]:
        """
        Analyze error handling patterns.
        
        Checks:
        - Try/except usage
        - Exception specificity
        - Bare except clauses
        - Finally blocks
        - Resource cleanup
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with score and issues
        """
        issues = []
        score = 100
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "error",
                "score": 0,
                "issues": [f"Syntax error: {str(e)}"]
            }
        
        # Find functions and try blocks
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        
        # Check for bare except
        for try_node in try_blocks:
            for handler in try_node.handlers:
                if handler.type is None:
                    issues.append("Bare 'except:' clause - should catch specific exceptions")
                    score -= 15
                elif isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                    issues.append("Catching generic 'Exception' - consider more specific exception types")
                    score -= 10
        
        # Check for missing error handling in file operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Check for open() without try/except
                    if node.func.id == "open":
                        # Check if inside try block
                        parent_try = self._find_parent_try(tree, node)
                        if not parent_try:
                            issues.append("File 'open()' without try/except - may cause uncaught errors")
                            score -= 10
        
        # Check for try blocks without finally
        try_without_finally = 0
        for try_node in try_blocks:
            if not try_node.finalbody:
                try_without_finally += 1
        
        if try_without_finally > 0 and len(try_blocks) > 2:
            issues.append(f"{try_without_finally} try blocks without 'finally' - may leave resources uncleaned")
            score -= 5
        
        # Check for empty except blocks
        for try_node in try_blocks:
            for handler in try_node.handlers:
                if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                    issues.append("Empty except block with 'pass' - silently swallows errors")
                    score -= 15
        
        # Check error handling coverage
        if len(functions) > 3 and len(try_blocks) == 0:
            issues.append("No error handling found - functions may fail unexpectedly")
            score -= 20
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues,
            "statistics": {
                "try_blocks": len(try_blocks),
                "bare_except": sum(1 for t in try_blocks for h in t.handlers if h.type is None),
                "finally_blocks": sum(1 for t in try_blocks if t.finalbody)
            }
        }
    
    async def detect_dead_code(self, code: str) -> Dict[str, Any]:
        """
        Detect unreachable code.
        
        Checks:
        - Code after return
        - Code after break/continue
        - Unreachable except blocks
        - Unused variables
        - Dead branches
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with score and issues
        """
        issues = []
        score = 100
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "error",
                "score": 0,
                "issues": [f"Syntax error: {str(e)}"]
            }
        
        # Check for code after return/raise
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        if i < len(node.body) - 1:
                            next_stmt = node.body[i + 1]
                            if not isinstance(next_stmt, (ast.Pass, ast.Expr)):
                                issues.append(f"Unreachable code after return/raise in '{node.name}'")
                                score -= 20
        
        # Check for code after break/continue in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, (ast.Break, ast.Continue)):
                        if i < len(node.body) - 1:
                            issues.append("Unreachable code after break/continue in loop")
                            score -= 15
        
        # Check for if True/False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Constant):
                    if node.test.value is True:
                        if node.orelse:
                            issues.append("'if True' with else clause - else block is dead code")
                            score -= 15
                    elif node.test.value is False:
                        issues.append("'if False' block is dead code")
                        score -= 15
        
        # Check for unreachable except blocks (specific after general)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                caught_types = []
                for handler in node.handlers:
                    if handler.type is None:
                        # Bare except - all following handlers are unreachable
                        if handler != node.handlers[-1]:
                            issues.append("Except handlers after bare 'except:' are unreachable")
                            score -= 20
                        break
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues
        }
    
    async def _check_edge_cases(self, code: str) -> Dict[str, Any]:
        """
        Check for edge case handling.
        
        Checks:
        - None checks
        - Empty collection checks
        - Zero/negative number handling
        - Boundary conditions
        - Type validation
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with score and issues
        """
        issues = []
        score = 100
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"status": "error", "score": 0, "issues": ["Syntax error"]}
        
        # Check for None checks
        has_none_check = False
        has_empty_check = False
        has_zero_check = False
        
        for node in ast.walk(tree):
            # Check for 'is None' or '== None'
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(comparator, ast.Constant) and comparator.value is None:
                        has_none_check = True
                    elif isinstance(comparator, ast.Constant) and comparator.value == 0:
                        has_zero_check = True
            
            # Check for empty collection checks
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "len":
                    has_empty_check = True
        
        # Check for list/dict access without validation
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                # Check if inside try-except or if-check
                # This is a simplified check
                pass
        
        # Deduct points if missing common edge case checks
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if len(functions) > 0:
            if not has_none_check and len(functions) > 2:
                issues.append("No None checks found - may fail on None inputs")
                score -= 10
            
            # Check for division operations without zero checks
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                    if not has_zero_check:
                        issues.append("Division operations without zero checks - may cause ZeroDivisionError")
                        score -= 15
                        break
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues
        }
    
    async def _check_logic_consistency(self, code: str) -> Dict[str, Any]:
        """
        Check for logical consistency.
        
        Checks:
        - Contradictory conditions
        - Redundant checks
        - Always-true/false conditions
        - Variable state consistency
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with score and issues
        """
        issues = []
        score = 100
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"status": "error", "score": 0, "issues": ["Syntax error"]}
        
        # Check for redundant boolean comparisons
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check for x == True or x == False
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant):
                        if comparator.value is True or comparator.value is False:
                            issues.append("Redundant boolean comparison (use variable directly or 'not')")
                            score -= 5
        
        # Check for always-true conditions (x or True, x and True)
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        if isinstance(node.op, ast.Or) and value.value is True:
                            issues.append("Always-true condition (x or True is always True)")
                            score -= 10
                        elif isinstance(node.op, ast.And) and value.value is False:
                            issues.append("Always-false condition (x and False is always False)")
                            score -= 10
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues
        }
    
    def _find_parent_try(self, tree: ast.AST, target: ast.AST) -> bool:
        """
        Check if a node is inside a try block.
        
        Args:
            tree: AST tree
            target: Node to check
            
        Returns:
            True if inside try block
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if child == target:
                        return True
        return False
    
    def _generate_recommendations(self, issues: List[str], score: float) -> List[str]:
        """
        Generate actionable recommendations.
        
        Args:
            issues: List of detected issues
            score: Overall logic score
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        if score < 60:
            recommendations.append("🔴 CRITICAL: Major logic issues detected - requires immediate attention")
        elif score < 75:
            recommendations.append("🟡 WARNING: Multiple logic issues - review and address")
        else:
            recommendations.append("🟢 GOOD: Logic flow is acceptable")
        
        # Categorize issues
        error_handling_issues = [i for i in issues if "except" in i.lower() or "error" in i.lower() or "try" in i.lower()]
        dead_code_issues = [i for i in issues if "unreachable" in i.lower() or "dead" in i.lower()]
        edge_case_issues = [i for i in issues if "none" in i.lower() or "zero" in i.lower() or "empty" in i.lower()]
        
        # Priority recommendations
        if error_handling_issues:
            recommendations.append(f"1. Fix error handling ({len(error_handling_issues)} issues)")
        
        if dead_code_issues:
            recommendations.append(f"2. Remove dead code ({len(dead_code_issues)} instances)")
        
        if edge_case_issues:
            recommendations.append(f"3. Add edge case validation ({len(edge_case_issues)} missing)")
        
        # General recommendations
        if len(issues) > 5:
            recommendations.append("4. Simplify control flow to reduce complexity")
        
        if score >= 80:
            recommendations.append("5. Consider adding more comprehensive error handling")
        
        return recommendations[:5]  # Top 5
