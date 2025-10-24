"""
Performance Reviewer Agent

Analyzes code for performance issues and optimization opportunities:
- Time complexity analysis (Big O notation)
- Space complexity analysis (memory usage)
- Bottleneck detection
- Memory leak identification
- Loop optimization opportunities
- Redundant computation detection
- Caching opportunities
- Performance best practices

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import re
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from loguru import logger

from ..base_agent import BaseAgent


class PerformanceReviewer(BaseAgent):
    """
    Specialized agent for analyzing code performance.
    
    Analyzes:
    1. Time Complexity (35%) - Big O analysis, algorithmic efficiency
    2. Space Complexity (25%) - Memory usage, data structure choices
    3. Optimization Opportunities (20%) - Redundancy, caching, vectorization
    4. Performance Best Practices (20%) - Efficient patterns, anti-patterns
    
    Returns score 0-100 with detailed performance analysis.
    """
    
    ASPECT_WEIGHTS = {
        "time_complexity": 0.35,
        "space_complexity": 0.25,
        "optimization_opportunities": 0.20,
        "performance_best_practices": 0.20
    }
    
    def __init__(self, name: str = "performance_reviewer"):
        """
        Initialize Performance Reviewer.
        
        Args:
            name: Agent name (default: "performance_reviewer")
        """
        role = "Expert Code Performance Analyst for Python"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"  # Uses temperature 0.2
        )
        logger.info(f"{self.name} ready for performance analysis")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute performance review task (async interface for BaseAgent compatibility).
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of review task (e.g., 'review_performance')
                - data: Task-specific parameters (must contain 'code')
            
        Returns:
            Dict with performance analysis
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        if task_type == "review_performance":
            return self._review_performance(data)
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    def review_performance(self, code: str) -> Dict[str, Any]:
        """
        Synchronous performance review (for direct calls and testing).
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dict with performance analysis and score
        """
        return self._review_performance({"code": code})
    
    def _review_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete performance review.
        
        Args:
            context: Must contain 'code', optionally 'project_files'
            
        Returns:
            Dict with performance score and analysis
        """
        code = context.get("code", "")
        if not code:
            return {
                "status": "error",
                "message": "No code provided for review"
            }
        
        logger.info("Starting performance review...")
        
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Analyze different aspects
            time_complexity = self._analyze_time_complexity(tree, code)
            space_complexity = self._analyze_space_complexity(tree, code)
            optimization_opps = self._analyze_optimization_opportunities(tree, code)
            best_practices = self._analyze_performance_best_practices(tree, code)
            
            # Calculate weighted score
            total_score = (
                time_complexity["score"] * self.ASPECT_WEIGHTS["time_complexity"] +
                space_complexity["score"] * self.ASPECT_WEIGHTS["space_complexity"] +
                optimization_opps["score"] * self.ASPECT_WEIGHTS["optimization_opportunities"] +
                best_practices["score"] * self.ASPECT_WEIGHTS["performance_best_practices"]
            )
            
            # Aggregate issues and suggestions
            all_issues = (
                time_complexity.get("issues", []) +
                space_complexity.get("issues", []) +
                optimization_opps.get("issues", []) +
                best_practices.get("issues", [])
            )
            
            all_suggestions = (
                time_complexity.get("suggestions", []) +
                space_complexity.get("suggestions", []) +
                optimization_opps.get("suggestions", []) +
                best_practices.get("suggestions", [])
            )
            
            result_data = {
                "score": round(total_score, 2),
                "aspects": {
                    "time_complexity": time_complexity,
                    "space_complexity": space_complexity,
                    "optimization_opportunities": optimization_opps,
                    "performance_best_practices": best_practices
                },
                "issues": all_issues,
                "suggestions": all_suggestions,
                "statistics": {
                    "total_issues": len(all_issues),
                    "critical_issues": len([i for i in all_issues if i.get("severity") == "critical"]),
                    "loops_analyzed": time_complexity.get("loops_count", 0),
                    "functions_analyzed": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                }
            }
            
            logger.info(f"Performance review complete. Score: {total_score:.2f}")
            
            return {
                "success": True,
                "status": "success",
                **result_data
            }
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            return {
                "success": True,
                "status": "success",
                "score": 0,
                "issues": [{
                    "type": "syntax_error",
                    "severity": "critical",
                    "message": f"Code has syntax errors: {str(e)}",
                    "line": getattr(e, 'lineno', None)
                }],
                "suggestions": ["Fix syntax errors before performance analysis"],
                "aspects": {},
                "statistics": {}
            }
        except Exception as e:
            logger.error(f"Error during performance review: {e}")
            return {
                "success": False,
                "status": "error",
                "score": 0,
                "message": f"Performance review failed: {str(e)}"
            }
    
    def _analyze_time_complexity(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze time complexity of code.
        
        Checks for:
        - Nested loops (O(n²), O(n³), etc.)
        - Recursive calls
        - Linear searches in loops
        - Inefficient algorithms
        
        Returns:
            Dict with score, issues, suggestions
        """
        issues = []
        suggestions = []
        score = 100
        loops_count = 0
        
        # Find nested loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loops_count += 1
                nesting_level = self._get_loop_nesting_level(node, tree)
                
                if nesting_level >= 3:
                    issues.append({
                        "type": "high_time_complexity",
                        "severity": "critical",
                        "message": f"Triple-nested loop detected (O(n³) or worse)",
                        "line": node.lineno,
                        "complexity": f"O(n^{nesting_level})"
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Consider algorithmic optimization "
                        f"to reduce {nesting_level}-level nesting"
                    )
                    score -= 25
                    
                elif nesting_level == 2:
                    issues.append({
                        "type": "moderate_time_complexity",
                        "severity": "warning",
                        "message": "Nested loop detected (O(n²))",
                        "line": node.lineno,
                        "complexity": "O(n²)"
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Evaluate if nested loop can be optimized "
                        f"(e.g., using hash tables, vectorization)"
                    )
                    score -= 10
        
        # Check for recursive functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_recursive(node):
                    # Check if memoized
                    has_memoization = self._has_memoization(node)
                    
                    if not has_memoization:
                        issues.append({
                            "type": "unmemoized_recursion",
                            "severity": "warning",
                            "message": f"Recursive function '{node.name}' without memoization",
                            "line": node.lineno,
                            "function": node.name
                        })
                        suggestions.append(
                            f"Line {node.lineno}: Consider adding memoization "
                            f"to recursive function '{node.name}' using @lru_cache"
                        )
                        score -= 8
        
        # Check for linear search in loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.Compare):
                        if self._is_linear_search_pattern(child):
                            issues.append({
                                "type": "linear_search_in_loop",
                                "severity": "warning",
                                "message": "Linear search inside loop (potential O(n²))",
                                "line": node.lineno
                            })
                            suggestions.append(
                                f"Line {node.lineno}: Replace linear search with "
                                f"hash-based lookup (set/dict) for O(1) access"
                            )
                            score -= 12
                            break
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions,
            "loops_count": loops_count,
            "analysis": f"Analyzed {loops_count} loops for time complexity"
        }
    
    def _analyze_space_complexity(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze space complexity and memory usage.
        
        Checks for:
        - Large data structure creation in loops
        - Unnecessary copies
        - Memory-inefficient patterns
        - Generator vs list opportunities
        
        Returns:
            Dict with score, issues, suggestions
        """
        issues = []
        suggestions = []
        score = 100
        
        # Check for list comprehensions that could be generators
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                # Check if list is only iterated over once
                parent = self._get_parent_node(node, tree)
                if self._is_single_iteration_context(parent):
                    issues.append({
                        "type": "inefficient_list_comprehension",
                        "severity": "info",
                        "message": "List comprehension used where generator would suffice",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Use generator expression () "
                        f"instead of list comprehension [] to save memory"
                    )
                    score -= 5
        
        # Check for unnecessary list() calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "list":
                    # Check if converting range or other iterables unnecessarily
                    if node.args and isinstance(node.args[0], ast.Call):
                        if isinstance(node.args[0].func, ast.Name):
                            if node.args[0].func.id in ["range", "map", "filter"]:
                                issues.append({
                                    "type": "unnecessary_list_conversion",
                                    "severity": "info",
                                    "message": f"Unnecessary list() conversion of {node.args[0].func.id}",
                                    "line": node.lineno
                                })
                                suggestions.append(
                                    f"Line {node.lineno}: Avoid converting "
                                    f"{node.args[0].func.id} to list if not needed"
                                )
                                score -= 3
        
        # Check for string concatenation in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign):
                        if isinstance(child.op, ast.Add):
                            if self._is_string_variable(child.target, node):
                                issues.append({
                                    "type": "string_concatenation_in_loop",
                                    "severity": "warning",
                                    "message": "String concatenation in loop (O(n²) memory)",
                                    "line": node.lineno
                                })
                                suggestions.append(
                                    f"Line {node.lineno}: Use list.append() "
                                    f"and ''.join() instead of string concatenation"
                                )
                                score -= 15
        
        # Check for copying large structures
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "copy" or node.func.attr == "deepcopy":
                        issues.append({
                            "type": "data_copy",
                            "severity": "info",
                            "message": "Data structure copy - ensure necessary",
                            "line": node.lineno
                        })
                        suggestions.append(
                            f"Line {node.lineno}: Verify that data copy is necessary, "
                            f"consider using views or references"
                        )
                        score -= 5
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions,
            "analysis": "Analyzed memory usage and space complexity"
        }
    
    def _analyze_optimization_opportunities(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Find optimization opportunities.
        
        Checks for:
        - Redundant computations
        - Caching opportunities
        - Vectorization opportunities (NumPy)
        - Loop invariant code motion
        
        Returns:
            Dict with score, issues, suggestions
        """
        issues = []
        suggestions = []
        score = 100
        
        # Check for repeated computations in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_vars = self._get_loop_variables(node)
                repeated_calls = self._find_repeated_calls_in_loop(node, loop_vars)
                
                for call_info in repeated_calls:
                    issues.append({
                        "type": "loop_invariant_computation",
                        "severity": "warning",
                        "message": f"Repeated computation in loop: {call_info['name']}",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Move '{call_info['name']}' "
                        f"outside loop (loop-invariant code motion)"
                    )
                    score -= 10
        
        # Check for missing NumPy vectorization opportunities
        has_numpy = "import numpy" in code or "from numpy" in code
        
        if not has_numpy:
            # Check for math operations in loops that could use NumPy
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    has_math_ops = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.BinOp):
                            if isinstance(child.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                                has_math_ops = True
                                break
                    
                    if has_math_ops:
                        issues.append({
                            "type": "vectorization_opportunity",
                            "severity": "info",
                            "message": "Loop with arithmetic operations - consider NumPy vectorization",
                            "line": node.lineno
                        })
                        suggestions.append(
                            f"Line {node.lineno}: Consider using NumPy arrays "
                            f"and vectorized operations for better performance"
                        )
                        score -= 8
                        break  # Only suggest once
        
        # Check for caching opportunities
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has no side effects and could benefit from caching
                if self._is_pure_function(node) and not self._has_memoization(node):
                    # Check if it's called multiple times
                    call_count = self._count_function_calls(node.name, tree)
                    
                    if call_count > 1:
                        issues.append({
                            "type": "missing_cache",
                            "severity": "info",
                            "message": f"Pure function '{node.name}' called {call_count} times without caching",
                            "line": node.lineno,
                            "function": node.name
                        })
                        suggestions.append(
                            f"Line {node.lineno}: Add @lru_cache decorator "
                            f"to '{node.name}' for automatic memoization"
                        )
                        score -= 5
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions,
            "analysis": "Analyzed optimization opportunities"
        }
    
    def _analyze_performance_best_practices(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Check adherence to performance best practices.
        
        Checks for:
        - Use of built-in functions (sum, max, min vs loops)
        - Appropriate data structures (set for membership, dict for lookup)
        - Efficient iteration patterns
        - Avoiding premature optimization
        
        Returns:
            Dict with score, issues, suggestions
        """
        issues = []
        suggestions = []
        score = 100
        
        # Check for manual sum/max/min instead of built-ins
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Look for accumulator pattern
                if self._is_manual_sum_pattern(node):
                    issues.append({
                        "type": "inefficient_builtin",
                        "severity": "info",
                        "message": "Manual sum calculation - use built-in sum()",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Use built-in sum() "
                        f"instead of manual accumulation"
                    )
                    score -= 5
                
                if self._is_manual_max_min_pattern(node):
                    issues.append({
                        "type": "inefficient_builtin",
                        "severity": "info",
                        "message": "Manual max/min calculation - use built-ins",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Use built-in max() or min() "
                        f"instead of manual comparison"
                    )
                    score -= 5
        
        # Check for inefficient membership testing (list instead of set)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        # Check if comparing against a list literal
                        for comparator in node.comparators:
                            if isinstance(comparator, ast.List):
                                if len(comparator.elts) > 5:
                                    issues.append({
                                        "type": "inefficient_membership_test",
                                        "severity": "warning",
                                        "message": "Membership test on list - use set for O(1) lookup",
                                        "line": node.lineno
                                    })
                                    suggestions.append(
                                        f"Line {node.lineno}: Convert list to set "
                                        f"for faster membership testing"
                                    )
                                    score -= 10
        
        # Check for dict.keys() in iteration (redundant)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Attribute):
                        if node.iter.func.attr == "keys":
                            issues.append({
                                "type": "redundant_dict_keys",
                                "severity": "info",
                                "message": "Unnecessary .keys() call in dict iteration",
                                "line": node.lineno
                            })
                            suggestions.append(
                                f"Line {node.lineno}: Remove .keys() - "
                                f"iterate over dict directly"
                            )
                            score -= 3
        
        # Check for using range(len()) instead of enumerate
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                        if node.iter.args:
                            if isinstance(node.iter.args[0], ast.Call):
                                if isinstance(node.iter.args[0].func, ast.Name):
                                    if node.iter.args[0].func.id == "len":
                                        issues.append({
                                            "type": "inefficient_iteration",
                                            "severity": "info",
                                            "message": "Use enumerate() instead of range(len())",
                                            "line": node.lineno
                                        })
                                        suggestions.append(
                                            f"Line {node.lineno}: Use enumerate() "
                                            f"for cleaner and more Pythonic iteration"
                                        )
                                        score -= 5
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions,
            "analysis": "Analyzed performance best practices"
        }
    
    # Helper methods
    
    def _get_loop_nesting_level(self, loop_node: ast.AST, tree: ast.AST) -> int:
        """
        Calculate nesting level of a loop.
        
        Args:
            loop_node: The loop node to check
            tree: The full AST tree
            
        Returns:
            Nesting level (1 for single loop, 2 for double-nested, etc.)
        """
        # Find parents by checking which nodes contain this loop
        def find_parent_loops(node, target, parents=None):
            if parents is None:
                parents = []
            
            if node == target:
                return parents
            
            for child in ast.iter_child_nodes(node):
                if isinstance(node, (ast.For, ast.While)):
                    result = find_parent_loops(child, target, parents + [node])
                else:
                    result = find_parent_loops(child, target, parents)
                if result is not None:
                    return result
            return None
        
        parent_loops = find_parent_loops(tree, loop_node)
        if parent_loops is None:
            parent_loops = []
        
        # Level is 1 + number of parent loops
        return len(parent_loops) + 1
    
    def _is_recursive(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is recursive."""
        func_name = func_node.name
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
        
        return False
    
    def _has_memoization(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has memoization decorator."""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ["lru_cache", "cache", "memoize"]:
                    return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if decorator.func.id in ["lru_cache", "cache", "memoize"]:
                        return True
        
        return False
    
    def _is_linear_search_pattern(self, node: ast.Compare) -> bool:
        """Check if comparison is a linear search pattern."""
        # Simple heuristic: x in list_variable
        for op in node.ops:
            if isinstance(op, (ast.In, ast.NotIn)):
                return True
        return False
    
    def _get_parent_node(self, node: ast.AST, tree: ast.AST) -> Optional[ast.AST]:
        """Get parent node of given node."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    return parent
        return None
    
    def _is_single_iteration_context(self, node: Optional[ast.AST]) -> bool:
        """Check if context only iterates once (e.g., for loop)."""
        if node and isinstance(node, ast.For):
            return True
        return False
    
    def _is_string_variable(self, target: ast.AST, scope: ast.AST) -> bool:
        """Check if variable is likely a string (heuristic)."""
        # This is a simplified heuristic
        return True  # Conservative assumption
    
    def _get_loop_variables(self, loop_node: ast.AST) -> Set[str]:
        """Get variables modified in loop."""
        vars = set()
        
        if isinstance(loop_node, ast.For):
            if isinstance(loop_node.target, ast.Name):
                vars.add(loop_node.target.id)
        
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        vars.add(target.id)
        
        return vars
    
    def _find_repeated_calls_in_loop(self, loop_node: ast.AST, loop_vars: Set[str]) -> List[Dict[str, Any]]:
        """Find calls that don't depend on loop variables."""
        calls = []
        call_names = defaultdict(int)
        
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                    
                    # Check if call uses loop variables
                    uses_loop_var = False
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in loop_vars:
                            uses_loop_var = True
                            break
                    
                    if not uses_loop_var:
                        call_names[call_name] += 1
        
        # Return calls that appear multiple times or could be hoisted
        return [{"name": name, "count": count} for name, count in call_names.items() if count > 1]
    
    def _is_pure_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if function appears to be pure (no side effects)."""
        # Heuristic: no global statements, no attribute assignments on external objects
        for node in ast.walk(func_node):
            if isinstance(node, ast.Global):
                return False
            if isinstance(node, ast.Nonlocal):
                return False
        
        return True
    
    def _count_function_calls(self, func_name: str, tree: ast.AST) -> int:
        """Count how many times a function is called."""
        count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    count += 1
        
        return count
    
    def _is_manual_sum_pattern(self, loop_node: ast.For) -> bool:
        """Check if loop manually calculates sum."""
        # Look for pattern: total += x
        for node in ast.walk(loop_node):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.op, ast.Add):
                    return True
        
        return False
    
    def _is_manual_max_min_pattern(self, loop_node: ast.For) -> bool:
        """Check if loop manually finds max/min."""
        # Look for comparison patterns
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.Gt, ast.Lt, ast.GtE, ast.LtE)):
                        # Check if followed by assignment
                        parent = self._get_parent_node(node, loop_node)
                        if isinstance(parent, ast.If):
                            return True
        
        return False
