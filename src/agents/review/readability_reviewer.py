"""
Readability Reviewer Agent - Code Readability Analysis

This agent analyzes code readability, naming conventions, comments, and structure.
Uses temperature=0.2 for precise, analytical review.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
import ast
from typing import Dict, List, Any, Optional, Set
from loguru import logger

from src.agents.base_agent import BaseAgent


class ReadabilityReviewer(BaseAgent):
    """
    Readability Reviewer Agent for code readability analysis.
    
    Responsibilities:
    - Analyze variable/function naming conventions
    - Evaluate code complexity (cyclomatic complexity)
    - Check comment quality and coverage
    - Assess code formatting and structure
    - Generate readability score (0-100)
    
    Uses temperature=0.2 for precise, analytical review.
    """
    
    def __init__(self, name: str = "readability_reviewer"):
        """
        Initialize Readability Reviewer.
        
        Args:
            name: Agent name (default: "readability_reviewer")
        """
        role = "Expert Code Readability Analyst for Python"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"  # Uses temperature 0.2
        )
        logger.info(f"{self.name} ready for readability analysis")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a readability review task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of review task
                - data: Task-specific parameters
        
        Returns:
            Dictionary with readability analysis results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        try:
            if task_type == "analyze_readability":
                return await self.analyze_readability(
                    code=data.get("code"),
                    filename=data.get("filename", "unknown.py")
                )
            
            elif task_type == "check_naming":
                return await self.check_naming_conventions(
                    code=data.get("code")
                )
            
            elif task_type == "analyze_complexity":
                return await self.analyze_complexity(
                    code=data.get("code")
                )
            
            elif task_type == "check_comments":
                return await self.check_comment_quality(
                    code=data.get("code")
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing readability review '{task_type}': {e}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def analyze_readability(
        self,
        code: str,
        filename: str = "unknown.py"
    ) -> Dict[str, Any]:
        """
        Comprehensive readability analysis.
        
        Args:
            code: Python code to analyze
            filename: Name of the file being analyzed
        
        Returns:
            Dictionary with readability analysis and score
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for readability analysis"
            }
        
        logger.info(f"{self.name} analyzing readability for {filename} ({len(code)} chars)")
        
        # Perform multiple analyses
        naming_result = await self.check_naming_conventions(code)
        complexity_result = await self.analyze_complexity(code)
        comment_result = await self.check_comment_quality(code)
        formatting_result = self._check_formatting(code)
        structure_result = self._check_structure(code)
        
        # Calculate weighted readability score
        naming_score = naming_result.get("score", 0)
        complexity_score = complexity_result.get("score", 0)
        comment_score = comment_result.get("score", 0)
        formatting_score = formatting_result.get("score", 0)
        structure_score = structure_result.get("score", 0)
        
        # Weights: naming(25%), complexity(25%), comments(20%), formatting(15%), structure(15%)
        total_score = (
            naming_score * 0.25 +
            complexity_score * 0.25 +
            comment_score * 0.20 +
            formatting_score * 0.15 +
            structure_score * 0.15
        )
        
        # Collect all issues
        all_issues = []
        all_issues.extend(naming_result.get("issues", []))
        all_issues.extend(complexity_result.get("issues", []))
        all_issues.extend(comment_result.get("issues", []))
        all_issues.extend(formatting_result.get("issues", []))
        all_issues.extend(structure_result.get("issues", []))
        
        # Generate AI-powered summary
        try:
            analysis_prompt = self.get_prompt("review_prompts", "readability_analysis", {
                "code": code[:3000],  # First 3000 chars for context
                "issues": "\n".join(f"- {issue}" for issue in all_issues[:10]),
                "naming_score": naming_score,
                "complexity_score": complexity_score,
                "comment_score": comment_score,
                "total_score": int(total_score)
            })
            
            ai_analysis = await self.generate_response(analysis_prompt)
        except Exception as e:
            logger.warning(f"AI analysis failed for readability: {e}")
            ai_analysis = f"Readability score: {int(total_score)}/100. {len(all_issues)} issues found."
        
        return {
            "success": True,
            "score": round(total_score, 2),
            "status": "success",
            "task": "analyze_readability",
            "filename": filename,
            "readability_score": round(total_score, 2),
            "breakdown": {
                "naming": round(naming_score, 2),
                "complexity": round(complexity_score, 2),
                "comments": round(comment_score, 2),
                "formatting": round(formatting_score, 2),
                "structure": round(structure_score, 2)
            },
            "issues": all_issues,
            "issue_count": len(all_issues),
            "analysis": ai_analysis,
            "recommendations": self._generate_recommendations(all_issues, total_score)
        }
    
    async def check_naming_conventions(self, code: str) -> Dict[str, Any]:
        """
        Check variable/function naming conventions.
        
        Args:
            code: Python code to analyze
        
        Returns:
            Dictionary with naming analysis
        """
        issues = []
        score = 100.0
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "error",
                "score": 0,
                "issues": [f"Syntax error prevents naming analysis: {e}"]
            }
        
        # Collect names
        variable_names = []
        function_names = []
        class_names = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variable_names.append(node.id)
        
        # Check function names (should be snake_case)
        for name in function_names:
            if name.startswith('_'):
                continue  # Private functions OK
            if not self._is_snake_case(name):
                issues.append(f"Function '{name}' should use snake_case naming")
                score -= 2
        
        # Check class names (should be PascalCase)
        for name in class_names:
            if not self._is_pascal_case(name):
                issues.append(f"Class '{name}' should use PascalCase naming")
                score -= 3
        
        # Check variable names
        for name in set(variable_names):
            # Skip loop variables and common patterns
            if len(name) == 1 or name in ['_', 'i', 'j', 'k', 'x', 'y', 'z']:
                continue
            
            # Check for non-descriptive names
            if len(name) < 3 and name not in ['id', 'df', 'np', 'pd']:
                issues.append(f"Variable '{name}' is too short (use descriptive names)")
                score -= 1
            
            # Check for ALL_CAPS (should be constants)
            if name.isupper() and not name.startswith('_'):
                # This is OK for constants at module level
                pass
            elif not self._is_snake_case(name) and not name.isupper():
                issues.append(f"Variable '{name}' should use snake_case naming")
                score -= 1
        
        # Check for overly generic names
        generic_names = ['data', 'temp', 'tmp', 'var', 'obj', 'item', 'thing', 'stuff']
        for name in variable_names + function_names:
            if name.lower() in generic_names:
                issues.append(f"Name '{name}' is too generic (use more descriptive names)")
                score -= 1
        
        score = max(0, min(100, score))
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": {
                "functions": len(function_names),
                "classes": len(class_names),
                "variables": len(set(variable_names))
            }
        }
    
    async def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """
        Analyze code complexity (cyclomatic complexity).
        
        Args:
            code: Python code to analyze
        
        Returns:
            Dictionary with complexity analysis
        """
        issues = []
        score = 100.0
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "error",
                "score": 0,
                "issues": [f"Syntax error prevents complexity analysis: {e}"]
            }
        
        # Calculate cyclomatic complexity for each function
        complexities = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                complexities[node.name] = complexity
                
                # Penalize high complexity
                if complexity > 15:
                    issues.append(f"Function '{node.name}' has very high complexity ({complexity}) - consider refactoring")
                    score -= 10
                elif complexity > 10:
                    issues.append(f"Function '{node.name}' has high complexity ({complexity}) - consider simplifying")
                    score -= 5
                elif complexity > 7:
                    issues.append(f"Function '{node.name}' has moderate complexity ({complexity})")
                    score -= 2
        
        # Check for deeply nested code
        max_nesting = self._calculate_max_nesting(tree)
        if max_nesting > 5:
            issues.append(f"Maximum nesting depth is {max_nesting} - consider extracting nested logic")
            score -= 5 * (max_nesting - 5)
        elif max_nesting > 3:
            issues.append(f"Nesting depth of {max_nesting} is acceptable but could be improved")
            score -= 2
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if line_count > 100:
                    issues.append(f"Function '{node.name}' is very long ({line_count} lines) - consider splitting")
                    score -= 10
                elif line_count > 50:
                    issues.append(f"Function '{node.name}' is long ({line_count} lines)")
                    score -= 5
        
        score = max(0, min(100, score))
        
        avg_complexity = sum(complexities.values()) / len(complexities) if complexities else 0
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": {
                "function_count": len(complexities),
                "average_complexity": round(avg_complexity, 2),
                "max_complexity": max(complexities.values()) if complexities else 0,
                "max_nesting": max_nesting,
                "complexities": complexities
            }
        }
    
    async def check_comment_quality(self, code: str) -> Dict[str, Any]:
        """
        Check comment quality and coverage.
        
        Args:
            code: Python code to analyze
        
        Returns:
            Dictionary with comment analysis
        """
        issues = []
        score = 100.0
        
        lines = code.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        docstring_count = len(re.findall(r'"""[\s\S]*?"""', code)) + len(re.findall(r"'''[\s\S]*?'''", code))
        
        # Calculate comment ratio
        comment_ratio = (comment_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Check for missing docstrings
        try:
            tree = ast.parse(code)
            functions_without_docstrings = 0
            classes_without_docstrings = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        if not node.name.startswith('_'):  # Ignore private functions
                            functions_without_docstrings += 1
                            issues.append(f"Function '{node.name}' lacks docstring")
                            score -= 3
                
                elif isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        classes_without_docstrings += 1
                        issues.append(f"Class '{node.name}' lacks docstring")
                        score -= 5
        except SyntaxError:
            pass
        
        # Check comment ratio
        if comment_ratio < 5:
            issues.append(f"Very low comment coverage ({comment_ratio:.1f}%) - add more explanatory comments")
            score -= 15
        elif comment_ratio < 10:
            issues.append(f"Low comment coverage ({comment_ratio:.1f}%)")
            score -= 10
        elif comment_ratio > 50:
            issues.append(f"Excessive comments ({comment_ratio:.1f}%) - code might be too complex")
            score -= 5
        
        # Check for TODO/FIXME comments
        todo_count = len(re.findall(r'#.*TODO', code, re.IGNORECASE))
        fixme_count = len(re.findall(r'#.*FIXME', code, re.IGNORECASE))
        
        if todo_count > 5:
            issues.append(f"Many TODO comments ({todo_count}) - address technical debt")
            score -= 5
        
        if fixme_count > 0:
            issues.append(f"FIXME comments present ({fixme_count}) - critical issues need attention")
            score -= 10
        
        score = max(0, min(100, score))
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": {
                "total_lines": total_lines,
                "comment_lines": comment_lines,
                "comment_ratio": round(comment_ratio, 2),
                "docstrings": docstring_count,
                "todos": todo_count,
                "fixmes": fixme_count
            }
        }
    
    def _check_formatting(self, code: str) -> Dict[str, Any]:
        """Check code formatting (line length, spacing, etc.)"""
        issues = []
        score = 100.0
        
        lines = code.split('\n')
        
        # Check line length (PEP 8: max 79 chars, but allow 100)
        long_lines = [(i+1, len(line)) for i, line in enumerate(lines) if len(line) > 100]
        if long_lines:
            for line_num, length in long_lines[:5]:  # Report first 5
                issues.append(f"Line {line_num} is too long ({length} chars) - keep under 100")
                score -= 1
            if len(long_lines) > 5:
                issues.append(f"... and {len(long_lines) - 5} more long lines")
        
        # Check for inconsistent indentation
        indent_sizes = set()
        for line in lines:
            if line and line[0] == ' ':
                indent = len(line) - len(line.lstrip(' '))
                if indent > 0:
                    indent_sizes.add(indent % 4)
        
        if len(indent_sizes) > 1:
            issues.append("Inconsistent indentation detected - use 4 spaces consistently")
            score -= 10
        
        # Check for trailing whitespace
        trailing_ws = sum(1 for line in lines if line.endswith(' ') or line.endswith('\t'))
        if trailing_ws > 5:
            issues.append(f"Trailing whitespace on {trailing_ws} lines")
            score -= 5
        
        # Check for multiple blank lines
        blank_line_groups = re.findall(r'\n\n\n+', code)
        if blank_line_groups:
            issues.append(f"Multiple consecutive blank lines found ({len(blank_line_groups)} occurrences)")
            score -= 3
        
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "issues": issues
        }
    
    def _check_structure(self, code: str) -> Dict[str, Any]:
        """Check code structure and organization"""
        issues = []
        score = 100.0
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"score": 0, "issues": ["Syntax error prevents structure analysis"]}
        
        # Check import organization
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            # Check if imports are at the top
            first_import_idx = next((i for i, node in enumerate(tree.body) if isinstance(node, (ast.Import, ast.ImportFrom))), None)
            non_import_before = any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)) 
                                   for node in tree.body[:first_import_idx] if first_import_idx)
            
            if non_import_before:
                issues.append("Imports should be at the top of the file")
                score -= 5
        
        # Check for magic numbers
        magic_numbers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in [0, 1, -1, 2, 10, 100, 1000]:  # Common acceptable numbers
                    magic_numbers.append(node.value)
        
        if len(set(magic_numbers)) > 5:
            issues.append(f"Many magic numbers found ({len(set(magic_numbers))}) - consider using named constants")
            score -= 10
        
        # Check for proper class structure
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if __init__ is first method
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if methods and methods[0].name != '__init__' and '__init__' in [m.name for m in methods]:
                    issues.append(f"Class '{node.name}': __init__ should be the first method")
                    score -= 3
        
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "issues": issues
        }
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Count decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def traverse(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                depth += 1
            
            for child in ast.iter_child_nodes(node):
                traverse(child, depth)
        
        traverse(tree)
        return max_depth
    
    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention"""
        return bool(re.match(r'^[a-z_][a-z0-9_]*$', name))
    
    def _is_pascal_case(self, name: str) -> bool:
        """Check if name follows PascalCase convention"""
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))
    
    def _generate_recommendations(self, issues: List[str], score: float) -> List[str]:
        """Generate actionable recommendations based on issues"""
        recommendations = []
        
        if score < 60:
            recommendations.append("🔴 CRITICAL: Major readability improvements needed")
        elif score < 75:
            recommendations.append("🟡 WARNING: Several readability issues should be addressed")
        else:
            recommendations.append("✅ GOOD: Code readability is acceptable")
        
        # Categorize issues
        naming_issues = [i for i in issues if 'naming' in i.lower() or 'name' in i.lower()]
        complexity_issues = [i for i in issues if 'complexity' in i.lower() or 'nesting' in i.lower()]
        comment_issues = [i for i in issues if 'comment' in i.lower() or 'docstring' in i.lower()]
        
        if naming_issues:
            recommendations.append(f"📝 Fix {len(naming_issues)} naming convention issues")
        if complexity_issues:
            recommendations.append(f"🔧 Reduce complexity in {len(complexity_issues)} locations")
        if comment_issues:
            recommendations.append(f"💬 Improve documentation ({len(comment_issues)} issues)")
        
        # Specific recommendations
        if any('too short' in i for i in issues):
            recommendations.append("Use descriptive variable names (minimum 3 characters)")
        if any('too long' in i for i in issues):
            recommendations.append("Keep line length under 100 characters")
        if any('complexity' in i.lower() for i in issues):
            recommendations.append("Extract complex logic into smaller functions")
        if any('docstring' in i for i in issues):
            recommendations.append("Add docstrings to all public functions and classes")
        
        return recommendations[:5]  # Return top 5 recommendations
