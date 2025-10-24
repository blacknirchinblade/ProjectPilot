"""
Code Connectivity Reviewer Agent

Analyzes code connectivity and dependencies within a single file.
Provides comprehensive connectivity quality scoring.

Features:
- Function dependency analysis
- Data flow tracking
- API consistency checking
- Module cohesion analysis
- Coupling detection

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import re
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from src.agents.base_agent import BaseAgent
from src.utils.logger import logger


class CodeConnectivityReviewer(BaseAgent):
    """
    Agent specialized in analyzing code connectivity within files.
    
    Provides scoring on:
    - Function dependencies (25%)
    - Data flow (25%)
    - API consistency (20%)
    - Module cohesion (15%)
    - Coupling level (15%)
    """
    
    def __init__(self, name: str = "code_connectivity_reviewer"):
        """
        Initialize Code Connectivity Reviewer.
        
        Args:
            name: Agent name (default: "code_connectivity_reviewer")
        """
        role = "Expert Code Connectivity and Dependency Analyst for Python"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"  # Uses temperature 0.2
        )
        logger.info(f"{self.name} ready for connectivity analysis")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute connectivity analysis task.
        
        Task Types:
        - analyze_connectivity: Full connectivity analysis
        - check_dependencies: Function dependency only
        - check_data_flow: Data flow only
        - check_api_consistency: API patterns only
        
        Args:
            task: Task dictionary with type and data
            
        Returns:
            Analysis results with score and issues
        """
        task_type = task.get("task_type", "")
        data = task.get("data", {})
        
        try:
            if task_type == "analyze_connectivity":
                code = data.get("code", "")
                filename = data.get("filename", "unknown.py")
                return await self.analyze_connectivity(code, filename)
            
            elif task_type == "check_dependencies":
                code = data.get("code", "")
                return await self.check_dependencies(code)
            
            elif task_type == "check_data_flow":
                code = data.get("code", "")
                return await self.check_data_flow(code)
            
            elif task_type == "check_api_consistency":
                code = data.get("code", "")
                return await self.check_api_consistency(code)
            
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
    
    async def analyze_connectivity(self, code: str, filename: str) -> Dict[str, Any]:
        """
        Comprehensive connectivity analysis.
        
        Analyzes:
        - Function dependencies
        - Data flow
        - API consistency
        - Module cohesion
        - Coupling levels
        
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
                "connectivity_score": 0
            }
        
        logger.info(f"{self.name} analyzing connectivity for {filename} ({len(code)} chars)")
        
        # Run all analysis components
        dependency_result = await self.check_dependencies(code)
        data_flow_result = await self.check_data_flow(code)
        api_result = await self.check_api_consistency(code)
        cohesion_result = await self._check_module_cohesion(code)
        coupling_result = await self._check_coupling(code)
        
        # Calculate weighted score
        dependency_score = dependency_result.get("score", 100)
        data_flow_score = data_flow_result.get("score", 100)
        api_score = api_result.get("score", 100)
        cohesion_score = cohesion_result.get("score", 100)
        coupling_score = coupling_result.get("score", 100)
        
        total_score = (
            dependency_score * 0.25 +
            data_flow_score * 0.25 +
            api_score * 0.20 +
            cohesion_score * 0.15 +
            coupling_score * 0.15
        )
        
        # Collect all issues
        all_issues = []
        all_issues.extend(dependency_result.get("issues", []))
        all_issues.extend(data_flow_result.get("issues", []))
        all_issues.extend(api_result.get("issues", []))
        all_issues.extend(cohesion_result.get("issues", []))
        all_issues.extend(coupling_result.get("issues", []))
        
        # Get AI analysis
        try:
            analysis_prompt = self.get_prompt("review_prompts", "connectivity_analysis", {
                "code": code[:3000],
                "filename": filename,
                "issues": "\n".join(f"- {issue}" for issue in all_issues[:10]) if all_issues else "No major issues",
                "dependency_score": dependency_score,
                "data_flow_score": data_flow_score,
                "total_score": round(total_score, 1)
            })
            
            analysis = await self.generate_response(analysis_prompt)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            analysis = f"Connectivity score: {round(total_score, 1)}/100. {len(all_issues)} issues found."
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, total_score)
        
        return {
            "success": True,
            "score": round(total_score, 1),
            "status": "success",
            "connectivity_score": round(total_score, 1),
            "breakdown": {
                "dependencies": round(dependency_score, 1),
                "data_flow": round(data_flow_score, 1),
                "api_consistency": round(api_score, 1),
                "cohesion": round(cohesion_score, 1),
                "coupling": round(coupling_score, 1)
            },
            "issues": all_issues,
            "analysis": analysis,
            "recommendations": recommendations
        }
    
    async def check_dependencies(self, code: str) -> Dict[str, Any]:
        """
        Analyze function dependencies.
        
        Checks:
        - Function call graph
        - Circular dependencies
        - Unused functions
        - Dependency complexity
        
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
        
        # Build function call graph
        functions = {}
        function_calls = defaultdict(set)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
                # Find calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            function_calls[node.name].add(child.func.id)
        
        # Check for circular dependencies
        for func_name in functions:
            if self._has_circular_dependency(func_name, function_calls, set()):
                issues.append(f"Circular dependency detected involving '{func_name}'")
                score -= 20
        
        # Check for unused functions
        called_functions = set()
        for calls in function_calls.values():
            called_functions.update(calls)
        
        unused = set(functions.keys()) - called_functions - {"main", "__init__"}
        if len(unused) > 0:
            issues.append(f"{len(unused)} potentially unused functions: {', '.join(list(unused)[:3])}")
            score -= min(len(unused) * 5, 15)
        
        # Check dependency complexity
        max_dependencies = max([len(calls) for calls in function_calls.values()] + [0])
        if max_dependencies > 5:
            issues.append(f"High function dependency: {max_dependencies} calls in one function")
            score -= 10
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues,
            "statistics": {
                "total_functions": len(functions),
                "unused_functions": len(unused),
                "max_dependencies": max_dependencies
            }
        }
    
    async def check_data_flow(self, code: str) -> Dict[str, Any]:
        """
        Analyze data flow patterns.
        
        Checks:
        - Variable usage
        - Parameter passing
        - Return value consistency
        - Global variable usage
        
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
        
        # Check for global variable usage
        globals_used = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                globals_used.extend(node.names)
        
        if len(globals_used) > 3:
            issues.append(f"Excessive global variable usage: {len(globals_used)} globals")
            score -= 20
        elif len(globals_used) > 0:
            issues.append(f"Global variables used: {', '.join(globals_used[:3])}")
            score -= 10
        
        # Check for inconsistent return patterns
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in functions:
            returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
            if len(returns) > 0:
                # Check if some returns have values and some don't
                has_value = [r for r in returns if r.value is not None]
                no_value = [r for r in returns if r.value is None]
                
                if len(has_value) > 0 and len(no_value) > 0:
                    issues.append(f"Function '{func.name}' has inconsistent return patterns")
                    score -= 10
        
        # Check for unused variables
        # This is a simplified check
        assigned_vars = set()
        used_vars = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
        
        unused_vars = assigned_vars - used_vars - {"_"}
        if len(unused_vars) > 2:
            issues.append(f"{len(unused_vars)} potentially unused variables")
            score -= min(len(unused_vars) * 3, 10)
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues,
            "statistics": {
                "global_variables": len(globals_used),
                "unused_variables": len(unused_vars)
            }
        }
    
    async def check_api_consistency(self, code: str) -> Dict[str, Any]:
        """
        Check API consistency patterns.
        
        Checks:
        - Function signature consistency
        - Naming patterns
        - Return type consistency
        - Parameter order patterns
        
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
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if len(functions) == 0:
            return {
                "status": "success",
                "score": 100,
                "issues": []
            }
        
        # Check for inconsistent parameter counts
        param_counts = {}
        for func in functions:
            if func.name.startswith("_"):
                continue  # Skip private functions
            param_count = len(func.args.args)
            param_counts[func.name] = param_count
        
        # Check for functions with too many parameters
        for func_name, count in param_counts.items():
            if count > 5:
                issues.append(f"Function '{func_name}' has {count} parameters (consider refactoring)")
                score -= 5
        
        # Check for inconsistent naming in similar functions
        # Group functions by prefix
        prefixes = defaultdict(list)
        for func in functions:
            if "_" in func.name:
                prefix = func.name.split("_")[0]
                prefixes[prefix].append(func.name)
        
        # Check if functions with same prefix have consistent patterns
        for prefix, names in prefixes.items():
            if len(names) > 2:
                # Check if all use same verb pattern
                verbs = set()
                for name in names:
                    parts = name.split("_")
                    if len(parts) > 1:
                        verbs.add(parts[0])
                
                if len(verbs) > 1:
                    issues.append(f"Inconsistent naming for '{prefix}_*' functions")
                    score -= 5
        
        # Check for missing type hints (if any function has them)
        has_type_hints = any(func.returns is not None for func in functions)
        if has_type_hints:
            no_hints = [f.name for f in functions if f.returns is None and not f.name.startswith("_")]
            if len(no_hints) > 0:
                issues.append(f"Inconsistent type hints: {len(no_hints)} functions missing return types")
                score -= 10
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues
        }
    
    async def _check_module_cohesion(self, code: str) -> Dict[str, Any]:
        """
        Check module cohesion.
        
        Checks:
        - Related functions grouping
        - Class organization
        - Shared state usage
        
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
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Check for mix of classes and standalone functions
        if len(classes) > 0 and len(functions) > len(classes) * 2:
            issues.append("Mix of classes and many standalone functions - consider organizing")
            score -= 10
        
        # Check for very small classes (should be functions)
        for cls in classes:
            methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
            if len(methods) <= 2 and "__init__" in [m.name for m in methods]:
                issues.append(f"Class '{cls.name}' has minimal functionality - consider using functions")
                score -= 5
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues
        }
    
    async def _check_coupling(self, code: str) -> Dict[str, Any]:
        """
        Check coupling levels.
        
        Checks:
        - Inter-function dependencies
        - Shared mutable state
        - Import coupling
        
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
        
        # Check for excessive imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        if len(imports) > 15:
            issues.append(f"High number of imports: {len(imports)} - consider splitting module")
            score -= 15
        elif len(imports) > 10:
            issues.append(f"Moderate import coupling: {len(imports)} imports")
            score -= 5
        
        # Check for module-level mutable defaults
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(f"Function '{node.name}' uses mutable default argument")
                        score -= 10
                        break
        
        return {
            "status": "success",
            "score": max(0, score),
            "issues": issues,
            "statistics": {
                "import_count": len(imports)
            }
        }
    
    def _has_circular_dependency(self, func: str, call_graph: Dict[str, Set[str]], 
                                 visited: Set[str]) -> bool:
        """
        Check for circular dependencies in function calls.
        
        Args:
            func: Function name to check
            call_graph: Dictionary of function -> set of called functions
            visited: Set of already visited functions
            
        Returns:
            True if circular dependency detected
        """
        if func in visited:
            return True
        
        visited.add(func)
        
        if func in call_graph:
            for called in call_graph[func]:
                if self._has_circular_dependency(called, call_graph, visited.copy()):
                    return True
        
        return False
    
    def _generate_recommendations(self, issues: List[str], score: float) -> List[str]:
        """
        Generate actionable recommendations.
        
        Args:
            issues: List of detected issues
            score: Overall connectivity score
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        if score < 60:
            recommendations.append("🔴 CRITICAL: Major connectivity issues - requires refactoring")
        elif score < 75:
            recommendations.append("🟡 WARNING: Multiple connectivity issues - review architecture")
        else:
            recommendations.append("🟢 GOOD: Code connectivity is acceptable")
        
        # Categorize issues
        dependency_issues = [i for i in issues if "dependency" in i.lower() or "circular" in i.lower()]
        data_flow_issues = [i for i in issues if "global" in i.lower() or "variable" in i.lower()]
        api_issues = [i for i in issues if "parameter" in i.lower() or "inconsistent" in i.lower()]
        coupling_issues = [i for i in issues if "import" in i.lower() or "coupling" in i.lower()]
        
        # Priority recommendations
        if dependency_issues:
            recommendations.append(f"1. Fix circular dependencies ({len(dependency_issues)} issues)")
        
        if data_flow_issues:
            recommendations.append(f"2. Improve data flow ({len(data_flow_issues)} issues)")
        
        if api_issues:
            recommendations.append(f"3. Standardize API patterns ({len(api_issues)} issues)")
        
        if coupling_issues:
            recommendations.append(f"4. Reduce coupling ({len(coupling_issues)} issues)")
        
        if score >= 80:
            recommendations.append("5. Consider adding type hints for better clarity")
        
        return recommendations[:5]  # Top 5
