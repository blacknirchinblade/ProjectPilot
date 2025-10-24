"""
Project Connectivity Reviewer Agent

Analyzes code connectivity and dependencies ACROSS multiple files in a project.
Detects issues like circular imports, high module coupling, broken API contracts,
and provides recommendations for better project structure.

This complements CodeConnectivityReviewer which analyzes single files.
Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import re

from src.agents.base_agent import BaseAgent
from src.utils.logger import logger


class ProjectConnectivityReviewer(BaseAgent):
    """
    Reviews cross-file connectivity and dependencies in Python projects.
    
    Analyzes:
    1. Import Dependencies (30%): Import graph, circular imports, import depth
    2. Module Coupling (25%): Coupling levels, fan-in/fan-out analysis
    3. API Contracts (20%): Cross-file interface consistency
    4. Package Structure (15%): Organization, cohesion
    5. Dependency Health (10%): Unused imports, redundant dependencies
    
    Output: 0-100 score with detailed breakdown and recommendations
    """
    
    def __init__(self, name: str = "project_connectivity_reviewer"):
        """Initialize the project connectivity reviewer"""
        role = "Expert Project Architecture and Cross-File Dependency Analyst for Python"
        super().__init__(name=name, role=role, agent_type="review")
        logger.info(f"{self.name} ready for project connectivity analysis")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a project connectivity review task
        
        Supported task types:
        - analyze_project: Full project connectivity analysis
        - check_imports: Import dependency analysis only
        - check_coupling: Module coupling analysis only
        - check_api_contracts: API consistency across files
        
        Args:
            task: Task dictionary with type and data
            
        Returns:
            Analysis results with score, breakdown, issues, recommendations
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        if task_type == "analyze_project":
            return await self.analyze_project_connectivity(
                files=data.get("files", {}),
                project_root=data.get("project_root", "")
            )
        
        elif task_type == "check_imports":
            return await self.check_import_dependencies(
                files=data.get("files", {})
            )
        
        elif task_type == "check_coupling":
            return await self.check_module_coupling(
                files=data.get("files", {})
            )
        
        elif task_type == "check_api_contracts":
            return await self.check_api_contracts(
                files=data.get("files", {})
            )
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    async def analyze_project_connectivity(
        self,
        files: Dict[str, str],
        project_root: str = ""
    ) -> Dict[str, Any]:
        """
        Perform complete project connectivity analysis
        
        Args:
            files: Dict mapping file paths to code content
            project_root: Root directory of project (for relative paths)
            
        Returns:
            Comprehensive analysis with score, breakdown, issues, recommendations
        """
        if not files:
            return {
                "status": "error",
                "message": "No files provided for analysis"
            }
        
        # Run all analysis components
        import_result = await self.check_import_dependencies(files)
        coupling_result = await self.check_module_coupling(files)
        api_result = await self.check_api_contracts(files)
        structure_result = await self._check_package_structure(files, project_root)
        health_result = await self._check_dependency_health(files)
        
        # Calculate weighted total score
        total_score = (
            import_result["score"] * 0.30 +
            coupling_result["score"] * 0.25 +
            api_result["score"] * 0.20 +
            structure_result["score"] * 0.15 +
            health_result["score"] * 0.10
        )
        
        # Aggregate all issues
        all_issues = (
            import_result["issues"] +
            coupling_result["issues"] +
            api_result["issues"] +
            structure_result["issues"] +
            health_result["issues"]
        )
        
        # Generate AI-enhanced analysis
        analysis = ""
        try:
            prompt = self.get_prompt(
                "project_connectivity_analysis",
                files=str(list(files.keys())),
                issues="\n".join(all_issues),
                import_score=import_result["score"],
                coupling_score=coupling_result["score"],
                total_score=total_score
            )
            analysis = await self.generate(prompt)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            analysis = "AI analysis unavailable - see issues list for details"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, total_score)
        
        return {
            "status": "success",
            "project_score": round(total_score, 1),
            "breakdown": {
                "imports": round(import_result["score"], 1),
                "coupling": round(coupling_result["score"], 1),
                "api_contracts": round(api_result["score"], 1),
                "structure": round(structure_result["score"], 1),
                "health": round(health_result["score"], 1)
            },
            "issues": all_issues,
            "analysis": analysis,
            "recommendations": recommendations,
            "statistics": {
                "total_files": len(files),
                "import_graph": import_result.get("statistics", {}),
                "coupling_metrics": coupling_result.get("statistics", {})
            }
        }
    
    async def check_import_dependencies(
        self,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze import dependencies across files
        
        Checks:
        - Import graph construction
        - Circular import detection
        - Import depth analysis
        - Unused external imports
        
        Args:
            files: Dict mapping file paths to code content
            
        Returns:
            Dict with score, issues, statistics
        """
        score = 100
        issues = []
        
        # Build import graph: file -> set of imported modules
        import_graph = defaultdict(set)
        external_imports = defaultdict(set)  # Non-project imports
        
        for filepath, code in files.items():
            try:
                tree = ast.parse(code)
                module_name = self._path_to_module(filepath)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_name = alias.name
                            if self._is_project_import(import_name, files):
                                import_graph[module_name].add(import_name)
                            else:
                                external_imports[module_name].add(import_name)
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_name = node.module
                            if self._is_project_import(import_name, files):
                                import_graph[module_name].add(import_name)
                            else:
                                external_imports[module_name].add(import_name)
            
            except SyntaxError:
                continue
        
        # Detect circular imports
        circular_imports = self._find_circular_imports(import_graph)
        if circular_imports:
            score -= 30
            for cycle in circular_imports[:3]:  # Show first 3
                issues.append(f"Circular import detected: {' -> '.join(cycle)}")
        
        # Check import depth (max depth in dependency chain)
        max_depth = self._calculate_max_import_depth(import_graph)
        if max_depth > 5:
            score -= 15
            issues.append(f"Deep import chain detected (depth: {max_depth})")
        elif max_depth > 3:
            score -= 5
            issues.append(f"Import chain depth is {max_depth} (consider flattening)")
        
        # Check for files importing too many others
        for module, imports in import_graph.items():
            if len(imports) > 10:
                score -= 10
                issues.append(f"Module {module} imports too many other modules ({len(imports)})")
                break  # Only penalize once
        
        # Check for too many external dependencies per file
        for module, imports in external_imports.items():
            if len(imports) > 15:
                score -= 5
                issues.append(f"Module {module} has many external dependencies ({len(imports)})")
                break
        
        score = max(0, score)
        
        statistics = {
            "total_modules": len(import_graph),
            "circular_imports": len(circular_imports),
            "max_import_depth": max_depth,
            "avg_imports_per_file": (
                sum(len(imports) for imports in import_graph.values()) / len(import_graph)
                if import_graph else 0
            )
        }
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": statistics
        }
    
    async def check_module_coupling(
        self,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze module coupling levels
        
        Checks:
        - Afferent coupling (fan-in): How many modules depend on this one
        - Efferent coupling (fan-out): How many modules this one depends on
        - Instability metric: Ce / (Ca + Ce)
        - Bidirectional dependencies
        
        Args:
            files: Dict mapping file paths to code content
            
        Returns:
            Dict with score, issues, statistics
        """
        score = 100
        issues = []
        
        # Build dependency graph
        dependencies = defaultdict(set)  # A -> {B, C} means A imports B and C
        
        for filepath, code in files.items():
            try:
                tree = ast.parse(code)
                module_name = self._path_to_module(filepath)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if self._is_project_import(alias.name, files):
                                    dependencies[module_name].add(alias.name)
                        else:
                            if node.module and self._is_project_import(node.module, files):
                                dependencies[module_name].add(node.module)
            
            except SyntaxError:
                continue
        
        # Calculate coupling metrics
        afferent = defaultdict(int)  # How many depend on this module (fan-in)
        efferent = defaultdict(int)  # How many this module depends on (fan-out)
        
        for module, deps in dependencies.items():
            efferent[module] = len(deps)
            for dep in deps:
                afferent[dep] += 1
        
        # Find highly coupled modules
        high_coupling = []
        for module in set(list(afferent.keys()) + list(efferent.keys())):
            total_coupling = afferent[module] + efferent[module]
            if total_coupling > 10:
                high_coupling.append((module, total_coupling))
        
        if high_coupling:
            score -= min(20, len(high_coupling) * 5)
            for module, coupling in high_coupling[:3]:
                issues.append(
                    f"High coupling in {module}: "
                    f"{afferent[module]} dependents, {efferent[module]} dependencies"
                )
        
        # Detect bidirectional dependencies (A imports B and B imports A)
        bidirectional = []
        for module_a, deps_a in dependencies.items():
            for module_b in deps_a:
                if module_a in dependencies.get(module_b, set()):
                    pair = tuple(sorted([module_a, module_b]))
                    if pair not in bidirectional:
                        bidirectional.append(pair)
        
        if bidirectional:
            score -= min(25, len(bidirectional) * 10)
            for mod_a, mod_b in bidirectional[:3]:
                issues.append(f"Bidirectional dependency: {mod_a} ↔ {mod_b}")
        
        # Calculate instability (0 = stable, 1 = unstable)
        unstable_modules = []
        for module in dependencies.keys():
            ca = afferent[module]
            ce = efferent[module]
            if ca + ce > 0:
                instability = ce / (ca + ce)
                if instability > 0.8 and ce > 3:
                    unstable_modules.append((module, instability))
        
        if unstable_modules:
            score -= min(10, len(unstable_modules) * 3)
            issues.append(
                f"{len(unstable_modules)} highly unstable module(s) detected "
                "(high efferent, low afferent coupling)"
            )
        
        score = max(0, score)
        
        statistics = {
            "high_coupling_count": len(high_coupling),
            "bidirectional_deps": len(bidirectional),
            "avg_afferent": sum(afferent.values()) / len(afferent) if afferent else 0,
            "avg_efferent": sum(efferent.values()) / len(efferent) if efferent else 0
        }
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": statistics
        }
    
    async def check_api_contracts(
        self,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Check API consistency across files
        
        Checks:
        - Function signature consistency
        - Return type consistency
        - Exception handling patterns
        - Naming convention consistency
        
        Args:
            files: Dict mapping file paths to code content
            
        Returns:
            Dict with score, issues, statistics
        """
        score = 100
        issues = []
        
        # Collect all public functions (not starting with _)
        public_functions = defaultdict(list)  # name -> [(file, signature)]
        
        for filepath, code in files.items():
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):
                            # Build signature
                            args = [arg.arg for arg in node.args.args]
                            returns = ast.unparse(node.returns) if node.returns else None
                            
                            public_functions[node.name].append({
                                'file': filepath,
                                'args': args,
                                'returns': returns,
                                'arg_count': len(args)
                            })
            
            except SyntaxError:
                continue
        
        # Check for same-named functions with different signatures
        inconsistent_apis = []
        for func_name, definitions in public_functions.items():
            if len(definitions) > 1:
                # Check if all have same signature
                arg_counts = [d['arg_count'] for d in definitions]
                if len(set(arg_counts)) > 1:
                    inconsistent_apis.append(func_name)
        
        if inconsistent_apis:
            score -= min(30, len(inconsistent_apis) * 10)
            for func in inconsistent_apis[:3]:
                defs = public_functions[func]
                issues.append(
                    f"Inconsistent API: {func}() has different signatures in "
                    f"{len(defs)} files"
                )
        
        # Check for consistent naming patterns
        naming_issues = self._check_naming_consistency(public_functions)
        if naming_issues:
            score -= 10
            issues.extend(naming_issues[:2])
        
        # Check return type consistency for same-named functions
        return_inconsistencies = 0
        for func_name, definitions in public_functions.items():
            if len(definitions) > 1:
                returns = [d['returns'] for d in definitions]
                if None in returns and any(r is not None for r in returns):
                    return_inconsistencies += 1
        
        if return_inconsistencies > 0:
            score -= min(15, return_inconsistencies * 5)
            issues.append(
                f"{return_inconsistencies} function(s) have inconsistent return type hints"
            )
        
        score = max(0, score)
        
        statistics = {
            "total_public_functions": len(public_functions),
            "duplicate_names": sum(1 for defs in public_functions.values() if len(defs) > 1),
            "inconsistent_apis": len(inconsistent_apis)
        }
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": statistics
        }
    
    async def _check_package_structure(
        self,
        files: Dict[str, str],
        project_root: str
    ) -> Dict[str, Any]:
        """
        Check package organization and structure
        
        Checks:
        - __init__.py presence in packages
        - Package depth
        - File organization
        - Module cohesion
        """
        score = 100
        issues = []
        
        # Get all directories
        directories = set()
        for filepath in files.keys():
            parts = Path(filepath).parts
            for i in range(1, len(parts)):
                directories.add('/'.join(parts[:i]))
        
        # Check for __init__.py in packages
        missing_init = []
        for directory in directories:
            init_path_1 = f"{directory}/__init__.py"
            init_path_2 = f"{directory}\\__init__.py"
            if init_path_1 not in files and init_path_2 not in files:
                # Only care about directories with Python files
                has_py_files = any(
                    str(Path(f).parent) == directory and f.endswith('.py')
                    for f in files.keys()
                )
                if has_py_files:
                    missing_init.append(directory)
        
        if missing_init:
            score -= min(15, len(missing_init) * 5)
            for dir_path in missing_init[:2]:
                issues.append(f"Package missing __init__.py: {dir_path}")
        
        # Check package depth
        max_depth = max((len(Path(f).parts) for f in files.keys()), default=0)
        if max_depth > 5:
            score -= 10
            issues.append(f"Deep package nesting (depth: {max_depth})")
        
        # Check for too many files in one directory
        file_counts = defaultdict(int)
        for filepath in files.keys():
            directory = str(Path(filepath).parent)
            file_counts[directory] += 1
        
        crowded_dirs = [d for d, count in file_counts.items() if count > 15]
        if crowded_dirs:
            score -= 5
            issues.append(f"{len(crowded_dirs)} director(ies) have too many files (>15)")
        
        score = max(0, score)
        
        statistics = {
            "total_directories": len(directories),
            "missing_init_files": len(missing_init),
            "max_depth": max_depth
        }
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": statistics
        }
    
    async def _check_dependency_health(
        self,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Check dependency health
        
        Checks:
        - Unused imports
        - Redundant imports
        - Import order consistency
        """
        score = 100
        issues = []
        
        total_unused = 0
        
        for filepath, code in files.items():
            try:
                tree = ast.parse(code)
                
                # Collect imports
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.asname or alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports.add(alias.asname or alias.name)
                
                # Collect name usage
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        used_names.add(node.id)
                
                # Find unused imports
                unused = imports - used_names
                # Filter out common false positives
                unused = {u for u in unused if u not in ['annotations', 'TYPE_CHECKING']}
                
                total_unused += len(unused)
            
            except SyntaxError:
                continue
        
        if total_unused > 10:
            score -= 20
            issues.append(f"{total_unused} unused import(s) detected across project")
        elif total_unused > 5:
            score -= 10
            issues.append(f"{total_unused} unused import(s) detected")
        
        score = max(0, score)
        
        statistics = {
            "unused_imports": total_unused
        }
        
        return {
            "status": "success",
            "score": score,
            "issues": issues,
            "statistics": statistics
        }
    
    def _find_circular_imports(
        self,
        import_graph: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """
        Find all circular import chains
        
        Args:
            import_graph: Module -> set of imported modules
            
        Returns:
            List of circular import chains
        """
        cycles = []
        
        def dfs(node: str, path: List[str], visited: Set[str]) -> None:
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                # Normalize cycle (smallest node first for deduplication)
                min_idx = cycle.index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                if normalized not in cycles:
                    cycles.append(normalized)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in import_graph.get(node, set()):
                dfs(neighbor, path[:], visited)
        
        for node in import_graph.keys():
            dfs(node, [], set())
        
        return cycles
    
    def _calculate_max_import_depth(
        self,
        import_graph: Dict[str, Set[str]]
    ) -> int:
        """
        Calculate maximum depth of import chain
        
        Args:
            import_graph: Module -> set of imported modules
            
        Returns:
            Maximum depth
        """
        def get_depth(node: str, visited: Set[str]) -> int:
            if node in visited:
                return 0
            
            visited.add(node)
            
            if node not in import_graph or not import_graph[node]:
                return 1
            
            max_child_depth = max(
                (get_depth(child, visited.copy()) for child in import_graph[node]),
                default=0
            )
            
            return 1 + max_child_depth
        
        if not import_graph:
            return 0
        
        return max(get_depth(node, set()) for node in import_graph.keys())
    
    def _is_project_import(self, import_name: str, files: Dict[str, str]) -> bool:
        """
        Check if import is from the project (vs external library)
        
        Args:
            import_name: Name being imported
            files: Dict of project files
            
        Returns:
            True if it's a project import
        """
        # Check if any file matches this module
        for filepath in files.keys():
            module = self._path_to_module(filepath)
            if module.startswith(import_name) or import_name.startswith(module):
                return True
        
        return False
    
    def _path_to_module(self, filepath: str) -> str:
        """
        Convert file path to module name
        
        Example: src/agents/base_agent.py -> src.agents.base_agent
        """
        # Remove .py extension
        path = filepath.replace('.py', '')
        # Replace path separators with dots
        path = path.replace('/', '.').replace('\\', '.')
        # Remove __init__
        path = path.replace('.__init__', '')
        return path
    
    def _check_naming_consistency(
        self,
        functions: Dict[str, List[Dict]]
    ) -> List[str]:
        """
        Check naming pattern consistency
        
        Args:
            functions: Function name -> list of definitions
            
        Returns:
            List of naming issues
        """
        issues = []
        
        # Group by prefix (get_, set_, create_, etc.)
        prefixes = defaultdict(set)
        for func_name in functions.keys():
            if '_' in func_name:
                prefix = func_name.split('_')[0]
                prefixes[prefix].add(func_name)
        
        # Check if related functions have consistent patterns
        for prefix, func_names in prefixes.items():
            if len(func_names) >= 3:
                # Check if they all follow similar patterns
                patterns = set()
                for name in func_names:
                    parts = name.split('_')
                    if len(parts) >= 2:
                        patterns.add(len(parts))
                
                if len(patterns) > 2:
                    issues.append(
                        f"Inconsistent naming for {prefix}_* functions "
                        f"({len(func_names)} functions with varying structures)"
                    )
        
        return issues
    
    def _generate_recommendations(
        self,
        issues: List[str],
        score: float
    ) -> List[str]:
        """
        Generate prioritized recommendations
        
        Args:
            issues: List of detected issues
            score: Overall score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Overall assessment
        if score >= 90:
            recommendations.append("🟢 EXCELLENT: Project connectivity is very well structured")
        elif score >= 75:
            recommendations.append("🟡 GOOD: Project structure is solid with minor improvements needed")
        elif score >= 60:
            recommendations.append("🟠 FAIR: Several connectivity issues need attention")
        else:
            recommendations.append("🔴 CRITICAL: Major architectural improvements required")
        
        # Categorize issues
        import_issues = [i for i in issues if 'import' in i.lower() or 'circular' in i.lower()]
        coupling_issues = [i for i in issues if 'coupling' in i.lower() or 'depend' in i.lower()]
        api_issues = [i for i in issues if 'api' in i.lower() or 'signature' in i.lower()]
        structure_issues = [i for i in issues if '__init__' in i or 'package' in i.lower()]
        
        # Add specific recommendations
        if import_issues:
            recommendations.append(f"1. Fix import structure ({len(import_issues)} issues)")
            if any('circular' in i.lower() for i in import_issues):
                recommendations.append("   • Break circular imports through refactoring or dependency injection")
        
        if coupling_issues:
            recommendations.append(f"2. Reduce module coupling ({len(coupling_issues)} issues)")
            recommendations.append("   • Apply dependency inversion and interface segregation")
        
        if api_issues:
            recommendations.append(f"3. Standardize API contracts ({len(api_issues)} issues)")
            recommendations.append("   • Ensure consistent function signatures across modules")
        
        if structure_issues:
            recommendations.append(f"4. Improve package structure ({len(structure_issues)} issues)")
            recommendations.append("   • Add missing __init__.py files and organize modules")
        
        # Limit to top 5 recommendations
        return recommendations[:5]


# Module-level logger is already imported
