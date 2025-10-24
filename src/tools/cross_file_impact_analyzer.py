"""
Cross-File Impact Analyzer

Predicts which files in a project are affected by changes to a given file.
Uses dependency graph analysis to determine:
1. Direct dependencies (files that import the changed file)
2. Transitive dependencies (files that depend on direct dependencies)
3. Reverse dependencies (files imported by the changed file)
4. Impact radius (how far the changes might propagate)
5. Affected tests (test files that should be run)

This tool is essential for:
- Intelligent test selection
- Change impact assessment
- Code review scope determination
- Refactoring risk analysis
- Deployment planning

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ImpactLevel(Enum):
    """Level of impact a change might have."""
    DIRECT = "direct"  # File directly imports or is imported by changed file
    TRANSITIVE = "transitive"  # File indirectly affected through dependency chain
    TEST = "test"  # Test file for changed file or affected files
    POTENTIAL = "potential"  # Might be affected based on heuristics


@dataclass
class FileImpact:
    """
    Information about how a file is impacted by changes.
    
    Attributes:
        file_path: Path to the impacted file
        impact_level: Level of impact
        distance: Number of hops in dependency graph (0 = changed file itself)
        dependency_chain: List of files in the dependency path
        reasons: Why this file is affected
        confidence: Confidence score (0-1) that file is actually affected
    """
    file_path: str
    impact_level: ImpactLevel
    distance: int
    dependency_chain: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "impact_level": self.impact_level.value,
            "distance": self.distance,
            "dependency_chain": self.dependency_chain,
            "reasons": self.reasons,
            "confidence": self.confidence
        }


@dataclass
class ImpactAnalysis:
    """
    Complete impact analysis result.
    
    Attributes:
        changed_file: File that was changed
        total_affected: Total number of affected files
        direct_impacts: Files directly depending on changed file
        transitive_impacts: Files indirectly affected
        reverse_impacts: Files that changed file depends on
        affected_tests: Test files that should be run
        impact_radius: Maximum distance in dependency graph
        confidence_score: Overall confidence in analysis (0-1)
        recommendations: Suggested actions
    """
    changed_file: str
    total_affected: int
    direct_impacts: List[FileImpact]
    transitive_impacts: List[FileImpact]
    reverse_impacts: List[FileImpact]
    affected_tests: List[FileImpact]
    impact_radius: int
    confidence_score: float
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "changed_file": self.changed_file,
            "total_affected": self.total_affected,
            "direct_impacts": [fi.to_dict() for fi in self.direct_impacts],
            "transitive_impacts": [fi.to_dict() for fi in self.transitive_impacts],
            "reverse_impacts": [fi.to_dict() for fi in self.reverse_impacts],
            "affected_tests": [fi.to_dict() for fi in self.affected_tests],
            "impact_radius": self.impact_radius,
            "confidence_score": self.confidence_score,
            "recommendations": self.recommendations
        }


class CrossFileImpactAnalyzer:
    """
    Analyzes the impact of changes across files in a project.
    
    Uses dependency graph analysis to determine:
    - Which files are affected by changes
    - How far changes might propagate
    - Which tests need to be run
    - Risk level of changes
    
    Features:
    - Forward impact analysis (who depends on this file)
    - Reverse impact analysis (what this file depends on)
    - Test mapping (which tests cover which files)
    - Confidence scoring based on coupling strength
    - Impact radius calculation
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        include_tests: bool = True,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize the impact analyzer.
        
        Args:
            max_depth: Maximum depth for transitive dependency analysis
            include_tests: Whether to identify affected test files
            confidence_threshold: Minimum confidence for including impacts
        """
        self.max_depth = max_depth
        self.include_tests = include_tests
        self.confidence_threshold = confidence_threshold
        
        # Dependency graphs
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> files it imports
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> files that import it
        self.test_mapping: Dict[str, Set[str]] = defaultdict(set)  # file -> test files
        
        logger.info(
            f"CrossFileImpactAnalyzer initialized: "
            f"max_depth={max_depth}, include_tests={include_tests}, "
            f"confidence_threshold={confidence_threshold}"
        )
    
    def build_dependency_graph(
        self,
        files: Dict[str, str],
        project_root: str = ""
    ) -> None:
        """
        Build dependency graphs from project files.
        
        Args:
            files: Dict mapping file paths to code content
            project_root: Root directory of project
        """
        logger.info(f"Building dependency graph for {len(files)} files...")
        
        # Reset graphs
        self.import_graph.clear()
        self.reverse_graph.clear()
        self.test_mapping.clear()
        
        # Build import graph
        for file_path, code in files.items():
            try:
                module_name = self._get_module_name(file_path, project_root)
                imports = self._extract_imports(code, project_root)
                
                for import_name in imports:
                    self.import_graph[module_name].add(import_name)
                    self.reverse_graph[import_name].add(module_name)
                
                # Identify test files
                if self._is_test_file(file_path):
                    # Map test to the modules it tests
                    tested_modules = self._infer_tested_modules(file_path, imports, project_root)
                    for tested_module in tested_modules:
                        self.test_mapping[tested_module].add(module_name)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(
            f"Dependency graph built: "
            f"{len(self.import_graph)} modules, "
            f"{sum(len(deps) for deps in self.import_graph.values())} dependencies, "
            f"{len(self.test_mapping)} tested modules"
        )
    
    def analyze_impact(
        self,
        changed_file: str,
        change_type: str = "modification",
        project_root: str = ""
    ) -> ImpactAnalysis:
        """
        Analyze the impact of changes to a file.
        
        Args:
            changed_file: Path to the file that changed
            change_type: Type of change (modification, addition, deletion)
            project_root: Root directory of project
            
        Returns:
            ImpactAnalysis with affected files and recommendations
        """
        logger.info(f"Analyzing impact of {change_type} to {changed_file}...")
        
        module_name = self._get_module_name(changed_file, project_root)
        
        # Find directly affected files (forward dependencies)
        direct_impacts = self._find_direct_impacts(module_name)
        
        # Find transitively affected files
        transitive_impacts = self._find_transitive_impacts(module_name, direct_impacts)
        
        # Find reverse dependencies (what this file depends on)
        reverse_impacts = self._find_reverse_impacts(module_name)
        
        # Find affected tests
        affected_tests = self._find_affected_tests(
            module_name, 
            direct_impacts + transitive_impacts
        )
        
        # Calculate impact radius
        impact_radius = max(
            (fi.distance for fi in direct_impacts + transitive_impacts),
            default=0
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            direct_impacts, 
            transitive_impacts, 
            affected_tests
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            change_type,
            len(direct_impacts),
            len(transitive_impacts),
            len(affected_tests),
            impact_radius
        )
        
        total_affected = (
            len(direct_impacts) + 
            len(transitive_impacts) + 
            len(reverse_impacts) + 
            len(affected_tests)
        )
        
        return ImpactAnalysis(
            changed_file=changed_file,
            total_affected=total_affected,
            direct_impacts=direct_impacts,
            transitive_impacts=transitive_impacts,
            reverse_impacts=reverse_impacts,
            affected_tests=affected_tests,
            impact_radius=impact_radius,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
    
    def _find_direct_impacts(self, module_name: str) -> List[FileImpact]:
        """
        Find files directly depending on the changed file.
        
        Args:
            module_name: Module that changed
            
        Returns:
            List of FileImpact for directly affected files
        """
        direct_impacts = []
        
        for dependent in self.reverse_graph.get(module_name, set()):
            impact = FileImpact(
                file_path=dependent,
                impact_level=ImpactLevel.DIRECT,
                distance=1,
                dependency_chain=[module_name, dependent],
                reasons=[f"Directly imports {module_name}"],
                confidence=1.0
            )
            direct_impacts.append(impact)
        
        return direct_impacts
    
    def _find_transitive_impacts(
        self, 
        module_name: str, 
        direct_impacts: List[FileImpact]
    ) -> List[FileImpact]:
        """
        Find files transitively affected through dependency chains.
        
        Args:
            module_name: Module that changed
            direct_impacts: Already identified direct impacts
            
        Returns:
            List of FileImpact for transitively affected files
        """
        transitive_impacts = []
        visited = {module_name}
        visited.update(fi.file_path for fi in direct_impacts)
        
        # BFS from directly affected files
        queue = deque([(fi.file_path, fi.dependency_chain, 2) for fi in direct_impacts])
        
        while queue and len(transitive_impacts) < 100:  # Limit to prevent explosion
            current, chain, distance = queue.popleft()
            
            if distance > self.max_depth:
                continue
            
            for dependent in self.reverse_graph.get(current, set()):
                if dependent not in visited:
                    visited.add(dependent)
                    
                    new_chain = chain + [dependent]
                    confidence = max(0.3, 1.0 / distance)  # Decrease confidence with distance
                    
                    if confidence >= self.confidence_threshold:
                        impact = FileImpact(
                            file_path=dependent,
                            impact_level=ImpactLevel.TRANSITIVE,
                            distance=distance,
                            dependency_chain=new_chain,
                            reasons=[
                                f"Transitively depends on {module_name} through {len(new_chain)-2} intermediaries"
                            ],
                            confidence=confidence
                        )
                        transitive_impacts.append(impact)
                        
                        # Continue BFS
                        queue.append((dependent, new_chain, distance + 1))
        
        return transitive_impacts
    
    def _find_reverse_impacts(self, module_name: str) -> List[FileImpact]:
        """
        Find files that the changed file depends on (reverse dependencies).
        
        These might need updates if the changed file's interface changes.
        
        Args:
            module_name: Module that changed
            
        Returns:
            List of FileImpact for reverse dependencies
        """
        reverse_impacts = []
        
        for imported in self.import_graph.get(module_name, set()):
            impact = FileImpact(
                file_path=imported,
                impact_level=ImpactLevel.DIRECT,
                distance=1,
                dependency_chain=[module_name, imported],
                reasons=[f"Imported by {module_name}"],
                confidence=0.7  # Lower confidence - might not need changes
            )
            reverse_impacts.append(impact)
        
        return reverse_impacts
    
    def _find_affected_tests(
        self, 
        module_name: str, 
        affected_files: List[FileImpact]
    ) -> List[FileImpact]:
        """
        Find test files that should be run based on affected files.
        
        Args:
            module_name: Module that changed
            affected_files: Files affected by the change
            
        Returns:
            List of FileImpact for affected test files
        """
        if not self.include_tests:
            return []
        
        affected_tests = []
        seen_tests = set()
        
        # Tests for the changed module
        for test in self.test_mapping.get(module_name, set()):
            if test not in seen_tests:
                seen_tests.add(test)
                impact = FileImpact(
                    file_path=test,
                    impact_level=ImpactLevel.TEST,
                    distance=1,
                    dependency_chain=[module_name, test],
                    reasons=[f"Tests {module_name}"],
                    confidence=1.0
                )
                affected_tests.append(impact)
        
        # Tests for affected modules
        for file_impact in affected_files:
            for test in self.test_mapping.get(file_impact.file_path, set()):
                if test not in seen_tests:
                    seen_tests.add(test)
                    impact = FileImpact(
                        file_path=test,
                        impact_level=ImpactLevel.TEST,
                        distance=file_impact.distance + 1,
                        dependency_chain=file_impact.dependency_chain + [test],
                        reasons=[f"Tests affected module {file_impact.file_path}"],
                        confidence=file_impact.confidence * 0.9
                    )
                    affected_tests.append(impact)
        
        return affected_tests
    
    def _calculate_confidence(
        self,
        direct_impacts: List[FileImpact],
        transitive_impacts: List[FileImpact],
        affected_tests: List[FileImpact]
    ) -> float:
        """
        Calculate overall confidence in the impact analysis.
        
        Args:
            direct_impacts: Direct impact files
            transitive_impacts: Transitive impact files
            affected_tests: Affected test files
            
        Returns:
            Confidence score (0-1)
        """
        if not direct_impacts and not transitive_impacts:
            return 1.0  # High confidence when no impacts
        
        # Weight by impact type
        total_confidence = (
            sum(fi.confidence * 1.0 for fi in direct_impacts) +
            sum(fi.confidence * 0.5 for fi in transitive_impacts) +
            sum(fi.confidence * 0.8 for fi in affected_tests)
        )
        
        total_weight = (
            len(direct_impacts) * 1.0 +
            len(transitive_impacts) * 0.5 +
            len(affected_tests) * 0.8
        )
        
        if total_weight == 0:
            return 1.0
        
        return min(1.0, total_confidence / total_weight)
    
    def _generate_recommendations(
        self,
        change_type: str,
        direct_count: int,
        transitive_count: int,
        test_count: int,
        impact_radius: int
    ) -> List[str]:
        """
        Generate recommendations based on impact analysis.
        
        Args:
            change_type: Type of change
            direct_count: Number of directly affected files
            transitive_count: Number of transitively affected files
            impact_radius: Maximum dependency distance
            test_count: Number of affected tests
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Overall impact assessment
        total_impact = direct_count + transitive_count
        
        if total_impact == 0:
            recommendations.append("✅ No dependencies affected - change is isolated")
        elif total_impact <= 3:
            recommendations.append("✅ LOW IMPACT: Small number of files affected")
        elif total_impact <= 10:
            recommendations.append("⚠️ MEDIUM IMPACT: Moderate number of files affected")
        else:
            recommendations.append("🔴 HIGH IMPACT: Many files affected - proceed with caution")
        
        # Direct impacts
        if direct_count > 0:
            recommendations.append(
                f"📋 Review {direct_count} directly dependent file(s) for compatibility"
            )
        
        # Transitive impacts
        if transitive_count > 0:
            recommendations.append(
                f"📋 Consider {transitive_count} transitively affected file(s)"
            )
        
        # Impact radius
        if impact_radius > 3:
            recommendations.append(
                f"⚠️ High impact radius ({impact_radius}) - changes propagate far"
            )
        
        # Testing
        if test_count == 0:
            recommendations.append(
                "❌ No tests found - consider adding tests for changed file"
            )
        elif test_count <= 5:
            recommendations.append(
                f"✅ Run {test_count} affected test(s) to verify changes"
            )
        else:
            recommendations.append(
                f"📋 Run {test_count} affected tests - consider focused testing"
            )
        
        # Change type specific
        if change_type == "deletion":
            recommendations.append(
                "🔴 DELETION: Verify all dependencies are updated or removed"
            )
        elif change_type == "addition":
            recommendations.append(
                "✅ ADDITION: New file - minimal risk to existing code"
            )
        elif change_type == "modification":
            if direct_count > 0:
                recommendations.append(
                    "📋 MODIFICATION: Check if interface/API changes break dependents"
                )
        
        return recommendations
    
    def _get_module_name(self, file_path: str, project_root: str) -> str:
        """
        Convert file path to Python module name.
        
        Args:
            file_path: Path to Python file
            project_root: Project root directory
            
        Returns:
            Module name (e.g., "src.agents.base_agent")
        """
        # Remove project root if present
        if project_root and file_path.startswith(project_root):
            file_path = file_path[len(project_root):].lstrip(os.sep)
        
        # Convert to module path
        module_path = Path(file_path)
        if module_path.suffix == '.py':
            module_path = module_path.with_suffix('')
        
        # Convert path separators to dots
        module_name = str(module_path).replace(os.sep, '.')
        
        # Remove __init__ if present
        if module_name.endswith('.__init__'):
            module_name = module_name[:-9]
        
        return module_name
    
    def _extract_imports(self, code: str, project_root: str) -> Set[str]:
        """
        Extract import statements from Python code.
        
        Args:
            code: Python source code
            project_root: Project root directory
            
        Returns:
            Set of imported module names
        """
        imports = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        
        except SyntaxError:
            pass  # Ignore syntax errors
        
        return imports
    
    def _is_test_file(self, file_path: str) -> bool:
        """
        Check if a file is a test file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is a test file
        """
        file_name = Path(file_path).name
        return (
            file_name.startswith('test_') or
            file_name.endswith('_test.py') or
            'test' in Path(file_path).parts
        )
    
    def _infer_tested_modules(
        self, 
        test_file: str, 
        imports: Set[str], 
        project_root: str
    ) -> Set[str]:
        """
        Infer which modules a test file is testing.
        
        Args:
            test_file: Path to test file
            imports: Imports from the test file
            project_root: Project root directory
            
        Returns:
            Set of module names being tested
        """
        tested_modules = set()
        
        # Heuristic: test_xyz.py tests xyz.py
        test_name = Path(test_file).stem
        if test_name.startswith('test_'):
            module_name = test_name[5:]  # Remove 'test_' prefix
            tested_modules.add(module_name)
        
        # Also include non-test imports from same project
        for imported in imports:
            if not imported.startswith(('unittest', 'pytest', 'mock')):
                tested_modules.add(imported)
        
        return tested_modules
    
    def get_summary(self, analysis: ImpactAnalysis) -> str:
        """
        Generate human-readable summary of impact analysis.
        
        Args:
            analysis: Impact analysis result
            
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            "CROSS-FILE IMPACT ANALYSIS",
            "=" * 70,
            f"Changed File: {analysis.changed_file}",
            f"Total Affected: {analysis.total_affected} files",
            f"Impact Radius: {analysis.impact_radius} hops",
            f"Confidence: {analysis.confidence_score:.1%}",
            "",
            f"DIRECT IMPACTS ({len(analysis.direct_impacts)}):",
        ]
        
        for impact in analysis.direct_impacts[:10]:  # Show max 10
            lines.append(f"  • {impact.file_path} (confidence: {impact.confidence:.1%})")
            for reason in impact.reasons:
                lines.append(f"    - {reason}")
        
        if len(analysis.direct_impacts) > 10:
            lines.append(f"  ... and {len(analysis.direct_impacts) - 10} more")
        
        if analysis.transitive_impacts:
            lines.append("")
            lines.append(f"TRANSITIVE IMPACTS ({len(analysis.transitive_impacts)}):")
            for impact in analysis.transitive_impacts[:5]:  # Show max 5
                lines.append(
                    f"  • {impact.file_path} (distance: {impact.distance}, "
                    f"confidence: {impact.confidence:.1%})"
                )
            
            if len(analysis.transitive_impacts) > 5:
                lines.append(f"  ... and {len(analysis.transitive_impacts) - 5} more")
        
        if analysis.affected_tests:
            lines.append("")
            lines.append(f"AFFECTED TESTS ({len(analysis.affected_tests)}):")
            for impact in analysis.affected_tests[:10]:
                lines.append(f"  • {impact.file_path}")
            
            if len(analysis.affected_tests) > 10:
                lines.append(f"  ... and {len(analysis.affected_tests) - 10} more")
        
        if analysis.recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(analysis.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
