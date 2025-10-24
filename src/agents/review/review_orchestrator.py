"""
Review Orchestrator - Unified Interface for All Code Reviewers

Coordinates and aggregates results from all 6 specialized code reviewers:
1. ReadabilityReviewer (15%) - Naming, formatting, documentation
2. LogicFlowReviewer (20%) - Control flow, error handling, logic quality
3. CodeConnectivityReviewer (15%) - Function cohesion, coupling, modularity
4. ProjectConnectivityReviewer (15%) - Integration with project structure
5. PerformanceReviewer (20%) - Time/space complexity, optimization
6. SecurityReviewer (15%) - Vulnerability detection, security best practices

Features:
- Single unified review_all() interface
- Parallel reviewer execution (optional)
- Automatic score aggregation with configurable weights
- Combined issues and suggestions from all reviewers
- Comprehensive statistics aggregation

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
import traceback

from ..base_agent import BaseAgent
from .readability_reviewer import ReadabilityReviewer
from .logic_flow_reviewer import LogicFlowReviewer
from .code_connectivity_reviewer import CodeConnectivityReviewer
from .project_connectivity_reviewer import ProjectConnectivityReviewer
from .performance_reviewer import PerformanceReviewer
from .security_reviewer import SecurityReviewer


class ReviewOrchestrator(BaseAgent):
    """
    Orchestrates all 6 code reviewers to provide comprehensive quality assessment.
    
    This class serves as a unified interface to run all reviewers and aggregate
    their results into a single comprehensive review report.
    
    Default Weights:
    - Readability: 15%
    - Logic Flow: 20%
    - Code Connectivity: 15%
    - Project Connectivity: 15%
    - Performance: 20%
    - Security: 15%
    """
    
    DEFAULT_WEIGHTS = {
        "readability": 0.15,
        "logic_flow": 0.20,
        "code_connectivity": 0.15,
        "project_connectivity": 0.15,
        "performance": 0.20,
        "security": 0.15
    }
    
    def __init__(
        self,
        readability_reviewer: ReadabilityReviewer,
        logic_flow_reviewer: LogicFlowReviewer,
        code_connectivity_reviewer: CodeConnectivityReviewer,
        project_connectivity_reviewer: ProjectConnectivityReviewer,
        performance_reviewer: PerformanceReviewer,
        security_reviewer: SecurityReviewer,
        weights: Optional[Dict[str, float]] = None,
        name: str = "review_orchestrator"
    ):
        """
        Initialize Review Orchestrator with all 6 reviewers.
        
        Args:
            readability_reviewer: ReadabilityReviewer instance
            logic_flow_reviewer: LogicFlowReviewer instance
            code_connectivity_reviewer: CodeConnectivityReviewer instance
            project_connectivity_reviewer: ProjectConnectivityReviewer instance
            performance_reviewer: PerformanceReviewer instance
            security_reviewer: SecurityReviewer instance
            weights: Optional custom weights for each reviewer (must sum to 1.0)
            name: Agent name (default: "review_orchestrator")
        """
        role = "Code Review Orchestrator - Coordinates All Quality Reviewers"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"
        )
        
        # Store reviewer instances
        self.reviewers = {
            "readability": readability_reviewer,
            "logic_flow": logic_flow_reviewer,
            "code_connectivity": code_connectivity_reviewer,
            "project_connectivity": project_connectivity_reviewer,
            "performance": performance_reviewer,
            "security": security_reviewer
        }
        
        # Set weights
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()
        
        # Validate weights
        self._validate_weights()
        
        logger.info(
            f"{self.name} initialized with 6 reviewers. "
            f"Weights: {self.weights}"
        )
    
    def _validate_weights(self):
        """Validate that weights are properly configured."""
        required_keys = set(self.DEFAULT_WEIGHTS.keys())
        provided_keys = set(self.weights.keys())
        
        if required_keys != provided_keys:
            missing = required_keys - provided_keys
            extra = provided_keys - required_keys
            error_msg = []
            if missing:
                error_msg.append(f"Missing weights: {missing}")
            if extra:
                error_msg.append(f"Extra weights: {extra}")
            raise ValueError(". ".join(error_msg))
        
        # Check sum is approximately 1.0
        weight_sum = sum(self.weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum}. "
                f"Weights: {self.weights}"
            )
        
        # Check all weights are non-negative
        for key, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Weight '{key}' must be non-negative, got {weight}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a review orchestration task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: "review_all" or specific review type
                - data: Task-specific parameters
                    - code: Source code to review
                    - parallel: Optional bool for parallel execution (default: True)
        
        Returns:
            Dictionary with aggregated review results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        if task_type == "review_all":
            files_content = data.get("files_content", {})
            parallel = data.get("parallel", True)
            project_context = data.get("project_context", {})
            
            return await self._review_all_async(files_content, parallel, project_context)
        
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    async def _review_python_file(
        self,
        file_path: str,
        code: str,
        parallel: bool = True,
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run all 6 reviewers and aggregate results.
        
        Args:
            code: Source code to review
            parallel: Whether to run reviewers in parallel (default: True)
            project_context: Optional project context for project connectivity reviewer
        
        Returns:
            Dictionary with aggregated results:
                - overall_score: Weighted average of all reviewer scores
                - individual_scores: Dict of scores from each reviewer
                - issues: Combined list of all issues found
                - suggestions: Combined list of all suggestions
                - statistics: Aggregated statistics from all reviewers
                - reviewer_results: Full results from each reviewer
        """
        logger.debug(f"Starting review for Python file: {file_path} (parallel={parallel})")
        
        if parallel:
            results = await self._run_parallel(file_path, code, project_context)
        else:
            results = await self._run_sequential(file_path, code, project_context)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        logger.debug(
            f"Python file review complete ({file_path}). Overall score: {aggregated['overall_score']:.1f}"
        )
        
        return aggregated
    
    async def _run_parallel(
        self,
        file_path: str,
        code: str,
        project_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run all reviewers in parallel using asyncio."""
        tasks = [
            self._run_readability(file_path, code),
            self._run_logic_flow(file_path, code),
            self._run_code_connectivity(file_path, code),
            self._run_project_connectivity(file_path, code, project_context),
            self._run_performance(file_path, code),
            self._run_security(file_path, code)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results to reviewer names
        reviewer_names = list(self.reviewers.keys())
        
        result_dict = {}
        for name, result in zip(reviewer_names, results):
            if isinstance(result, Exception):
                logger.error(f"{name} reviewer failed for file {file_path}: {result}")
                result_dict[name] = {
                    "success": False,
                    "error": str(result),
                    "score": 0,
                    "issues": [],
                    "suggestions": []
                }
            else:
                result_dict[name] = result
        
        return result_dict
    
    async def _run_sequential(
        self,
        file_path: str,
        code: str,
        project_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run all reviewers sequentially."""
        results = {}
        
        try:
            results["readability"] = await self._run_readability(file_path, code)
        except Exception as e:
            logger.error(f"Readability reviewer failed: {e}")
            results["readability"] = {"success": False, "error": str(e), "score": 0}
        
        try:
            results["logic_flow"] = await self._run_logic_flow(file_path, code)
        except Exception as e:
            logger.error(f"Logic flow reviewer failed: {e}")
            results["logic_flow"] = {"success": False, "error": str(e), "score": 0}
        
        try:
            results["code_connectivity"] = await self._run_code_connectivity(file_path, code)
        except Exception as e:
            logger.error(f"Code connectivity reviewer failed: {e}")
            results["code_connectivity"] = {"success": False, "error": str(e), "score": 0}
        
        try:
            results["project_connectivity"] = await self._run_project_connectivity(
                file_path, code, project_context
            )
        except Exception as e:
            logger.error(f"Project connectivity reviewer failed: {e}")
            results["project_connectivity"] = {"success": False, "error": str(e), "score": 0}
        
        try:
            results["performance"] = await self._run_performance(file_path, code)
        except Exception as e:
            logger.error(f"Performance reviewer failed: {e}")
            results["performance"] = {"success": False, "error": str(e), "score": 0}
        
        try:
            results["security"] = await self._run_security(file_path, code)
        except Exception as e:
            logger.error(f"Security reviewer failed: {e}")
            results["security"] = {"success": False, "error": str(e), "score": 0}
        
        return results

        
        
        
    # --- Individual runner helpers ---
    
    async def _run_readability(self, file_path: str, code: str) -> Dict[str, Any]:
        return await self.reviewers["readability"].execute_task({
            "task_type": "analyze_readability", "data": {"file_path": file_path, "code": code}
        })
    
    async def _run_logic_flow(self, file_path: str, code: str) -> Dict[str, Any]:
        return await self.reviewers["logic_flow"].execute_task({
            "task_type": "analyze_logic_flow", "data": {"file_path": file_path, "code": code}
        })
    
    async def _run_code_connectivity(self, file_path: str, code: str) -> Dict[str, Any]:
        return await self.reviewers["code_connectivity"].execute_task({
            "task_type": "analyze_connectivity", "data": {"file_path": file_path, "code": code}
        })
    
    async def _run_project_connectivity(
        self, file_path: str, code: str, project_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return await self.reviewers["project_connectivity"].execute_task({
            "task_type": "analyze_project",
            "data": {"file_path": file_path, "code": code, "project_context": project_context or {}}
        })
    
    async def _run_performance(self, file_path: str, code: str) -> Dict[str, Any]:
        return await self.reviewers["performance"].execute_task({
            "task_type": "review_performance", "data": {"file_path": file_path, "code": code}
        })
    
    async def _run_security(self, file_path: str, code: str) -> Dict[str, Any]:
        return await self.reviewers["security"].execute_task({
            "task_type": "review_security", "data": {"file_path": file_path, "code": code}
        })

    

    
    def _aggregate_results(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from all reviewers.
        
        Args:
            results: Dictionary mapping reviewer name to results
        
        Returns:
            Aggregated results with overall score, combined issues, etc.
        """
        # Calculate weighted overall score
        overall_score = 0.0
        individual_scores = {}
        
        for reviewer_name, weight in self.weights.items():
            result = results.get(reviewer_name, {})
            score = result.get("score", 0)
            individual_scores[reviewer_name] = score
            overall_score += score * weight
        
        # Combine issues from all reviewers
        all_issues = []
        for reviewer_name, result in results.items():
            issues = result.get("issues", [])
            # Tag issues with reviewer name
            for issue in issues:
                issue_copy = issue.copy() if isinstance(issue, dict) else {"description": str(issue)}
                issue_copy["reviewer"] = reviewer_name
                all_issues.append(issue_copy)
        
        # Combine suggestions from all reviewers
        all_suggestions = []
        for reviewer_name, result in results.items():
            suggestions = result.get("suggestions", [])
            # Tag suggestions with reviewer name
            for suggestion in suggestions:
                suggestion_copy = suggestion.copy() if isinstance(suggestion, dict) else {"description": str(suggestion)}
                suggestion_copy["reviewer"] = reviewer_name
                all_suggestions.append(suggestion_copy)
        
        # Aggregate statistics
        aggregated_stats = self._aggregate_statistics(results)
        
        return {
            "success": True,
            "overall_score": round(overall_score, 2),
            "individual_scores": individual_scores,
            "weights": self.weights.copy(),
            "issues": all_issues,
            "suggestions": all_suggestions,
            "statistics": aggregated_stats,
            "reviewer_results": results
        }
    
    def _aggregate_statistics(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate statistics from all reviewers.
        
        Args:
            results: Dictionary mapping reviewer name to results
        
        Returns:
            Combined statistics dictionary
        """
        stats = {
            "total_reviewers": len(results),
            "successful_reviews": sum(
                1 for r in results.values() if r.get("success", False)
            ),
            "failed_reviews": sum(
                1 for r in results.values() if not r.get("success", False)
            ),
            "total_issues": 0,
            "total_suggestions": 0,
            "issues_by_reviewer": {},
            "suggestions_by_reviewer": {},
            "reviewer_statistics": {}
        }
        
        for reviewer_name, result in results.items():
            issues_count = len(result.get("issues", []))
            suggestions_count = len(result.get("suggestions", []))
            
            stats["total_issues"] += issues_count
            stats["total_suggestions"] += suggestions_count
            stats["issues_by_reviewer"][reviewer_name] = issues_count
            stats["suggestions_by_reviewer"][reviewer_name] = suggestions_count
            
            # Include reviewer-specific statistics if available
            if "statistics" in result:
                stats["reviewer_statistics"][reviewer_name] = result["statistics"]
        
        return stats
    
    # Synchronous convenience method
    def review_all(
        self,
        files_content: Dict[str, str],
        parallel: bool = True,
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the review operation on multiple files.
        """
        logger.info(f"Starting synchronous review_all for {len(files_content)} files.")
        try:
            # Check if an event loop is already running
            try:
                loop = asyncio.get_running_loop()
                logger.debug("Reusing existing event loop for review_all.")
                return loop.run_until_complete(self._review_all_async(files_content, parallel, project_context))
            except RuntimeError:
                logger.debug("No running event loop. Creating new one for review_all.")
                return asyncio.run(self._review_all_async(files_content, parallel, project_context))
        except Exception as e:
            logger.error(f"Error in review_all: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "overall_score": 0.0,
                "message": f"Review orchestration failed: {e}",
                "issues": [],
                "suggestions": []
            }

    async def _review_all_async(
        self,
        files_content: Dict[str, str],
        parallel: bool = True,
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously review all files and aggregate the results.
        
        --- THIS IS THE FIX ---
        Filters files and only sends .py files to Python-specific reviewers.
        """
        
        # --- FIX: Filter for Python files ---
        python_files = {
            fp: code for fp, code in files_content.items() 
            if fp.endswith(".py") and code.strip()
        }
        
        non_python_files = {
            fp: code for fp, code in files_content.items() 
            if not fp.endswith(".py")
        }
        
        if not python_files:
            logger.warning("No Python files found to review. Skipping code review.")
            return {
                "success": True,
                "overall_score": 100.0, # No errors found
                "message": "No Python files to review.",
                "issues": [],
                "suggestions": [],
                "statistics": {"total_files_reviewed": 0, "total_issues": 0, "total_suggestions": 0}
            }

        logger.info(f"Reviewing {len(python_files)} Python files. ({len(non_python_files)} non-Python files skipped)")
        # --- END FIX ---

        file_reports = {}
        for file_path, code in python_files.items():
            logger.debug(f"Reviewing: {file_path}")
            try:
                # Pass the *single file's code* to the review method
                file_reports[file_path] = await self._review_python_file(file_path, code, parallel, project_context)
            except Exception as e:
                logger.error(f"Failed to review file {file_path}: {e}")
                file_reports[file_path] = {
                    "success": False,
                    "overall_score": 0.0,
                    "error": str(e),
                    "issues": [],
                    "suggestions": []
                }
        
        return self._aggregate_file_reports(file_reports)

    def _aggregate_file_reports(self, file_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate reports from multiple files into a single project report.
        """
        if not file_reports:
            return {"overall_score": 100.0, "issues": [], "suggestions": [], "statistics": {}}

        total_score = 0
        all_issues = []
        all_suggestions = []
        
        for file_path, report in file_reports.items():
            total_score += report.get("overall_score", 0)
            for issue in report.get("issues", []):
                issue["file_path"] = file_path
                all_issues.append(issue)
            for suggestion in report.get("suggestions", []):
                suggestion["file_path"] = file_path
                all_suggestions.append(suggestion)

        avg_score = total_score / len(file_reports) if file_reports else 100.0
        return {
            "success": True,
            "overall_score": round(avg_score, 2),
            "individual_file_reports": file_reports,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "statistics": {
                "total_files_reviewed": len(file_reports),
                "total_issues": len(all_issues),
                "total_suggestions": len(all_suggestions),
            }
        }
