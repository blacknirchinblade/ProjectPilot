"""
Review Agent - Code Quality Analysis & Review (DEPRECATED)

⚠️ DEPRECATION WARNING ⚠️
This ReviewAgent is DEPRECATED and will be removed in a future version.
Please use ReviewOrchestrator with specialized reviewers instead:

from src.agents.review import ReviewOrchestrator
orchestrator = ReviewOrchestrator()
result = orchestrator.review_all(code, project_files)

The new architecture provides better quality through 6 specialized reviewers:
- ReadabilityReviewer
- LogicFlowReviewer  
- CodeConnectivityReviewer
- ProjectConnectivityReviewer
- PerformanceReviewer
- SecurityReviewer

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
import warnings
from typing import Dict, List, Any, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent


class ReviewAgent(BaseAgent):
    """
    Review Agent for comprehensive code quality analysis.
    
    ⚠️ DEPRECATED: Use ReviewOrchestrator instead ⚠️
    
    This class is maintained for backward compatibility only.
    New code should use ReviewOrchestrator with specialized reviewers.
    
    Migration example:
        # Old way (deprecated)
        review_agent = ReviewAgent()
        result = await review_agent.review_code(code)
        
        # New way (recommended)
        from agents.review import ReviewOrchestrator
        orchestrator = ReviewOrchestrator()
        result = orchestrator.review_all(code, project_files)
    
    Responsibilities:
    - Analyze code quality and assign scores
    - Identify bugs, security issues, performance problems
    - Check code style, best practices, and patterns
    - Suggest improvements and refactoring opportunities
    - Generate detailed review reports
    
    Uses temperature=0.2 for precise, consistent reviews.
    """
    
    def __init__(self, name: str = "review_agent"):
        """
        Initialize Review Agent.
        
        ⚠️ DEPRECATION WARNING ⚠️
        ReviewAgent is deprecated. Use ReviewOrchestrator instead.
        
        Args:
            name: Agent name (default: "review_agent")
        """
        # Issue deprecation warning
        warnings.warn(
            "ReviewAgent is deprecated and will be removed in a future version. "
            "Please use ReviewOrchestrator with specialized reviewers instead. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        
        role = "Expert Code Reviewer for Python ML/DL Projects"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"  # Uses temperature 0.2
        )
        logger.warning(
            f"{self.name} initialized (DEPRECATED - use ReviewOrchestrator instead)"
        )
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a review task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of review task
                - data: Task-specific parameters
        
        Returns:
            Dictionary with review results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        try:
            if task_type == "review_code":
                return await self.review_code(
                    code=data.get("code"),
                    context=data.get("context", ""),
                    focus_areas=data.get("focus_areas", [])
                )
            
            elif task_type == "analyze_quality":
                return await self.analyze_quality(
                    code=data.get("code"),
                    metrics=data.get("metrics", [])
                )
            
            elif task_type == "check_security":
                return await self.check_security(
                    code=data.get("code")
                )
            
            elif task_type == "check_performance":
                return await self.check_performance(
                    code=data.get("code")
                )
            
            elif task_type == "check_best_practices":
                return await self.check_best_practices(
                    code=data.get("code"),
                    language=data.get("language", "python")
                )
            
            elif task_type == "generate_review_report":
                return await self.generate_review_report(
                    code=data.get("code"),
                    include_suggestions=data.get("include_suggestions", True)
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing review task '{task_type}': {e}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def review_code(
        self,
        code: str,
        context: str = "",
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive code review.
        
        Args:
            code: Code to review
            context: Additional context about the code
            focus_areas: Specific areas to focus on (e.g., "performance", "security")
        
        Returns:
            Dictionary with review results including score and issues
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for review"
            }
        
        logger.info(f"{self.name} reviewing code ({len(code)} chars)")
        
        # Build focus areas context
        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nFocus specifically on: {', '.join(focus_areas)}"
        
        # Get review prompt
        prompt_data = {
            "code": code,
            "context": context,
            "focus_areas": focus_text
        }
        
        prompt = self.get_prompt("review_prompts", "review_code", prompt_data)
        
        # Generate review
        response = await self.generate_response(prompt)
        
        # Parse review response
        review_data = self._parse_review_response(response)
        
        return {
            "status": "success",
            "task": "review_code",
            "code_length": len(code),
            "review": response,
            "score": review_data.get("score", 0),
            "issues": review_data.get("issues", []),
            "suggestions": review_data.get("suggestions", []),
            "focus_areas": focus_areas or []
        }
    
    async def analyze_quality(
        self,
        code: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze code quality with specific metrics.
        
        Args:
            code: Code to analyze
            metrics: Quality metrics to evaluate (e.g., "readability", "maintainability")
        
        Returns:
            Dictionary with quality scores and analysis
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for quality analysis"
            }
        
        logger.info(f"{self.name} analyzing code quality")
        
        # Default metrics if not provided
        if not metrics:
            metrics = [
                "readability",
                "maintainability",
                "complexity",
                "documentation",
                "error_handling",
                "type_safety"
            ]
        
        prompt_data = {
            "code": code,
            "metrics": ", ".join(metrics)
        }
        
        prompt = self.get_prompt("review_prompts", "analyze_quality", prompt_data)
        response = await self.generate_response(prompt)
        
        # Parse quality scores
        quality_scores = self._parse_quality_scores(response)
        
        return {
            "status": "success",
            "task": "analyze_quality",
            "metrics_evaluated": metrics,
            "analysis": response,
            "scores": quality_scores,
            "overall_score": quality_scores.get("overall", 0)
        }
    
    async def check_security(self, code: str) -> Dict[str, Any]:
        """
        Perform security analysis on code.
        
        Args:
            code: Code to check for security issues
        
        Returns:
            Dictionary with security findings
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for security check"
            }
        
        logger.info(f"{self.name} performing security analysis")
        
        prompt_data = {"code": code}
        prompt = self.get_prompt("review_prompts", "check_security", prompt_data)
        response = await self.generate_response(prompt)
        
        # Parse security issues
        security_issues = self._parse_security_issues(response)
        
        return {
            "status": "success",
            "task": "check_security",
            "analysis": response,
            "vulnerabilities": security_issues.get("vulnerabilities", []),
            "risk_level": security_issues.get("risk_level", "unknown"),
            "recommendations": security_issues.get("recommendations", [])
        }
    
    async def check_performance(self, code: str) -> Dict[str, Any]:
        """
        Analyze code for performance issues.
        
        Args:
            code: Code to analyze for performance
        
        Returns:
            Dictionary with performance analysis
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for performance check"
            }
        
        logger.info(f"{self.name} analyzing performance")
        
        prompt_data = {"code": code}
        prompt = self.get_prompt("review_prompts", "check_performance", prompt_data)
        response = await self.generate_response(prompt)
        
        # Parse performance issues
        perf_issues = self._parse_performance_issues(response)
        
        return {
            "status": "success",
            "task": "check_performance",
            "analysis": response,
            "bottlenecks": perf_issues.get("bottlenecks", []),
            "optimizations": perf_issues.get("optimizations", []),
            "complexity_analysis": perf_issues.get("complexity", "")
        }
    
    async def check_best_practices(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Check code against best practices and style guidelines.
        
        Args:
            code: Code to check
            language: Programming language (default: "python")
        
        Returns:
            Dictionary with best practice violations and suggestions
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for best practices check"
            }
        
        logger.info(f"{self.name} checking best practices for {language}")
        
        prompt_data = {
            "code": code,
            "language": language
        }
        
        prompt = self.get_prompt("review_prompts", "check_best_practices", prompt_data)
        response = await self.generate_response(prompt)
        
        # Parse best practice violations
        violations = self._parse_best_practices(response)
        
        return {
            "status": "success",
            "task": "check_best_practices",
            "language": language,
            "analysis": response,
            "violations": violations.get("violations", []),
            "recommendations": violations.get("recommendations", []),
            "style_score": violations.get("style_score", 0)
        }
    
    async def generate_review_report(
        self,
        code: str,
        include_suggestions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive review report.
        
        Args:
            code: Code to review
            include_suggestions: Whether to include improvement suggestions
        
        Returns:
            Dictionary with complete review report
        """
        if not code:
            return {
                "status": "error",
                "message": "Code is required for review report"
            }
        
        logger.info(f"{self.name} generating comprehensive review report")
        
        prompt_data = {
            "code": code,
            "include_suggestions": "Include detailed suggestions for improvement." if include_suggestions else ""
        }
        
        prompt = self.get_prompt("review_prompts", "generate_review_report", prompt_data)
        response = await self.generate_response(prompt)
        
        # Parse comprehensive report
        report = self._parse_review_report(response)
        
        return {
            "status": "success",
            "task": "generate_review_report",
            "report": response,
            "summary": report.get("summary", ""),
            "overall_score": report.get("overall_score", 0),
            "categories": report.get("categories", {}),
            "critical_issues": report.get("critical_issues", []),
            "suggestions": report.get("suggestions", []) if include_suggestions else []
        }
    
    # ==================== Helper Methods ====================
    
    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """
        Parse review response to extract score and issues.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with parsed review data
        """
        result = {
            "score": 0,
            "issues": [],
            "suggestions": []
        }
        
        # Extract score (0-100)
        score_match = re.search(r'(?:score|rating)[:\s]+(\d+)', response, re.IGNORECASE)
        if score_match:
            result["score"] = int(score_match.group(1))
        
        # Extract issues (lines starting with -, •, or numbered)
        issue_pattern = r'(?:^|\n)[\s]*(?:-|\•|\d+\.)\s*(.+?)(?=\n|$)'
        issues = re.findall(issue_pattern, response)
        
        # Categorize as issues or suggestions
        for item in issues:
            item = item.strip()
            if any(word in item.lower() for word in ['issue', 'error', 'bug', 'problem', 'vulnerability']):
                result["issues"].append(item)
            elif any(word in item.lower() for word in ['suggest', 'recommend', 'improve', 'consider', 'could']):
                result["suggestions"].append(item)
        
        return result
    
    def _parse_quality_scores(self, response: str) -> Dict[str, int]:
        """
        Parse quality metric scores from response.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with metric scores
        """
        scores = {}
        
        # Extract metric scores (e.g., "Readability: 85/100")
        metric_pattern = r'(\w+)[\s:]+(\d+)(?:/100)?'
        matches = re.findall(metric_pattern, response)
        
        total_score = 0
        count = 0
        
        for metric, score in matches:
            metric_lower = metric.lower()
            score_int = int(score)
            
            # Convert to 0-100 scale if needed
            if score_int > 100:
                score_int = 100
            
            scores[metric_lower] = score_int
            total_score += score_int
            count += 1
        
        # Calculate overall score
        if count > 0:
            scores["overall"] = total_score // count
        
        return scores
    
    def _parse_security_issues(self, response: str) -> Dict[str, Any]:
        """
        Parse security issues from response.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with security findings
        """
        result = {
            "vulnerabilities": [],
            "risk_level": "low",
            "recommendations": []
        }
        
        # Extract risk level
        if any(word in response.lower() for word in ['critical', 'high risk', 'severe']):
            result["risk_level"] = "high"
        elif any(word in response.lower() for word in ['medium', 'moderate']):
            result["risk_level"] = "medium"
        
        # Extract vulnerabilities
        vuln_keywords = ['vulnerability', 'security issue', 'exploit', 'injection', 'xss', 'sql injection']
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in vuln_keywords):
                result["vulnerabilities"].append(line.strip())
            elif any(word in line_lower for word in ['recommend', 'should', 'use', 'implement']):
                result["recommendations"].append(line.strip())
        
        return result
    
    def _parse_performance_issues(self, response: str) -> Dict[str, Any]:
        """
        Parse performance analysis from response.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with performance findings
        """
        result = {
            "bottlenecks": [],
            "optimizations": [],
            "complexity": ""
        }
        
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            if any(word in line_lower for word in ['bottleneck', 'slow', 'inefficient', 'performance issue']):
                result["bottlenecks"].append(line.strip())
            elif any(word in line_lower for word in ['optimize', 'improve', 'faster', 'cache', 'vectorize']):
                result["optimizations"].append(line.strip())
            elif 'complexity' in line_lower and ('O(' in line or 'o(' in line):
                result["complexity"] = line.strip()
        
        return result
    
    def _parse_best_practices(self, response: str) -> Dict[str, Any]:
        """
        Parse best practice violations from response.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with violations and recommendations
        """
        result = {
            "violations": [],
            "recommendations": [],
            "style_score": 0
        }
        
        # Extract style score
        score_match = re.search(r'(?:style|pep\s*8|best\s*practice)[:\s]+(\d+)', response, re.IGNORECASE)
        if score_match:
            result["style_score"] = int(score_match.group(1))
        
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            if any(word in line_lower for word in ['violation', 'not follow', 'should not', 'avoid']):
                result["violations"].append(line.strip())
            elif any(word in line_lower for word in ['recommend', 'should', 'consider', 'use']):
                result["recommendations"].append(line.strip())
        
        return result
    
    def _parse_review_report(self, response: str) -> Dict[str, Any]:
        """
        Parse comprehensive review report.
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with report components
        """
        result = {
            "summary": "",
            "overall_score": 0,
            "categories": {},
            "critical_issues": [],
            "suggestions": []
        }
        
        # Extract overall score
        score_match = re.search(r'(?:overall|total)[:\s]+(\d+)', response, re.IGNORECASE)
        if score_match:
            result["overall_score"] = int(score_match.group(1))
        
        # Extract summary (first paragraph)
        paragraphs = response.split('\n\n')
        if paragraphs:
            result["summary"] = paragraphs[0].strip()
        
        # Extract critical issues
        if 'critical' in response.lower():
            lines = response.split('\n')
            in_critical_section = False
            
            for line in lines:
                if 'critical' in line.lower():
                    in_critical_section = True
                elif in_critical_section and line.strip():
                    if line.strip().startswith(('-', '•', '*')) or line.strip()[0].isdigit():
                        result["critical_issues"].append(line.strip())
        
        return result
