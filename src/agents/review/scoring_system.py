"""
Scoring System for Code Review

Aggregates scores from multiple specialized reviewers into a unified quality assessment.
Provides weighted scoring, quality dimensions, and comprehensive reports.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import logger


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    FAIR = "fair"           # 60-74
    POOR = "poor"           # 40-59
    CRITICAL = "critical"   # 0-39


@dataclass
class ReviewScore:
    """Individual reviewer score"""
    reviewer: str
    score: float
    breakdown: Dict[str, float]
    issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "reviewer": self.reviewer,
            "score": round(self.score, 1),
            "breakdown": {k: round(v, 1) for k, v in self.breakdown.items()},
            "issues": self.issues,
            "recommendations": self.recommendations
        }


@dataclass
class AggregatedScore:
    """Aggregated score from all reviewers"""
    overall_score: float
    quality_level: QualityLevel
    dimension_scores: Dict[str, float]
    reviewer_scores: List[ReviewScore]
    total_issues: int
    critical_issues: List[str]
    recommendations: List[str]
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": round(self.overall_score, 1),
            "quality_level": self.quality_level.value,
            "dimension_scores": {k: round(v, 1) for k, v in self.dimension_scores.items()},
            "reviewer_scores": [rs.to_dict() for rs in self.reviewer_scores],
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "statistics": self.statistics
        }


class ScoringSystem:
    """
    Aggregates and analyzes scores from multiple code reviewers.
    
    Reviewers (Phase 2.5 - 6 Reviewers):
    - ReadabilityReviewer: Code style, naming, complexity, comments
    - LogicFlowReviewer: Control flow, error handling, edge cases
    - CodeConnectivityReviewer: Function dependencies, data flow, API
    - ProjectConnectivityReviewer: Import graph, module coupling, architecture
    - PerformanceReviewer: Time/space complexity, optimization opportunities
    - SecurityReviewer: Vulnerabilities, security best practices
    
    Scoring:
    - Each reviewer provides 0-100 score with breakdown
    - Weighted aggregation based on reviewer importance
    - Quality level classification (Excellent, Good, Fair, Poor, Critical)
    - Dimension-specific scores (readability, logic, connectivity, performance, security)
    """
    
    # Default weights for each reviewer (must sum to 1.0)
    # Phase 2.5: Updated to support 6 reviewers
    DEFAULT_WEIGHTS = {
        "readability": 0.15,
        "logic_flow": 0.20,
        "code_connectivity": 0.15,
        "project_connectivity": 0.15,
        "performance": 0.20,
        "security": 0.15
    }
    
    # Quality thresholds
    THRESHOLDS = {
        QualityLevel.EXCELLENT: 90,
        QualityLevel.GOOD: 75,
        QualityLevel.FAIR: 60,
        QualityLevel.POOR: 40,
        QualityLevel.CRITICAL: 0
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize scoring system
        
        Args:
            weights: Custom weights for reviewers (optional)
                    If provided, must sum to 1.0
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Validate weights
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights.values())}")
        
        logger.info("ScoringSystem initialized with weights: {}", self.weights)
    
    def aggregate_scores(
        self,
        readability_score: Optional[Dict[str, Any]] = None,
        logic_flow_score: Optional[Dict[str, Any]] = None,
        code_connectivity_score: Optional[Dict[str, Any]] = None,
        project_connectivity_score: Optional[Dict[str, Any]] = None,
        performance_score: Optional[Dict[str, Any]] = None,
        security_score: Optional[Dict[str, Any]] = None
    ) -> AggregatedScore:
        """
        Aggregate scores from multiple reviewers
        
        Args:
            readability_score: Score from ReadabilityReviewer
            logic_flow_score: Score from LogicFlowReviewer
            code_connectivity_score: Score from CodeConnectivityReviewer
            project_connectivity_score: Score from ProjectConnectivityReviewer
            performance_score: Score from PerformanceReviewer (Phase 2.5)
            security_score: Score from SecurityReviewer (Phase 2.5)
            
        Returns:
            AggregatedScore with overall quality assessment
        """
        # Collect reviewer scores
        reviewer_scores = []
        dimension_scores = {}
        all_issues = []
        
        # Process readability
        if readability_score:
            rs = ReviewScore(
                reviewer="readability",
                score=readability_score.get("readability_score", 0),
                breakdown=readability_score.get("breakdown", {}),
                issues=readability_score.get("issues", []),
                recommendations=readability_score.get("recommendations", [])
            )
            reviewer_scores.append(rs)
            dimension_scores["readability"] = rs.score
            all_issues.extend(rs.issues)
        
        # Process logic flow
        if logic_flow_score:
            rs = ReviewScore(
                reviewer="logic_flow",
                score=logic_flow_score.get("logic_score", 0),
                breakdown=logic_flow_score.get("breakdown", {}),
                issues=logic_flow_score.get("issues", []),
                recommendations=logic_flow_score.get("recommendations", [])
            )
            reviewer_scores.append(rs)
            dimension_scores["logic_flow"] = rs.score
            all_issues.extend(rs.issues)
        
        # Process code connectivity
        if code_connectivity_score:
            rs = ReviewScore(
                reviewer="code_connectivity",
                score=code_connectivity_score.get("connectivity_score", 0),
                breakdown=code_connectivity_score.get("breakdown", {}),
                issues=code_connectivity_score.get("issues", []),
                recommendations=code_connectivity_score.get("recommendations", [])
            )
            reviewer_scores.append(rs)
            dimension_scores["code_connectivity"] = rs.score
            all_issues.extend(rs.issues)
        
        # Process project connectivity
        if project_connectivity_score:
            rs = ReviewScore(
                reviewer="project_connectivity",
                score=project_connectivity_score.get("project_score", 0),
                breakdown=project_connectivity_score.get("breakdown", {}),
                issues=project_connectivity_score.get("issues", []),
                recommendations=project_connectivity_score.get("recommendations", [])
            )
            reviewer_scores.append(rs)
            dimension_scores["project_connectivity"] = rs.score
            all_issues.extend(rs.issues)
        
        # Process performance (Phase 2.5)
        if performance_score:
            rs = ReviewScore(
                reviewer="performance",
                score=performance_score.get("performance_score", performance_score.get("score", 0)),
                breakdown=performance_score.get("breakdown", {}),
                issues=performance_score.get("issues", []),
                recommendations=performance_score.get("recommendations", performance_score.get("suggestions", []))
            )
            reviewer_scores.append(rs)
            dimension_scores["performance"] = rs.score
            all_issues.extend(rs.issues)
        
        # Process security (Phase 2.5)
        if security_score:
            rs = ReviewScore(
                reviewer="security",
                score=security_score.get("security_score", security_score.get("score", 0)),
                breakdown=security_score.get("breakdown", {}),
                issues=security_score.get("issues", []),
                recommendations=security_score.get("recommendations", security_score.get("suggestions", []))
            )
            reviewer_scores.append(rs)
            dimension_scores["security"] = rs.score
            all_issues.extend(rs.issues)
        
        # Calculate weighted overall score
        # Normalize weights based on available reviewers
        available_weights = {
            reviewer: self.weights[reviewer]
            for reviewer in dimension_scores.keys()
        }
        
        if available_weights:
            total_weight = sum(available_weights.values())
            normalized_weights = {
                reviewer: weight / total_weight
                for reviewer, weight in available_weights.items()
            }
            
            overall_score = sum(
                dimension_scores[reviewer] * normalized_weights[reviewer]
                for reviewer in dimension_scores.keys()
            )
        else:
            overall_score = 0.0
        
        # Determine quality level
        quality_level = self._get_quality_level(overall_score)
        
        # Extract critical issues
        critical_issues = self._extract_critical_issues(all_issues, overall_score)
        
        # Generate aggregated recommendations
        recommendations = self._generate_recommendations(
            reviewer_scores,
            overall_score,
            quality_level
        )
        
        # Calculate statistics
        statistics = self._calculate_statistics(
            reviewer_scores,
            dimension_scores,
            all_issues
        )
        
        return AggregatedScore(
            overall_score=overall_score,
            quality_level=quality_level,
            dimension_scores=dimension_scores,
            reviewer_scores=reviewer_scores,
            total_issues=len(all_issues),
            critical_issues=critical_issues,
            recommendations=recommendations,
            statistics=statistics
        )
    
    def _get_quality_level(self, score: float) -> QualityLevel:
        """
        Determine quality level from score
        
        Args:
            score: Overall score (0-100)
            
        Returns:
            QualityLevel enum
        """
        if score >= self.THRESHOLDS[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif score >= self.THRESHOLDS[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif score >= self.THRESHOLDS[QualityLevel.FAIR]:
            return QualityLevel.FAIR
        elif score >= self.THRESHOLDS[QualityLevel.POOR]:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _extract_critical_issues(
        self,
        all_issues: List[str],
        overall_score: float
    ) -> List[str]:
        """
        Extract the most critical issues
        
        Args:
            all_issues: All issues from all reviewers
            overall_score: Overall quality score
            
        Returns:
            List of critical issues (top 5)
        """
        # Keywords that indicate critical issues
        critical_keywords = [
            "circular", "security", "critical", "severe",
            "unsafe", "vulnerable", "deadlock", "infinite"
        ]
        
        critical = []
        for issue in all_issues:
            # Handle both string and dict issue formats
            issue_text = issue if isinstance(issue, str) else issue.get("message", str(issue))
            if any(keyword in issue_text.lower() for keyword in critical_keywords):
                critical.append(issue)
        
        # If score is very low, also include first few general issues
        if overall_score < 60 and len(critical) < 5:
            remaining = 5 - len(critical)
            for issue in all_issues:
                if issue not in critical and len(critical) < 5:
                    critical.append(issue)
                    remaining -= 1
                    if remaining == 0:
                        break
        
        return critical[:5]
    
    def _generate_recommendations(
        self,
        reviewer_scores: List[ReviewScore],
        overall_score: float,
        quality_level: QualityLevel
    ) -> List[str]:
        """
        Generate prioritized recommendations
        
        Args:
            reviewer_scores: Scores from all reviewers
            overall_score: Overall quality score
            quality_level: Quality classification
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Overall assessment
        if quality_level == QualityLevel.EXCELLENT:
            recommendations.append("🌟 EXCELLENT: Code quality is outstanding!")
        elif quality_level == QualityLevel.GOOD:
            recommendations.append("✅ GOOD: Code quality is solid with minor improvements possible")
        elif quality_level == QualityLevel.FAIR:
            recommendations.append("⚠️ FAIR: Code quality needs improvement in several areas")
        elif quality_level == QualityLevel.POOR:
            recommendations.append("❌ POOR: Significant quality issues require attention")
        else:
            recommendations.append("🚨 CRITICAL: Urgent refactoring needed - code quality is very low")
        
        # Find weakest dimensions
        weak_dimensions = []
        for rs in reviewer_scores:
            if rs.score < 70:
                weak_dimensions.append((rs.reviewer, rs.score))
        
        # Sort by score (lowest first)
        weak_dimensions.sort(key=lambda x: x[1])
        
        # Add dimension-specific recommendations
        dimension_names = {
            "readability": "Improve code readability",
            "logic_flow": "Improve logic flow and error handling",
            "code_connectivity": "Improve code structure and dependencies",
            "project_connectivity": "Improve project architecture",
            "performance": "Improve code performance and efficiency",
            "security": "Address security vulnerabilities"
        }
        
        for reviewer, score in weak_dimensions[:3]:
            if reviewer in dimension_names:
                recommendations.append(
                    f"{len(recommendations)}. {dimension_names[reviewer]} (score: {score:.1f})"
                )
        
        # Add specific recommendations from lowest-scoring reviewer
        if weak_dimensions:
            lowest_reviewer = weak_dimensions[0][0]
            for rs in reviewer_scores:
                if rs.reviewer == lowest_reviewer:
                    # Add first 2 recommendations from this reviewer
                    for rec in rs.recommendations[1:3]:  # Skip first (usually overall)
                        if rec and not rec.startswith(("🟢", "🟡", "🟠", "🔴", "🌟", "✅", "⚠️", "❌", "🚨")):
                            recommendations.append(f"   • {rec}")
        
        return recommendations[:7]  # Top 7 recommendations
    
    def _calculate_statistics(
        self,
        reviewer_scores: List[ReviewScore],
        dimension_scores: Dict[str, float],
        all_issues: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the review
        
        Args:
            reviewer_scores: Scores from all reviewers
            dimension_scores: Dimension-specific scores
            all_issues: All issues found
            
        Returns:
            Dictionary of statistics
        """
        return {
            "reviewers_run": len(reviewer_scores),
            "avg_score": sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0,
            "highest_dimension": max(dimension_scores.items(), key=lambda x: x[1])[0] if dimension_scores else None,
            "lowest_dimension": min(dimension_scores.items(), key=lambda x: x[1])[0] if dimension_scores else None,
            "score_variance": self._calculate_variance(list(dimension_scores.values())),
            "total_issues": len(all_issues),
            "issues_by_reviewer": {
                rs.reviewer: len(rs.issues) for rs in reviewer_scores
            }
        }
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores"""
        if not scores:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance
    
    def get_quality_report(self, aggregated_score: AggregatedScore) -> str:
        """
        Generate a human-readable quality report
        
        Args:
            aggregated_score: Aggregated score result
            
        Returns:
            Formatted quality report
        """
        report = []
        
        # Header
        report.append("=" * 70)
        report.append("CODE QUALITY REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall score
        report.append(f"Overall Score: {aggregated_score.overall_score:.1f}/100")
        report.append(f"Quality Level: {aggregated_score.quality_level.value.upper()}")
        report.append("")
        
        # Dimension scores
        report.append("Dimension Scores:")
        report.append("-" * 70)
        for dimension, score in aggregated_score.dimension_scores.items():
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            report.append(f"  {dimension:25s} {score:5.1f}/100  [{bar}]")
        report.append("")
        
        # Critical issues
        if aggregated_score.critical_issues:
            report.append("Critical Issues:")
            report.append("-" * 70)
            for i, issue in enumerate(aggregated_score.critical_issues, 1):
                report.append(f"  {i}. {issue}")
            report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 70)
        for rec in aggregated_score.recommendations:
            report.append(f"  {rec}")
        report.append("")
        
        # Statistics
        report.append("Statistics:")
        report.append("-" * 70)
        stats = aggregated_score.statistics
        report.append(f"  Reviewers Run: {stats['reviewers_run']}")
        report.append(f"  Total Issues: {stats['total_issues']}")
        if stats['highest_dimension']:
            report.append(f"  Highest Dimension: {stats['highest_dimension']} "
                         f"({aggregated_score.dimension_scores[stats['highest_dimension']]:.1f})")
        if stats['lowest_dimension']:
            report.append(f"  Lowest Dimension: {stats['lowest_dimension']} "
                         f"({aggregated_score.dimension_scores[stats['lowest_dimension']]:.1f})")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def is_acceptable_quality(
        self,
        aggregated_score: AggregatedScore,
        threshold: float = 75.0
    ) -> bool:
        """
        Check if code meets minimum quality threshold
        
        Args:
            aggregated_score: Aggregated score result
            threshold: Minimum acceptable score (default: 75.0)
            
        Returns:
            True if quality is acceptable
        """
        return aggregated_score.overall_score >= threshold
    
    def get_improvement_priority(
        self,
        aggregated_score: AggregatedScore
    ) -> List[Dict[str, Any]]:
        """
        Get prioritized list of improvement areas
        
        Args:
            aggregated_score: Aggregated score result
            
        Returns:
            List of improvement areas sorted by priority
        """
        improvements = []
        
        for dimension, score in aggregated_score.dimension_scores.items():
            # Calculate improvement potential (how much below 100)
            potential = 100 - score
            
            # Calculate priority (lower scores = higher priority)
            priority = "high" if score < 60 else "medium" if score < 80 else "low"
            
            improvements.append({
                "dimension": dimension,
                "current_score": score,
                "improvement_potential": potential,
                "priority": priority
            })
        
        # Sort by score (lowest first)
        improvements.sort(key=lambda x: x["current_score"])
        
        return improvements
    
    def compare_scores(
        self,
        before: AggregatedScore,
        after: AggregatedScore
    ) -> Dict[str, Any]:
        """
        Compare two aggregated scores to show improvement/regression
        
        Args:
            before: Score before changes
            after: Score after changes
            
        Returns:
            Comparison results
        """
        delta = after.overall_score - before.overall_score
        
        dimension_changes = {}
        for dimension in after.dimension_scores.keys():
            if dimension in before.dimension_scores:
                dimension_changes[dimension] = {
                    "before": before.dimension_scores[dimension],
                    "after": after.dimension_scores[dimension],
                    "delta": after.dimension_scores[dimension] - before.dimension_scores[dimension]
                }
        
        return {
            "overall_delta": delta,
            "improved": delta > 0,
            "quality_level_before": before.quality_level.value,
            "quality_level_after": after.quality_level.value,
            "dimension_changes": dimension_changes,
            "issues_delta": after.total_issues - before.total_issues
        }


# Module-level logger already imported
