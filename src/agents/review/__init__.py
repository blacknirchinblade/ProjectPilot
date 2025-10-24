"""
Review Agent Module

Provides code review and quality analysis capabilities for the AutoCoder system.



The new architecture uses 6 specialized reviewers coordinated by ReviewOrchestrator:
- ReadabilityReviewer: Code clarity, naming, structure
- LogicFlowReviewer: Control flow, algorithms, edge cases
- CodeConnectivityReviewer: Function relationships, cohesion
- ProjectConnectivityReviewer: Cross-file dependencies, imports
- PerformanceReviewer: Complexity, optimization, efficiency
- SecurityReviewer: Vulnerabilities, unsafe practices


Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com

"""

from .review_agent import ReviewAgent  # DEPRECATED - Use ReviewOrchestrator
from .review_orchestrator import ReviewOrchestrator
from .readability_reviewer import ReadabilityReviewer
from .logic_flow_reviewer import LogicFlowReviewer
from .code_connectivity_reviewer import CodeConnectivityReviewer
from .project_connectivity_reviewer import ProjectConnectivityReviewer
from .performance_reviewer import PerformanceReviewer
from .security_reviewer import SecurityReviewer

__all__ = [
    'ReviewAgent',  # DEPRECATED - Use ReviewOrchestrator instead
    'ReviewOrchestrator',  # RECOMMENDED: Unified interface for all reviewers
    'ReadabilityReviewer',
    'LogicFlowReviewer',
    'CodeConnectivityReviewer',
    'ProjectConnectivityReviewer',
    'PerformanceReviewer',
    'SecurityReviewer'
]
