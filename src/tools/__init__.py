"""
Tools Package

Utility tools for code analysis and automation.

Modules:
- cross_file_impact_analyzer: Analyze cross-file dependencies and change impact
- task_runner: Execute shell commands, Python scripts, and tests
- problem_checker: Validate code and detect issues
- search_refactor_tool: Code search and refactoring operations
- test_failure_handler: Analyze test failures and suggest fixes
- todo_manager: Find and manage TODO comments in code

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from .cross_file_impact_analyzer import (
    CrossFileImpactAnalyzer,
    ImpactAnalysis,
    FileImpact,
    ImpactLevel
)
from .task_runner import (
    TaskRunner,
    CommandResult,
    BackgroundTask,
    ProcessStatus,
    OutputMode
)
from .problem_checker import (
    ProblemChecker,
    Problem,
    ValidationResult,
    ProblemSeverity,
    ProblemCategory
)
from .search_refactor_tool import (
    SearchRefactorTool,
    Symbol,
    Reference,
    SearchResult,
    RefactoringResult,
    SymbolType,
    RefactoringType
)
from .test_failure_handler import (
    TestFailureHandler,
    TestFailure,
    TestResult,
    FailureType,
    FailureSeverity
)
from .todo_manager import (
    TodoManager,
    TodoItem,
    TodoReport,
    TodoPriority,
    TodoStatus
)

__all__ = [
    'CrossFileImpactAnalyzer',
    'ImpactAnalysis',
    'FileImpact',
    'ImpactLevel',
    'TaskRunner',
    'CommandResult',
    'BackgroundTask',
    'ProcessStatus',
    'OutputMode',
    'ProblemChecker',
    'Problem',
    'ValidationResult',
    'ProblemSeverity',
    'ProblemCategory',
    'SearchRefactorTool',
    'Symbol',
    'Reference',
    'SearchResult',
    'RefactoringResult',
    'SymbolType',
    'RefactoringType',
    'TestFailureHandler',
    'TestFailure',
    'TestResult',
    'FailureType',
    'FailureSeverity',
    'TodoManager',
    'TodoItem',
    'TodoReport',
    'TodoPriority',
    'TodoStatus',
]
