"""
Test Failure Handler - Analyze test failures and suggest fixes.

Provides comprehensive test failure analysis:
- Parse pytest output
- Extract failure information (assertions, exceptions, tracebacks)
- Categorize failure types
- Generate fix suggestions
- Track failure patterns
- Integration with test runners

Author: AutoCoder System
gmail:ganeshnaik214@gmail.com
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger


class FailureType(Enum):
    """Types of test failures."""
    ASSERTION_ERROR = "assertion_error"       # Assertion failed
    EXCEPTION = "exception"                   # Unexpected exception raised
    TIMEOUT = "timeout"                       # Test timed out
    IMPORT_ERROR = "import_error"            # Import failed
    SYNTAX_ERROR = "syntax_error"            # Syntax error in test
    ATTRIBUTE_ERROR = "attribute_error"      # Attribute not found
    TYPE_ERROR = "type_error"                # Type mismatch
    VALUE_ERROR = "value_error"              # Invalid value
    KEY_ERROR = "key_error"                  # Key not found
    INDEX_ERROR = "index_error"              # Index out of range
    NAME_ERROR = "name_error"                # Name not defined
    FILE_NOT_FOUND = "file_not_found"        # File not found
    SETUP_ERROR = "setup_error"              # Test setup failed
    TEARDOWN_ERROR = "teardown_error"        # Test teardown failed
    FIXTURE_ERROR = "fixture_error"          # Fixture failed
    UNKNOWN = "unknown"                      # Unknown failure type


class FailureSeverity(Enum):
    """Severity levels for failures."""
    CRITICAL = "critical"     # Blocks all tests
    HIGH = "high"            # Blocks test suite/module
    MEDIUM = "medium"        # Single test failure
    LOW = "low"              # Warning or expected failure


@dataclass
class TestFailure:
    """Represents a test failure."""
    test_name: str
    failure_type: FailureType
    severity: FailureSeverity
    error_message: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    traceback: Optional[str] = None
    assertion_details: Optional[Dict[str, Any]] = None
    suggested_fixes: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation."""
        location = ""
        if self.file_path:
            location = f" in {self.file_path}"
            if self.line_number:
                location += f":{self.line_number}"
        
        return f"{self.severity.value.upper()}: {self.test_name}{location} - {self.failure_type.value}: {self.error_message}"


@dataclass
class TestResult:
    """Result of test execution."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    failures: List[TestFailure] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def has_failures(self) -> bool:
        """Check if there are any failures."""
        return self.failed > 0 or self.errors > 0
    
    def get_failures_by_type(self, failure_type: FailureType) -> List[TestFailure]:
        """Get failures filtered by type."""
        return [f for f in self.failures if f.failure_type == failure_type]
    
    def get_failures_by_severity(self, severity: FailureSeverity) -> List[TestFailure]:
        """Get failures filtered by severity."""
        return [f for f in self.failures if f.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TestResult object to a dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Tests: {self.total_tests}, Passed: {self.passed}, Failed: {self.failed}, Skipped: {self.skipped}, Duration: {self.duration:.2f}s"


class TestFailureHandler:
    """
    Analyze test failures and suggest fixes.
    
    Features:
    - Parse pytest output
    - Extract failure information
    - Categorize failure types
    - Generate fix suggestions
    - Track failure patterns
    """
    
    def __init__(self):
        """Initialize test failure handler."""
        self.failure_patterns = self._build_failure_patterns()
        logger.info("TestFailureHandler initialized")
    
    def parse_pytest_output(self, output: str) -> TestResult:
        """
        Parse pytest output and extract test results.
        
        Args:
            output: Raw pytest output text
        
        Returns:
            TestResult with parsed information
        """
        result = TestResult()
        
        # Extract summary line (e.g., "5 passed, 2 failed in 1.23s")
        summary_match = re.search(
            r'=+\s*(\d+)\s+failed.*?(\d+)\s+passed.*?in\s+([\d.]+)s',
            output,
            re.IGNORECASE
        )
        if summary_match:
            result.failed = int(summary_match.group(1))
            result.passed = int(summary_match.group(2))
            result.duration = float(summary_match.group(3))
            result.total_tests = result.passed + result.failed
        else:
            # Try alternative format
            summary_match = re.search(
                r'=+\s*(\d+)\s+passed.*?in\s+([\d.]+)s',
                output,
                re.IGNORECASE
            )
            if summary_match:
                result.passed = int(summary_match.group(1))
                result.duration = float(summary_match.group(2))
                result.total_tests = result.passed
        
        # Extract skipped
        skipped_match = re.search(r'(\d+)\s+skipped', output, re.IGNORECASE)
        if skipped_match:
            result.skipped = int(skipped_match.group(1))
        
        # Extract errors
        error_match = re.search(r'(\d+)\s+error', output, re.IGNORECASE)
        if error_match:
            result.errors = int(error_match.group(1))
        
        # Parse individual failures
        failures = self._extract_failures(output)
        result.failures.extend(failures)
        
        logger.info(f"Parsed test results: {result}")
        return result
    
    def _extract_failures(self, output: str) -> List[TestFailure]:
        """Extract individual test failures from output."""
        failures = []
        
        # Find FAILED sections
        failed_pattern = r'FAILED\s+([\w/.:-]+)::([\w_]+)\s*-\s*(.*?)(?=\n(?:FAILED|=|$))'
        for match in re.finditer(failed_pattern, output, re.DOTALL | re.MULTILINE):
            file_test = match.group(1)
            test_name = match.group(2)
            error_text = match.group(3).strip()
            
            # Extract file path
            file_match = re.match(r'([\w/\\.-]+\.py)', file_test)
            file_path = Path(file_match.group(1)) if file_match else None
            
            # Determine failure type and extract details
            failure_type, error_message, line_number = self._categorize_failure(error_text)
            
            # Determine severity
            severity = self._determine_severity(failure_type, error_text)
            
            # Extract traceback
            traceback = self._extract_traceback(error_text)
            
            # Generate fix suggestions
            suggestions = self._generate_fix_suggestions(failure_type, error_message, traceback)
            
            failure = TestFailure(
                test_name=f"{file_test}::{test_name}",
                failure_type=failure_type,
                severity=severity,
                error_message=error_message,
                file_path=file_path,
                line_number=line_number,
                traceback=traceback,
                suggested_fixes=suggestions
            )
            failures.append(failure)
        
        return failures
    
    def _categorize_failure(self, error_text: str) -> Tuple[FailureType, str, Optional[int]]:
        """
        Categorize failure type from error text.
        
        Returns:
            Tuple of (failure_type, error_message, line_number)
        """
        # Extract line number if present
        line_match = re.search(r':(\d+):', error_text)
        line_number = int(line_match.group(1)) if line_match else None
        
        # Check each failure pattern
        for failure_type, patterns in self.failure_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, error_text, re.IGNORECASE)
                if match:
                    # Extract error message
                    error_message = match.group(0) if match.groups() == () else match.group(1)
                    return failure_type, error_message.strip(), line_number
        
        # Default to unknown
        # Try to extract first line as error message
        first_line = error_text.split('\n')[0].strip()
        return FailureType.UNKNOWN, first_line, line_number
    
    def _determine_severity(self, failure_type: FailureType, error_text: str) -> FailureSeverity:
        """Determine severity based on failure type and context."""
        # Critical failures
        if failure_type in [FailureType.SYNTAX_ERROR, FailureType.IMPORT_ERROR]:
            return FailureSeverity.CRITICAL
        
        # High severity
        if failure_type in [FailureType.FIXTURE_ERROR, FailureType.SETUP_ERROR]:
            return FailureSeverity.HIGH
        
        # Medium severity (most common)
        if failure_type in [
            FailureType.ASSERTION_ERROR,
            FailureType.EXCEPTION,
            FailureType.TYPE_ERROR,
            FailureType.VALUE_ERROR
        ]:
            return FailureSeverity.MEDIUM
        
        # Low severity
        return FailureSeverity.LOW
    
    def _extract_traceback(self, error_text: str) -> Optional[str]:
        """Extract traceback from error text."""
        # Look for traceback section
        traceback_match = re.search(
            r'(?:Traceback.*?|File ".*?")(.+?)(?=\n(?:[A-Z]|$))',
            error_text,
            re.DOTALL
        )
        if traceback_match:
            return traceback_match.group(0).strip()
        return None
    
    def _generate_fix_suggestions(
        self,
        failure_type: FailureType,
        error_message: str,
        traceback: Optional[str]
    ) -> List[str]:
        """Generate fix suggestions based on failure type."""
        suggestions = []
        
        if failure_type == FailureType.ASSERTION_ERROR:
            suggestions.append("Check the assertion condition - the expected and actual values don't match")
            suggestions.append("Verify test data setup is correct")
            suggestions.append("Review the logic in the code being tested")
        
        elif failure_type == FailureType.IMPORT_ERROR:
            suggestions.append("Verify the module/package is installed")
            suggestions.append("Check the import path is correct")
            suggestions.append("Ensure dependencies are in requirements.txt")
            if "No module named" in error_message:
                module_name = re.search(r"No module named '([\w.]+)'", error_message)
                if module_name:
                    suggestions.append(f"Install missing module: pip install {module_name.group(1)}")
        
        elif failure_type == FailureType.ATTRIBUTE_ERROR:
            suggestions.append("Check if the object has the expected attribute")
            suggestions.append("Verify the object type is correct")
            suggestions.append("Review recent API changes")
        
        elif failure_type == FailureType.TYPE_ERROR:
            suggestions.append("Check argument types match function signature")
            suggestions.append("Verify type conversions are correct")
            suggestions.append("Review function call arguments")
        
        elif failure_type == FailureType.VALUE_ERROR:
            suggestions.append("Validate input values before processing")
            suggestions.append("Check value ranges and constraints")
            suggestions.append("Add input validation")
        
        elif failure_type == FailureType.KEY_ERROR:
            suggestions.append("Check if the key exists in the dictionary")
            suggestions.append("Use .get() method with default value")
            suggestions.append("Add key existence check before access")
        
        elif failure_type == FailureType.INDEX_ERROR:
            suggestions.append("Check list/array length before indexing")
            suggestions.append("Verify index calculation is correct")
            suggestions.append("Add bounds checking")
        
        elif failure_type == FailureType.NAME_ERROR:
            suggestions.append("Check variable/function name spelling")
            suggestions.append("Verify the variable is defined before use")
            suggestions.append("Check scope - variable may not be in current scope")
        
        elif failure_type == FailureType.FILE_NOT_FOUND:
            suggestions.append("Verify the file path is correct")
            suggestions.append("Check if the file exists")
            suggestions.append("Use absolute paths or check working directory")
        
        elif failure_type == FailureType.TIMEOUT:
            suggestions.append("Increase timeout value")
            suggestions.append("Optimize the code being tested")
            suggestions.append("Check for infinite loops or blocking operations")
        
        elif failure_type == FailureType.FIXTURE_ERROR:
            suggestions.append("Check fixture definition and scope")
            suggestions.append("Verify fixture dependencies")
            suggestions.append("Review fixture setup code")
        
        elif failure_type == FailureType.SYNTAX_ERROR:
            suggestions.append("Fix syntax error in the code")
            suggestions.append("Check for missing colons, parentheses, or quotes")
            suggestions.append("Verify indentation is correct")
        
        elif failure_type == FailureType.EXCEPTION:
            suggestions.append("Add try-except block to handle exception")
            suggestions.append("Review the code causing the exception")
            suggestions.append("Check input validation")
        
        return suggestions
    
    def _build_failure_patterns(self) -> Dict[FailureType, List[str]]:
        """Build regex patterns for failure type detection."""
        return {
            FailureType.ASSERTION_ERROR: [
                r'AssertionError',
                r'assert .+ == .+',
                r'assert .+ is .+',
                r'assert .+ in .+',
            ],
            FailureType.IMPORT_ERROR: [
                r'ImportError',
                r'ModuleNotFoundError',
                r"No module named ['\"](.+?)['\"]",
            ],
            FailureType.SYNTAX_ERROR: [
                r'SyntaxError',
                r'invalid syntax',
            ],
            FailureType.ATTRIBUTE_ERROR: [
                r'AttributeError',
                r"'(.+?)' object has no attribute '(.+?)'",
            ],
            FailureType.TYPE_ERROR: [
                r'TypeError',
                r'takes \d+ positional argument',
                r'expected .+ got .+',
            ],
            FailureType.VALUE_ERROR: [
                r'ValueError',
                r'invalid literal',
            ],
            FailureType.KEY_ERROR: [
                r'KeyError',
            ],
            FailureType.INDEX_ERROR: [
                r'IndexError',
                r'list index out of range',
            ],
            FailureType.NAME_ERROR: [
                r'NameError',
                r"name '(.+?)' is not defined",
            ],
            FailureType.FILE_NOT_FOUND: [
                r'FileNotFoundError',
                r'No such file or directory',
            ],
            FailureType.TIMEOUT: [
                r'TimeoutError',
                r'timed out',
                r'timeout',
            ],
            FailureType.FIXTURE_ERROR: [
                r'fixture .+ not found',
                r'fixture .+ error',
            ],
            FailureType.SETUP_ERROR: [
                r'error at setup',
                r'setup_method',
            ],
            FailureType.TEARDOWN_ERROR: [
                r'error at teardown',
                r'teardown_method',
            ],
        }
    
    def analyze_failure(self, failure: TestFailure) -> Dict[str, Any]:
        """
        Perform detailed analysis of a failure.
        
        Args:
            failure: TestFailure to analyze
        
        Returns:
            Dictionary with analysis details
        """
        analysis = {
            'test_name': failure.test_name,
            'type': failure.failure_type.value,
            'severity': failure.severity.value,
            'message': failure.error_message,
            'suggestions': failure.suggested_fixes,
            'actionable': len(failure.suggested_fixes) > 0,
            'priority': self._calculate_priority(failure)
        }
        
        if failure.file_path:
            analysis['file'] = str(failure.file_path)
        if failure.line_number:
            analysis['line'] = failure.line_number
        if failure.traceback:
            analysis['traceback'] = failure.traceback
        
        return analysis
    
    def _calculate_priority(self, failure: TestFailure) -> int:
        """
        Calculate fix priority (1-10, 10 = highest).
        
        Based on severity and failure type.
        """
        base_priority = {
            FailureSeverity.CRITICAL: 10,
            FailureSeverity.HIGH: 7,
            FailureSeverity.MEDIUM: 5,
            FailureSeverity.LOW: 2
        }
        
        priority = base_priority[failure.severity]
        
        # Adjust based on failure type
        if failure.failure_type == FailureType.SYNTAX_ERROR:
            priority = 10  # Always highest
        elif failure.failure_type == FailureType.IMPORT_ERROR:
            priority = max(priority, 9)
        
        return priority
    
    def generate_failure_report(self, result: TestResult) -> str:
        """
        Generate a human-readable failure report.
        
        Args:
            result: TestResult to report on
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TEST FAILURE REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total Tests: {result.total_tests}")
        lines.append(f"Passed: {result.passed} ({result.pass_rate:.1f}%)")
        lines.append(f"Failed: {result.failed}")
        lines.append(f"Skipped: {result.skipped}")
        lines.append(f"Errors: {result.errors}")
        lines.append(f"Duration: {result.duration:.2f}s")
        lines.append("")
        
        if result.has_failures:
            lines.append("-" * 80)
            lines.append("FAILURES:")
            lines.append("-" * 80)
            lines.append("")
            
            # Group by severity
            for severity in [FailureSeverity.CRITICAL, FailureSeverity.HIGH, FailureSeverity.MEDIUM, FailureSeverity.LOW]:
                failures = result.get_failures_by_severity(severity)
                if failures:
                    lines.append(f"\n{severity.value.upper()} SEVERITY ({len(failures)} failures):")
                    lines.append("")
                    
                    for i, failure in enumerate(failures, 1):
                        lines.append(f"{i}. {failure.test_name}")
                        lines.append(f"   Type: {failure.failure_type.value}")
                        lines.append(f"   Error: {failure.error_message}")
                        
                        if failure.file_path:
                            location = str(failure.file_path)
                            if failure.line_number:
                                location += f":{failure.line_number}"
                            lines.append(f"   Location: {location}")
                        
                        if failure.suggested_fixes:
                            lines.append("   Suggested Fixes:")
                            for fix in failure.suggested_fixes:
                                lines.append(f"     - {fix}")
                        
                        lines.append("")
        else:
            lines.append("✅ All tests passed!")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
