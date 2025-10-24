"""
Problem Checker - Validate code and detect issues.

Provides comprehensive code validation:
- Syntax validation (AST parsing)
- Pylint integration (code quality)
- Flake8 integration (style checking)
- Import validation
- Problem severity classification
- Fix suggestions
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ProblemSeverity(Enum):
    """Problem severity levels."""
    ERROR = "error"          # Critical issues that prevent execution
    WARNING = "warning"      # Issues that should be fixed
    INFO = "info"           # Suggestions for improvement
    HINT = "hint"           # Style recommendations


class ProblemCategory(Enum):
    """Problem categories."""
    SYNTAX = "syntax"                    # Syntax errors
    IMPORT = "import"                    # Import issues
    STYLE = "style"                      # Code style issues
    CONVENTION = "convention"            # Naming conventions
    REFACTOR = "refactor"               # Code complexity/structure
    TYPE = "type"                        # Type checking issues
    SECURITY = "security"                # Security vulnerabilities
    PERFORMANCE = "performance"          # Performance issues


@dataclass
class Problem:
    """Represents a code problem/issue."""
    severity: ProblemSeverity
    category: ProblemCategory
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None  # Error code (e.g., E501, W291)
    source: str = "problem_checker"  # Tool that found the issue
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of problem."""
        location = ""
        if self.line is not None:
            location = f"Line {self.line}"
            if self.column is not None:
                location += f", Col {self.column}"
            location += ": "
        
        code_str = f"[{self.code}] " if self.code else ""
        return f"{self.severity.value.upper()} - {location}{code_str}{self.message}"


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    problems: List[Problem] = field(default_factory=list)
    syntax_valid: bool = True
    imports_valid: bool = True
    
    @property
    def error_count(self) -> int:
        """Count of error-level problems."""
        return sum(1 for p in self.problems if p.severity == ProblemSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level problems."""
        return sum(1 for p in self.problems if p.severity == ProblemSeverity.WARNING)
    
    @property
    def total_issues(self) -> int:
        """Total number of problems."""
        return len(self.problems)
    
    def get_problems_by_severity(self, severity: ProblemSeverity) -> List[Problem]:
        """Get problems filtered by severity."""
        return [p for p in self.problems if p.severity == severity]
    
    def get_problems_by_category(self, category: ProblemCategory) -> List[Problem]:
        """Get problems filtered by category."""
        return [p for p in self.problems if p.category == category]


class ProblemChecker:
    """
    Validate Python code and detect issues.
    
    Features:
    - Syntax validation using AST parsing
    - Pylint integration for code quality
    - Flake8 integration for style checking
    - Import validation
    - Problem severity classification
    - Fix suggestions
    """
    
    def __init__(
        self,
        enable_pylint: bool = False,
        enable_flake8: bool = False,
        max_line_length: int = 100,
        ignore_codes: Optional[List[str]] = None
    ):
        """
        Initialize problem checker.
        
        Args:
            enable_pylint: Enable pylint checking
            enable_flake8: Enable flake8 checking
            max_line_length: Maximum line length
            ignore_codes: List of error codes to ignore
        """
        self.enable_pylint = enable_pylint
        self.enable_flake8 = enable_flake8
        self.max_line_length = max_line_length
        self.ignore_codes = set(ignore_codes or [])
        
        # Check if external tools are available
        self.pylint_available = self._check_tool_available("pylint")
        self.flake8_available = self._check_tool_available("flake8")
        
        if enable_pylint and not self.pylint_available:
            logger.warning("Pylint requested but not available. Install with: pip install pylint")
        if enable_flake8 and not self.flake8_available:
            logger.warning("Flake8 requested but not available. Install with: pip install flake8")
        
        logger.info(f"ProblemChecker initialized (pylint: {self.pylint_available}, flake8: {self.flake8_available})")
    
    def check_code(self, code: str, filename: str = "<string>") -> ValidationResult:
        """
        Check code for problems.
        
        Args:
            code: Python code to check
            filename: Filename for error reporting
        
        Returns:
            ValidationResult with detected problems
        """
        problems: List[Problem] = []
        
        # 1. Syntax validation (always enabled)
        syntax_valid, syntax_problems = self._check_syntax(code, filename)
        problems.extend(syntax_problems)
        
        # If syntax is invalid, stop here (other checks won't work)
        if not syntax_valid:
            return ValidationResult(
                is_valid=False,
                problems=problems,
                syntax_valid=False
            )
        
        # 2. Import validation
        imports_valid, import_problems = self._check_imports(code)
        problems.extend(import_problems)
        
        # 3. Basic style checks (built-in)
        style_problems = self._check_basic_style(code)
        problems.extend(style_problems)
        
        # 4. Pylint (if enabled and available)
        if self.enable_pylint and self.pylint_available:
            pylint_problems = self._run_pylint(code, filename)
            problems.extend(pylint_problems)
        
        # 5. Flake8 (if enabled and available)
        if self.enable_flake8 and self.flake8_available:
            flake8_problems = self._run_flake8(code, filename)
            problems.extend(flake8_problems)
        
        # Filter ignored codes
        problems = [p for p in problems if p.code not in self.ignore_codes]
        
        # Determine if code is valid (no errors)
        is_valid = all(p.severity != ProblemSeverity.ERROR for p in problems)
        
        result = ValidationResult(
            is_valid=is_valid,
            problems=problems,
            syntax_valid=syntax_valid,
            imports_valid=imports_valid
        )
        
        logger.info(f"Code validation complete: {result.error_count} errors, {result.warning_count} warnings")
        return result
    
    def check_file(self, file_path: Path) -> ValidationResult:
        """
        Check a Python file for problems.
        
        Args:
            file_path: Path to Python file
        
        Returns:
            ValidationResult with detected problems
        """
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                problems=[Problem(
                    severity=ProblemSeverity.ERROR,
                    category=ProblemCategory.SYNTAX,
                    message=f"File not found: {file_path}"
                )]
            )
        
        try:
            code = file_path.read_text(encoding='utf-8')
            return self.check_code(code, str(file_path))
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                problems=[Problem(
                    severity=ProblemSeverity.ERROR,
                    category=ProblemCategory.SYNTAX,
                    message=f"Error reading file: {str(e)}"
                )]
            )
    
    def _check_syntax(self, code: str, filename: str) -> Tuple[bool, List[Problem]]:
        """
        Check Python syntax using AST parsing.
        
        Args:
            code: Python code to check
            filename: Filename for error reporting
        
        Returns:
            Tuple of (is_valid, problems)
        """
        problems = []
        
        try:
            ast.parse(code, filename=filename)
            return True, problems
        except SyntaxError as e:
            problem = Problem(
                severity=ProblemSeverity.ERROR,
                category=ProblemCategory.SYNTAX,
                message=e.msg or "Syntax error",
                line=e.lineno,
                column=e.offset,
                code="E0001",
                source="ast",
                suggestion="Fix the syntax error before proceeding"
            )
            problems.append(problem)
            return False, problems
        except Exception as e:
            problem = Problem(
                severity=ProblemSeverity.ERROR,
                category=ProblemCategory.SYNTAX,
                message=f"Parse error: {str(e)}",
                code="E0002",
                source="ast"
            )
            problems.append(problem)
            return False, problems
    
    def _check_imports(self, code: str) -> Tuple[bool, List[Problem]]:
        """
        Check import statements for issues.
        
        Args:
            code: Python code to check
        
        Returns:
            Tuple of (all_valid, problems)
        """
        problems = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check for unused imports (basic check)
                        # This is a simplified check - full analysis would require symbol table
                        pass
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        problem = Problem(
                            severity=ProblemSeverity.WARNING,
                            category=ProblemCategory.IMPORT,
                            message="Relative import without module",
                            line=node.lineno,
                            column=node.col_offset,
                            code="W0401",
                            source="import_checker"
                        )
                        problems.append(problem)
            
            return len(problems) == 0, problems
            
        except Exception as e:
            logger.debug(f"Import check failed: {e}")
            return True, []
    
    def _check_basic_style(self, code: str) -> List[Problem]:
        """
        Perform basic style checks.
        
        Args:
            code: Python code to check
        
        Returns:
            List of problems found
        """
        problems = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, start=1):
            # Check line length
            if len(line) > self.max_line_length:
                problem = Problem(
                    severity=ProblemSeverity.WARNING,
                    category=ProblemCategory.STYLE,
                    message=f"Line too long ({len(line)} > {self.max_line_length} characters)",
                    line=i,
                    code="E501",
                    source="style_checker",
                    suggestion=f"Break line into multiple lines"
                )
                problems.append(problem)
            
            # Check trailing whitespace
            if line.rstrip() != line:
                problem = Problem(
                    severity=ProblemSeverity.INFO,
                    category=ProblemCategory.STYLE,
                    message="Trailing whitespace",
                    line=i,
                    code="W291",
                    source="style_checker",
                    suggestion="Remove trailing whitespace"
                )
                problems.append(problem)
            
            # Check multiple statements on one line
            if ';' in line and not line.strip().startswith('#'):
                problem = Problem(
                    severity=ProblemSeverity.WARNING,
                    category=ProblemCategory.STYLE,
                    message="Multiple statements on one line",
                    line=i,
                    code="E701",
                    source="style_checker",
                    suggestion="Put each statement on its own line"
                )
                problems.append(problem)
        
        return problems
    
    def _run_pylint(self, code: str, filename: str) -> List[Problem]:
        """
        Run pylint on code.
        
        Args:
            code: Python code to check
            filename: Filename for error reporting
        
        Returns:
            List of problems found
        """
        problems = []
        
        try:
            # Write code to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_path = f.name
            
            # Run pylint
            result = subprocess.run(
                ['pylint', '--output-format=json', temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse JSON output
            import json
            if result.stdout:
                pylint_issues = json.loads(result.stdout)
                
                for issue in pylint_issues:
                    severity = self._pylint_severity_to_problem_severity(issue.get('type', 'warning'))
                    category = self._pylint_category_to_problem_category(issue.get('type', 'convention'))
                    
                    problem = Problem(
                        severity=severity,
                        category=category,
                        message=issue.get('message', ''),
                        line=issue.get('line'),
                        column=issue.get('column'),
                        code=issue.get('message-id', ''),
                        source="pylint",
                        suggestion=issue.get('suggestion')
                    )
                    problems.append(problem)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
        except subprocess.TimeoutExpired:
            logger.warning("Pylint timed out")
        except Exception as e:
            logger.debug(f"Pylint check failed: {e}")
        
        return problems
    
    def _run_flake8(self, code: str, filename: str) -> List[Problem]:
        """
        Run flake8 on code.
        
        Args:
            code: Python code to check
            filename: Filename for error reporting
        
        Returns:
            List of problems found
        """
        problems = []
        
        try:
            # Write code to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_path = f.name
            
            # Run flake8
            result = subprocess.run(
                ['flake8', '--max-line-length', str(self.max_line_length), temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output (format: filename:line:column: code message)
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                
                match = re.match(r'(.+):(\d+):(\d+): ([A-Z]\d+) (.+)', line)
                if match:
                    _, line_num, col, code, message = match.groups()
                    
                    severity = self._flake8_code_to_severity(code)
                    category = self._flake8_code_to_category(code)
                    
                    problem = Problem(
                        severity=severity,
                        category=category,
                        message=message,
                        line=int(line_num),
                        column=int(col),
                        code=code,
                        source="flake8"
                    )
                    problems.append(problem)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
        except subprocess.TimeoutExpired:
            logger.warning("Flake8 timed out")
        except Exception as e:
            logger.debug(f"Flake8 check failed: {e}")
        
        return problems
    
    def _check_tool_available(self, tool_name: str) -> bool:
        """
        Check if a command-line tool is available.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            True if available, False otherwise
        """
        try:
            result = subprocess.run(
                [tool_name, '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _pylint_severity_to_problem_severity(self, pylint_type: str) -> ProblemSeverity:
        """Convert pylint issue type to ProblemSeverity."""
        mapping = {
            'error': ProblemSeverity.ERROR,
            'fatal': ProblemSeverity.ERROR,
            'warning': ProblemSeverity.WARNING,
            'refactor': ProblemSeverity.INFO,
            'convention': ProblemSeverity.HINT,
            'info': ProblemSeverity.INFO
        }
        return mapping.get(pylint_type.lower(), ProblemSeverity.WARNING)
    
    def _pylint_category_to_problem_category(self, pylint_type: str) -> ProblemCategory:
        """Convert pylint issue type to ProblemCategory."""
        mapping = {
            'error': ProblemCategory.SYNTAX,
            'fatal': ProblemCategory.SYNTAX,
            'warning': ProblemCategory.STYLE,
            'refactor': ProblemCategory.REFACTOR,
            'convention': ProblemCategory.CONVENTION,
            'info': ProblemCategory.STYLE
        }
        return mapping.get(pylint_type.lower(), ProblemCategory.STYLE)
    
    def _flake8_code_to_severity(self, code: str) -> ProblemSeverity:
        """Convert flake8 error code to ProblemSeverity."""
        if code.startswith('E'):
            return ProblemSeverity.ERROR
        elif code.startswith('W'):
            return ProblemSeverity.WARNING
        elif code.startswith('F'):
            return ProblemSeverity.ERROR
        else:
            return ProblemSeverity.INFO
    
    def _flake8_code_to_category(self, code: str) -> ProblemCategory:
        """Convert flake8 error code to ProblemCategory."""
        if code.startswith('E'):
            return ProblemCategory.STYLE
        elif code.startswith('W'):
            return ProblemCategory.STYLE
        elif code.startswith('F'):
            return ProblemCategory.SYNTAX
        elif code.startswith('C'):
            return ProblemCategory.CONVENTION
        else:
            return ProblemCategory.STYLE
