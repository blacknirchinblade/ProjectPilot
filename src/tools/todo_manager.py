"""
TODO Manager - Find and manage TODO comments in code.

Provides comprehensive TODO tracking:
- Find TODO/FIXME/NOTE/HACK comments
- Extract context (file, line, surrounding code)
- Categorize by priority
- Track completion status
- Generate TODO reports
- Filter and sort TODOs

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from loguru import logger


class TodoPriority(Enum):
    """Priority levels for TODOs."""
    CRITICAL = "critical"   # FIXME - must be fixed
    HIGH = "high"           # TODO - should be done soon
    MEDIUM = "medium"       # NOTE - good to address
    LOW = "low"             # HACK/XXX - technical debt
    INFO = "info"           # Other comments


class TodoStatus(Enum):
    """Status of TODOs."""
    OPEN = "open"           # Not yet addressed
    IN_PROGRESS = "in_progress"  # Being worked on
    COMPLETED = "completed" # Done
    WONTFIX = "wontfix"     # Not going to fix


@dataclass
class TodoItem:
    """Represents a TODO comment."""
    priority: TodoPriority
    message: str
    file_path: Path
    line_number: int
    keyword: str  # The actual keyword found (TODO, FIXME, etc.)
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    author: Optional[str] = None
    date_added: Optional[datetime] = None
    status: TodoStatus = TodoStatus.OPEN
    tags: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.priority.value.upper()}: {self.file_path}:{self.line_number} [{self.keyword}] {self.message}"


@dataclass
class TodoReport:
    """Report of TODOs found."""
    total_todos: int = 0
    by_priority: Dict[TodoPriority, int] = field(default_factory=dict)
    by_status: Dict[TodoStatus, int] = field(default_factory=dict)
    items: List[TodoItem] = field(default_factory=list)
    files_scanned: int = 0
    
    def __str__(self) -> str:
        """String representation."""
        return f"Found {self.total_todos} TODOs in {self.files_scanned} files"


class TodoManager:
    """
    Find and manage TODO comments in code.
    
    Features:
    - Find TODO/FIXME/NOTE/HACK comments
    - Extract context and metadata
    - Categorize by priority
    - Generate reports
    - Filter and sort
    """
    
    # Keywords and their priorities
    KEYWORDS = {
        'FIXME': TodoPriority.CRITICAL,
        'BUG': TodoPriority.CRITICAL,
        'TODO': TodoPriority.HIGH,
        'OPTIMIZE': TodoPriority.HIGH,
        'NOTE': TodoPriority.MEDIUM,
        'REVIEW': TodoPriority.MEDIUM,
        'HACK': TodoPriority.LOW,
        'XXX': TodoPriority.LOW,
        'DEPRECATED': TodoPriority.LOW,
    }
    
    def __init__(
        self,
        root_path: Optional[Path] = None,
        context_lines: int = 2,
        custom_keywords: Optional[Dict[str, TodoPriority]] = None
    ):
        """
        Initialize TODO manager.
        
        Args:
            root_path: Root directory for searching (default: current directory)
            context_lines: Number of context lines to include before/after
            custom_keywords: Additional keywords to search for
        """
        self.root_path = root_path or Path.cwd()
        self.context_lines = context_lines
        
        # Merge custom keywords with defaults
        self.keywords = self.KEYWORDS.copy()
        if custom_keywords:
            self.keywords.update(custom_keywords)
        
        # Build regex pattern for all keywords
        self.pattern = self._build_pattern()
        
        logger.info(f"TodoManager initialized with {len(self.keywords)} keywords")
    
    def find_todos(
        self,
        files: Optional[List[Path]] = None,
        include_pattern: str = "*.py",
        exclude_dirs: Optional[List[str]] = None
    ) -> TodoReport:
        """
        Find all TODO comments in files.
        
        Args:
            files: Specific files to search (None = search all)
            include_pattern: File pattern to include (default: *.py)
            exclude_dirs: Directories to exclude (e.g., ['venv', '__pycache__'])
        
        Returns:
            TodoReport with all found TODOs
        """
        if files is None:
            files = self._get_files(include_pattern, exclude_dirs or [])
        
        report = TodoReport(files_scanned=len(files))
        
        for file_path in files:
            try:
                todos = self._scan_file(file_path)
                report.items.extend(todos)
                report.total_todos += len(todos)
                
                # Update counts
                for todo in todos:
                    report.by_priority[todo.priority] = report.by_priority.get(todo.priority, 0) + 1
                    report.by_status[todo.status] = report.by_status.get(todo.status, 0) + 1
            
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {e}")
        
        logger.info(f"Found {report.total_todos} TODOs in {report.files_scanned} files")
        return report
    
    def find_by_priority(
        self,
        priority: TodoPriority,
        files: Optional[List[Path]] = None
    ) -> List[TodoItem]:
        """
        Find TODOs filtered by priority.
        
        Args:
            priority: Priority level to filter by
            files: Specific files to search
        
        Returns:
            List of matching TODOs
        """
        report = self.find_todos(files)
        return [todo for todo in report.items if todo.priority == priority]
    
    def find_by_keyword(
        self,
        keyword: str,
        files: Optional[List[Path]] = None
    ) -> List[TodoItem]:
        """
        Find TODOs filtered by keyword.
        
        Args:
            keyword: Keyword to search for (e.g., 'TODO', 'FIXME')
            files: Specific files to search
        
        Returns:
            List of matching TODOs
        """
        report = self.find_todos(files)
        return [todo for todo in report.items if todo.keyword.upper() == keyword.upper()]
    
    def find_by_file(
        self,
        file_path: Path
    ) -> List[TodoItem]:
        """
        Find all TODOs in a specific file.
        
        Args:
            file_path: Path to file
        
        Returns:
            List of TODOs in file
        """
        return self._scan_file(file_path)
    
    def generate_report(
        self,
        report: TodoReport,
        sort_by: str = "priority",
        group_by: Optional[str] = None
    ) -> str:
        """
        Generate a formatted TODO report.
        
        Args:
            report: TodoReport to format
            sort_by: Sort field ('priority', 'file', 'line')
            group_by: Group field ('priority', 'file', 'status')
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TODO REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total TODOs: {report.total_todos}")
        lines.append(f"Files Scanned: {report.files_scanned}")
        lines.append("")
        
        # Priority breakdown
        if report.by_priority:
            lines.append("By Priority:")
            for priority in [TodoPriority.CRITICAL, TodoPriority.HIGH, TodoPriority.MEDIUM, TodoPriority.LOW]:
                count = report.by_priority.get(priority, 0)
                if count > 0:
                    lines.append(f"  {priority.value.upper()}: {count}")
            lines.append("")
        
        # Status breakdown
        if report.by_status:
            lines.append("By Status:")
            for status, count in report.by_status.items():
                lines.append(f"  {status.value.upper()}: {count}")
            lines.append("")
        
        # Sort items
        sorted_items = self._sort_todos(report.items, sort_by)
        
        # Group and display
        if group_by == "priority":
            lines.append("-" * 80)
            for priority in [TodoPriority.CRITICAL, TodoPriority.HIGH, TodoPriority.MEDIUM, TodoPriority.LOW]:
                items = [t for t in sorted_items if t.priority == priority]
                if items:
                    lines.append(f"\n{priority.value.upper()} PRIORITY ({len(items)} items):")
                    lines.append("")
                    for i, todo in enumerate(items, 1):
                        lines.extend(self._format_todo(todo, i))
        
        elif group_by == "file":
            lines.append("-" * 80)
            files = sorted(set(t.file_path for t in sorted_items))
            for file_path in files:
                items = [t for t in sorted_items if t.file_path == file_path]
                lines.append(f"\n{file_path} ({len(items)} items):")
                lines.append("")
                for i, todo in enumerate(items, 1):
                    lines.extend(self._format_todo(todo, i))
        
        else:
            # No grouping
            lines.append("-" * 80)
            lines.append("TODOs:")
            lines.append("")
            for i, todo in enumerate(sorted_items, 1):
                lines.extend(self._format_todo(todo, i))
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def _scan_file(self, file_path: Path) -> List[TodoItem]:
        """Scan a file for TODO comments."""
        todos = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                # Check for TODO keywords
                match = self.pattern.search(line)
                if match:
                    keyword = match.group(1)
                    message = line[match.end():].strip()
                    
                    # Remove common comment markers
                    message = message.lstrip('#').lstrip('//').lstrip('/*').lstrip('*').strip()
                    
                    # Extract author if present (format: "TODO(author): message")
                    author = None
                    author_match = re.match(r'\(([^)]+)\):\s*(.+)', message)
                    if author_match:
                        author = author_match.group(1)
                        message = author_match.group(2)
                    
                    # Extract tags if present (format: "[tag1, tag2]")
                    tags = []
                    tag_match = re.search(r'\[([^\]]+)\]', message)
                    if tag_match:
                        tags = [t.strip() for t in tag_match.group(1).split(',')]
                        message = message.replace(tag_match.group(0), '').strip()
                    
                    # Get context lines
                    context_before = []
                    context_after = []
                    
                    if self.context_lines > 0:
                        start = max(0, i - self.context_lines)
                        context_before = lines[start:i]
                        
                        end = min(len(lines), i + self.context_lines + 1)
                        context_after = lines[i + 1:end]
                    
                    todo = TodoItem(
                        priority=self.keywords[keyword.upper()],
                        message=message,
                        file_path=file_path,
                        line_number=i + 1,
                        keyword=keyword,
                        context_before=context_before,
                        context_after=context_after,
                        author=author,
                        tags=tags
                    )
                    todos.append(todo)
        
        except Exception as e:
            logger.debug(f"Error scanning {file_path}: {e}")
        
        return todos
    
    def _build_pattern(self) -> re.Pattern:
        """Build regex pattern for finding TODO keywords."""
        # Match keywords in comments (# or //)
        keywords_str = '|'.join(self.keywords.keys())
        pattern = rf'(?:#|//)\s*({keywords_str})'
        return re.compile(pattern, re.IGNORECASE)
    
    def _get_files(self, include_pattern: str, exclude_dirs: List[str]) -> List[Path]:
        """Get list of files to scan."""
        files = []
        
        for file_path in self.root_path.rglob(include_pattern):
            # Check if file is in excluded directory
            if any(exc_dir in file_path.parts for exc_dir in exclude_dirs):
                continue
            
            if file_path.is_file():
                files.append(file_path)
        
        return files
    
    def _sort_todos(self, todos: List[TodoItem], sort_by: str) -> List[TodoItem]:
        """Sort TODOs by specified field."""
        if sort_by == "priority":
            # Sort by priority (CRITICAL first)
            priority_order = {
                TodoPriority.CRITICAL: 0,
                TodoPriority.HIGH: 1,
                TodoPriority.MEDIUM: 2,
                TodoPriority.LOW: 3,
                TodoPriority.INFO: 4
            }
            return sorted(todos, key=lambda t: (priority_order[t.priority], str(t.file_path), t.line_number))
        
        elif sort_by == "file":
            return sorted(todos, key=lambda t: (str(t.file_path), t.line_number))
        
        elif sort_by == "line":
            return sorted(todos, key=lambda t: (str(t.file_path), t.line_number))
        
        else:
            return todos
    
    def _format_todo(self, todo: TodoItem, index: int) -> List[str]:
        """Format a single TODO item."""
        lines = []
        lines.append(f"{index}. {todo.file_path}:{todo.line_number} [{todo.keyword}]")
        lines.append(f"   Priority: {todo.priority.value.upper()}")
        lines.append(f"   Message: {todo.message}")
        
        if todo.author:
            lines.append(f"   Author: {todo.author}")
        
        if todo.tags:
            lines.append(f"   Tags: {', '.join(todo.tags)}")
        
        if todo.context_before:
            lines.append("   Context:")
            for ctx_line in todo.context_before[-2:]:  # Last 2 lines
                lines.append(f"     {ctx_line}")
        
        lines.append("")
        return lines
    
    def get_statistics(self, report: TodoReport) -> Dict[str, Any]:
        """
        Get statistics about TODOs.
        
        Args:
            report: TodoReport to analyze
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total': report.total_todos,
            'files': report.files_scanned,
            'by_priority': {p.value: report.by_priority.get(p, 0) for p in TodoPriority},
            'by_status': {s.value: report.by_status.get(s, 0) for s in TodoStatus},
            'critical_count': report.by_priority.get(TodoPriority.CRITICAL, 0),
            'high_count': report.by_priority.get(TodoPriority.HIGH, 0),
            'completion_rate': 0.0
        }
        
        # Calculate completion rate
        completed = report.by_status.get(TodoStatus.COMPLETED, 0)
        if report.total_todos > 0:
            stats['completion_rate'] = (completed / report.total_todos) * 100
        
        # Files with most TODOs
        file_counts = {}
        for item in report.items:
            file_counts[item.file_path] = file_counts.get(item.file_path, 0) + 1
        
        if file_counts:
            stats['most_todos_file'] = max(file_counts.items(), key=lambda x: x[1])
        
        return stats
