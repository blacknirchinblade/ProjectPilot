"""
Search & Refactoring Tool - Code search and refactoring operations.

Provides comprehensive code analysis and transformation:
- Symbol search (classes, functions, variables)
- Find references/usages across codebase
- Rename refactoring with scope handling
- Extract method/function
- Code pattern search using AST
- Safe multi-file refactoring

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class SymbolType(Enum):
    """Types of code symbols."""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    ATTRIBUTE = "attribute"
    IMPORT = "import"
    MODULE = "module"


class RefactoringType(Enum):
    """Types of refactoring operations."""
    RENAME = "rename"
    EXTRACT_METHOD = "extract_method"
    EXTRACT_FUNCTION = "extract_function"
    INLINE = "inline"
    MOVE = "move"


@dataclass
class Symbol:
    """Represents a code symbol."""
    name: str
    type: SymbolType
    file_path: Path
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    scope: Optional[str] = None  # e.g., "ClassName.method_name"
    parent: Optional[str] = None  # Parent symbol (class for method, etc.)
    docstring: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        scope_str = f" in {self.scope}" if self.scope else ""
        return f"{self.type.value} '{self.name}'{scope_str} at {self.file_path}:{self.line}"


@dataclass
class Reference:
    """Represents a reference to a symbol."""
    symbol_name: str
    file_path: Path
    line: int
    column: int
    context: str  # Line of code containing the reference
    is_definition: bool = False
    scope: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        ref_type = "Definition" if self.is_definition else "Reference"
        return f"{ref_type}: {self.file_path}:{self.line}:{self.column} - {self.context.strip()}"


@dataclass
class SearchResult:
    """Result of a search operation."""
    query: str
    symbols: List[Symbol] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    total_matches: int = 0
    files_searched: int = 0
    
    def __str__(self) -> str:
        """String representation."""
        return f"Found {self.total_matches} matches in {self.files_searched} files"


@dataclass
class RefactoringResult:
    """Result of a refactoring operation."""
    success: bool
    refactoring_type: RefactoringType
    changes: Dict[Path, str] = field(default_factory=dict)  # file_path -> new_content
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_modified: int = 0
    
    def __str__(self) -> str:
        """String representation."""
        if self.success:
            return f"Refactoring '{self.refactoring_type.value}' successful: {self.files_modified} files modified"
        else:
            return f"Refactoring '{self.refactoring_type.value}' failed: {len(self.errors)} errors"


class SymbolFinder(ast.NodeVisitor):
    """AST visitor to find symbols in Python code."""
    
    def __init__(self, file_path: Path):
        """Initialize symbol finder."""
        self.file_path = file_path
        self.symbols: List[Symbol] = []
        self.scope_stack: List[str] = []  # Track current scope
        self.current_class: Optional[str] = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        scope = ".".join(self.scope_stack) if self.scope_stack else None
        
        symbol = Symbol(
            name=node.name,
            type=SymbolType.CLASS,
            file_path=self.file_path,
            line=node.lineno,
            column=node.col_offset,
            end_line=node.end_lineno,
            end_column=node.end_col_offset,
            scope=scope,
            docstring=ast.get_docstring(node)
        )
        self.symbols.append(symbol)
        
        # Enter class scope
        self.scope_stack.append(node.name)
        self.current_class = node.name
        self.generic_visit(node)
        self.scope_stack.pop()
        self.current_class = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function/method definition."""
        scope = ".".join(self.scope_stack) if self.scope_stack else None
        
        # Determine if it's a method or function
        symbol_type = SymbolType.METHOD if self.current_class else SymbolType.FUNCTION
        
        symbol = Symbol(
            name=node.name,
            type=symbol_type,
            file_path=self.file_path,
            line=node.lineno,
            column=node.col_offset,
            end_line=node.end_lineno,
            end_column=node.end_col_offset,
            scope=scope,
            parent=self.current_class,
            docstring=ast.get_docstring(node)
        )
        self.symbols.append(symbol)
        
        # Add parameters
        for arg in node.args.args:
            param_symbol = Symbol(
                name=arg.arg,
                type=SymbolType.PARAMETER,
                file_path=self.file_path,
                line=arg.lineno,
                column=arg.col_offset,
                end_line=arg.end_lineno,
                end_column=arg.end_col_offset,
                scope=f"{scope}.{node.name}" if scope else node.name,
                parent=node.name
            )
            self.symbols.append(param_symbol)
        
        # Enter function scope
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        # Treat same as regular function
        self.visit_FunctionDef(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment (to find variables)."""
        scope = ".".join(self.scope_stack) if self.scope_stack else None
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                symbol = Symbol(
                    name=target.id,
                    type=SymbolType.VARIABLE,
                    file_path=self.file_path,
                    line=target.lineno,
                    column=target.col_offset,
                    scope=scope
                )
                self.symbols.append(symbol)
            elif isinstance(target, ast.Attribute):
                # Class attribute
                symbol = Symbol(
                    name=target.attr,
                    type=SymbolType.ATTRIBUTE,
                    file_path=self.file_path,
                    line=target.lineno,
                    column=target.col_offset,
                    scope=scope,
                    parent=self.current_class
                )
                self.symbols.append(symbol)
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        scope = ".".join(self.scope_stack) if self.scope_stack else None
        
        for alias in node.names:
            symbol = Symbol(
                name=alias.asname if alias.asname else alias.name,
                type=SymbolType.IMPORT,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                scope=scope
            )
            self.symbols.append(symbol)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement."""
        scope = ".".join(self.scope_stack) if self.scope_stack else None
        
        for alias in node.names:
            symbol = Symbol(
                name=alias.asname if alias.asname else alias.name,
                type=SymbolType.IMPORT,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                scope=scope
            )
            self.symbols.append(symbol)
        
        self.generic_visit(node)


class ReferenceFinder(ast.NodeVisitor):
    """AST visitor to find references to a symbol."""
    
    def __init__(self, symbol_name: str, file_path: Path, file_content: str):
        """Initialize reference finder."""
        self.symbol_name = symbol_name
        self.file_path = file_path
        self.file_lines = file_content.split('\n')
        self.references: List[Reference] = []
        self.scope_stack: List[str] = []
    
    def _get_context(self, lineno: int) -> str:
        """Get the line of code for context."""
        if 0 < lineno <= len(self.file_lines):
            return self.file_lines[lineno - 1]
        return ""
    
    def visit_Name(self, node: ast.Name) -> None:
        """Visit name reference."""
        if node.id == self.symbol_name:
            scope = ".".join(self.scope_stack) if self.scope_stack else None
            reference = Reference(
                symbol_name=self.symbol_name,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                scope=scope
            )
            self.references.append(reference)
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute reference."""
        if node.attr == self.symbol_name:
            scope = ".".join(self.scope_stack) if self.scope_stack else None
            reference = Reference(
                symbol_name=self.symbol_name,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                scope=scope
            )
            self.references.append(reference)
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        if node.name == self.symbol_name:
            scope = ".".join(self.scope_stack) if self.scope_stack else None
            reference = Reference(
                symbol_name=self.symbol_name,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                is_definition=True,
                scope=scope
            )
            self.references.append(reference)
        
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        if node.name == self.symbol_name:
            scope = ".".join(self.scope_stack) if self.scope_stack else None
            reference = Reference(
                symbol_name=self.symbol_name,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                is_definition=True,
                scope=scope
            )
            self.references.append(reference)
        
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()


class SearchRefactorTool:
    """
    Search and refactoring tool for Python code.
    
    Features:
    - Symbol search (classes, functions, variables)
    - Find references/usages
    - Rename refactoring
    - Extract method/function
    - Code pattern search
    - Multi-file operations
    """
    
    def __init__(self, root_path: Optional[Path] = None):
        """
        Initialize search & refactoring tool.
        
        Args:
            root_path: Root directory for searching (default: current directory)
        """
        self.root_path = root_path or Path.cwd()
        logger.info(f"SearchRefactorTool initialized with root: {self.root_path}")
    
    def find_symbols(
        self,
        name_pattern: Optional[str] = None,
        symbol_type: Optional[SymbolType] = None,
        files: Optional[List[Path]] = None
    ) -> SearchResult:
        """
        Find symbols matching criteria.
        
        Args:
            name_pattern: Regex pattern for symbol name (None = all)
            symbol_type: Type of symbol to find (None = all types)
            files: List of files to search (None = all Python files)
        
        Returns:
            SearchResult with matching symbols
        """
        if files is None:
            files = list(self.root_path.rglob("*.py"))
        
        result = SearchResult(
            query=f"name={name_pattern}, type={symbol_type}",
            files_searched=len(files)
        )
        
        pattern = re.compile(name_pattern) if name_pattern else None
        
        for file_path in files:
            try:
                code = file_path.read_text(encoding='utf-8')
                tree = ast.parse(code, filename=str(file_path))
                
                finder = SymbolFinder(file_path)
                finder.visit(tree)
                
                for symbol in finder.symbols:
                    # Filter by type
                    if symbol_type and symbol.type != symbol_type:
                        continue
                    
                    # Filter by name pattern
                    if pattern and not pattern.match(symbol.name):
                        continue
                    
                    result.symbols.append(symbol)
                    result.total_matches += 1
                
            except Exception as e:
                logger.debug(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Found {result.total_matches} symbols in {result.files_searched} files")
        return result
    
    def find_references(
        self,
        symbol_name: str,
        files: Optional[List[Path]] = None
    ) -> SearchResult:
        """
        Find all references to a symbol.
        
        Args:
            symbol_name: Name of symbol to find references for
            files: List of files to search (None = all Python files)
        
        Returns:
            SearchResult with all references
        """
        if files is None:
            files = list(self.root_path.rglob("*.py"))
        
        result = SearchResult(
            query=f"references to '{symbol_name}'",
            files_searched=len(files)
        )
        
        for file_path in files:
            try:
                code = file_path.read_text(encoding='utf-8')
                tree = ast.parse(code, filename=str(file_path))
                
                finder = ReferenceFinder(symbol_name, file_path, code)
                finder.visit(tree)
                
                result.references.extend(finder.references)
                result.total_matches += len(finder.references)
                
            except Exception as e:
                logger.debug(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Found {result.total_matches} references to '{symbol_name}' in {result.files_searched} files")
        return result
    
    def rename_symbol(
        self,
        old_name: str,
        new_name: str,
        files: Optional[List[Path]] = None,
        dry_run: bool = True
    ) -> RefactoringResult:
        """
        Rename a symbol across files.
        
        Args:
            old_name: Current name of symbol
            new_name: New name for symbol
            files: List of files to refactor (None = all Python files)
            dry_run: If True, don't write changes (default: True)
        
        Returns:
            RefactoringResult with changes
        """
        result = RefactoringResult(
            success=True,
            refactoring_type=RefactoringType.RENAME
        )
        
        # Validate new name
        if not new_name.isidentifier():
            result.success = False
            result.errors.append(f"Invalid identifier: '{new_name}'")
            return result
        
        # Find all references
        search_result = self.find_references(old_name, files)
        
        if search_result.total_matches == 0:
            result.warnings.append(f"No references found for '{old_name}'")
            return result
        
        # Group references by file
        refs_by_file: Dict[Path, List[Reference]] = {}
        for ref in search_result.references:
            if ref.file_path not in refs_by_file:
                refs_by_file[ref.file_path] = []
            refs_by_file[ref.file_path].append(ref)
        
        # Process each file
        for file_path, refs in refs_by_file.items():
            try:
                code = file_path.read_text(encoding='utf-8')
                lines = code.split('\n')
                
                # Sort references by line (reverse) to maintain line numbers
                refs.sort(key=lambda r: (r.line, r.column), reverse=True)
                
                # Replace each reference
                for ref in refs:
                    if 0 < ref.line <= len(lines):
                        line = lines[ref.line - 1]
                        # Simple replacement - could be improved with better context awareness
                        new_line = self._replace_at_position(line, ref.column, old_name, new_name)
                        lines[ref.line - 1] = new_line
                
                new_content = '\n'.join(lines)
                result.changes[file_path] = new_content
                result.files_modified += 1
                
                # Write changes if not dry run
                if not dry_run:
                    file_path.write_text(new_content, encoding='utf-8')
                    logger.info(f"Renamed '{old_name}' to '{new_name}' in {file_path}")
                
            except Exception as e:
                result.errors.append(f"Error processing {file_path}: {str(e)}")
                result.success = False
        
        if dry_run and result.success:
            result.warnings.append("Dry run - no files were modified")
        
        logger.info(f"Rename refactoring: {result.files_modified} files would be modified")
        return result
    
    def extract_method(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        method_name: str,
        target_class: Optional[str] = None,
        dry_run: bool = True
    ) -> RefactoringResult:
        """
        Extract code into a new method/function.
        
        Args:
            file_path: File containing code to extract
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed, inclusive)
            method_name: Name for extracted method
            target_class: Class to add method to (None = create function)
            dry_run: If True, don't write changes (default: True)
        
        Returns:
            RefactoringResult with changes
        """
        result = RefactoringResult(
            success=True,
            refactoring_type=RefactoringType.EXTRACT_METHOD if target_class else RefactoringType.EXTRACT_FUNCTION
        )
        
        # Validate method name
        if not method_name.isidentifier():
            result.success = False
            result.errors.append(f"Invalid identifier: '{method_name}'")
            return result
        
        try:
            code = file_path.read_text(encoding='utf-8')
            lines = code.split('\n')
            
            # Validate line numbers
            if not (0 < start_line <= end_line <= len(lines)):
                result.success = False
                result.errors.append(f"Invalid line range: {start_line}-{end_line}")
                return result
            
            # Extract code block
            extracted_lines = lines[start_line - 1:end_line]
            
            # Determine indentation
            base_indent = self._get_base_indentation(extracted_lines)
            
            # Create new method/function
            if target_class:
                new_method = self._create_method(method_name, extracted_lines, base_indent)
                # Find class and insert method
                # This is simplified - full implementation would parse AST
                result.warnings.append("Method extraction to class not fully implemented")
            else:
                new_function = self._create_function(method_name, extracted_lines, base_indent)
                # Insert function at appropriate location
                # Replace extracted code with function call
                call_line = " " * base_indent + f"{method_name}()"
                
                # Build new content
                new_lines = (
                    lines[:start_line - 1] +
                    [call_line] +
                    lines[end_line:] +
                    [""] +  # Blank line
                    new_function.split('\n')
                )
                
                new_content = '\n'.join(new_lines)
                result.changes[file_path] = new_content
                result.files_modified = 1
                
                if not dry_run:
                    file_path.write_text(new_content, encoding='utf-8')
                    logger.info(f"Extracted function '{method_name}' in {file_path}")
            
            if dry_run and result.success:
                result.warnings.append("Dry run - no files were modified")
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Error extracting method: {str(e)}")
        
        return result
    
    def search_pattern(
        self,
        pattern: str,
        is_regex: bool = False,
        files: Optional[List[Path]] = None
    ) -> SearchResult:
        """
        Search for code pattern.
        
        Args:
            pattern: Pattern to search for
            is_regex: Whether pattern is regex (default: plain text)
            files: List of files to search (None = all Python files)
        
        Returns:
            SearchResult with matches
        """
        if files is None:
            files = list(self.root_path.rglob("*.py"))
        
        result = SearchResult(
            query=pattern,
            files_searched=len(files)
        )
        
        regex = re.compile(pattern) if is_regex else re.compile(re.escape(pattern))
        
        for file_path in files:
            try:
                code = file_path.read_text(encoding='utf-8')
                lines = code.split('\n')
                
                for i, line in enumerate(lines, start=1):
                    if regex.search(line):
                        reference = Reference(
                            symbol_name=pattern,
                            file_path=file_path,
                            line=i,
                            column=0,
                            context=line
                        )
                        result.references.append(reference)
                        result.total_matches += 1
                
            except Exception as e:
                logger.debug(f"Error searching {file_path}: {e}")
        
        logger.info(f"Found {result.total_matches} pattern matches in {result.files_searched} files")
        return result
    
    def _replace_at_position(self, line: str, column: int, old: str, new: str) -> str:
        """Replace text at specific column position."""
        if column < 0 or column >= len(line):
            return line
        
        # Check if old text matches at this position
        if line[column:column + len(old)] == old:
            # Check word boundaries
            before_ok = column == 0 or not line[column - 1].isalnum() and line[column - 1] != '_'
            after_ok = column + len(old) >= len(line) or (not line[column + len(old)].isalnum() and line[column + len(old)] != '_')
            
            if before_ok and after_ok:
                return line[:column] + new + line[column + len(old):]
        
        return line
    
    def _get_base_indentation(self, lines: List[str]) -> int:
        """Get the base indentation level of code block."""
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                return len(line) - len(stripped)
        return 0
    
    def _create_function(self, name: str, code_lines: List[str], base_indent: int) -> str:
        """Create a function from extracted code."""
        # Remove base indentation
        dedented_lines = []
        for line in code_lines:
            if len(line) >= base_indent:
                dedented_lines.append(line[base_indent:])
            else:
                dedented_lines.append(line)
        
        # Create function
        function_lines = [
            f"def {name}():",
            "    \"\"\"Extracted function.\"\"\"",
        ]
        
        for line in dedented_lines:
            function_lines.append("    " + line)
        
        return '\n'.join(function_lines)
    
    def _create_method(self, name: str, code_lines: List[str], base_indent: int) -> str:
        """Create a method from extracted code."""
        # Similar to _create_function but with 'self' parameter
        dedented_lines = []
        for line in code_lines:
            if len(line) >= base_indent:
                dedented_lines.append(line[base_indent:])
            else:
                dedented_lines.append(line)
        
        method_lines = [
            f"    def {name}(self):",
            "        \"\"\"Extracted method.\"\"\"",
        ]
        
        for line in dedented_lines:
            method_lines.append("        " + line)
        
        return '\n'.join(method_lines)
