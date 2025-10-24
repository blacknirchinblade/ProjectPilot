"""
Notebook Editor Tool - Programmatically edit Jupyter notebooks.

Provides comprehensive notebook manipulation:
- Create new notebooks
- Add/remove/edit cells (code, markdown, raw)
- Execute cells
- Manage cell metadata
- Extract cell outputs
- Convert between formats
- Validate notebook structure

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import json
import nbformat
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class CellType(Enum):
    """Types of notebook cells."""
    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


class ExecutionState(Enum):
    """Cell execution states."""
    NOT_EXECUTED = "not_executed"
    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"


@dataclass
class NotebookCell:
    """Represents a notebook cell."""
    cell_type: CellType
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    execution_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to notebook cell dictionary."""
        cell = {
            "cell_type": self.cell_type.value,
            "metadata": self.metadata,
            "source": self.source.split('\n') if '\n' in self.source else [self.source]
        }
        
        if self.cell_type == CellType.CODE:
            cell["outputs"] = self.outputs
            cell["execution_count"] = self.execution_count
        
        return cell
    
    @classmethod
    def from_dict(cls, cell_dict: Dict[str, Any]) -> 'NotebookCell':
        """Create from notebook cell dictionary."""
        cell_type = CellType(cell_dict["cell_type"])
        source = cell_dict.get("source", [])
        
        # Handle source as list or string
        if isinstance(source, list):
            source = ''.join(source)
        
        return cls(
            cell_type=cell_type,
            source=source,
            metadata=cell_dict.get("metadata", {}),
            outputs=cell_dict.get("outputs", []),
            execution_count=cell_dict.get("execution_count")
        )


@dataclass
class NotebookInfo:
    """Information about a notebook."""
    path: Path
    nbformat_version: int
    nbformat_minor: int
    total_cells: int
    code_cells: int
    markdown_cells: int
    raw_cells: int
    kernel: Optional[str] = None
    language: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Notebook: {self.path.name}\n"
            f"  Format: v{self.nbformat_version}.{self.nbformat_minor}\n"
            f"  Cells: {self.total_cells} (code: {self.code_cells}, "
            f"markdown: {self.markdown_cells}, raw: {self.raw_cells})\n"
            f"  Kernel: {self.kernel or 'None'}\n"
            f"  Language: {self.language or 'Unknown'}"
        )


class NotebookEditor:
    """
    Tool for editing Jupyter notebooks programmatically.
    
    Features:
    - Create new notebooks
    - Read/write notebooks
    - Add/remove/edit cells
    - Manage cell metadata
    - Extract outputs
    - Validate structure
    """
    
    def __init__(self, notebook_path: Optional[Path] = None):
        """
        Initialize notebook editor.
        
        Args:
            notebook_path: Path to notebook file (optional)
        """
        self.notebook_path = notebook_path
        self.notebook: Optional[nbformat.NotebookNode] = None
        
        if notebook_path and notebook_path.exists():
            self.load_notebook(notebook_path)
        
        logger.info(f"NotebookEditor initialized{' with ' + str(notebook_path) if notebook_path else ''}")
    
    def create_notebook(
        self,
        kernel_name: str = "python3",
        language: str = "python"
    ) -> nbformat.NotebookNode:
        """
        Create a new empty notebook.
        
        Args:
            kernel_name: Name of kernel (default: python3)
            language: Programming language (default: python)
        
        Returns:
            New notebook node
        """
        self.notebook = nbformat.v4.new_notebook()
        
        # Set kernel info
        self.notebook.metadata['kernelspec'] = {
            'display_name': f'Python 3 ({kernel_name})',
            'language': language,
            'name': kernel_name
        }
        
        self.notebook.metadata['language_info'] = {
            'name': language,
            'version': '3.10.0'
        }
        
        logger.info(f"Created new notebook with kernel: {kernel_name}")
        return self.notebook
    
    def load_notebook(self, path: Path) -> nbformat.NotebookNode:
        """
        Load notebook from file.
        
        Args:
            path: Path to notebook file
        
        Returns:
            Loaded notebook node
        """
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.notebook = nbformat.read(f, as_version=4)
        
        self.notebook_path = path
        logger.info(f"Loaded notebook from {path}")
        return self.notebook
    
    def save_notebook(self, path: Optional[Path] = None) -> Path:
        """
        Save notebook to file.
        
        Args:
            path: Path to save to (uses loaded path if None)
        
        Returns:
            Path where notebook was saved
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded or created")
        
        save_path = path or self.notebook_path
        if save_path is None:
            raise ValueError("No path specified and no notebook path set")
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            nbformat.write(self.notebook, f)
        
        logger.info(f"Saved notebook to {save_path}")
        return save_path
    
    def get_info(self) -> NotebookInfo:
        """
        Get information about the notebook.
        
        Returns:
            NotebookInfo object
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        cells = self.notebook.cells
        code_cells = sum(1 for c in cells if c.cell_type == "code")
        markdown_cells = sum(1 for c in cells if c.cell_type == "markdown")
        raw_cells = sum(1 for c in cells if c.cell_type == "raw")
        
        kernel = self.notebook.metadata.get('kernelspec', {}).get('name')
        language = self.notebook.metadata.get('language_info', {}).get('name')
        
        return NotebookInfo(
            path=self.notebook_path or Path("untitled.ipynb"),
            nbformat_version=self.notebook.nbformat,
            nbformat_minor=self.notebook.nbformat_minor,
            total_cells=len(cells),
            code_cells=code_cells,
            markdown_cells=markdown_cells,
            raw_cells=raw_cells,
            kernel=kernel,
            language=language
        )
    
    def add_cell(
        self,
        cell_type: CellType,
        source: str,
        index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a cell to the notebook.
        
        Args:
            cell_type: Type of cell (CODE, MARKDOWN, RAW)
            source: Cell source code/content
            index: Position to insert (None = append)
            metadata: Cell metadata
        
        Returns:
            Index where cell was inserted
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded or created")
        
        # Create cell
        if cell_type == CellType.CODE:
            cell = nbformat.v4.new_code_cell(source)
        elif cell_type == CellType.MARKDOWN:
            cell = nbformat.v4.new_markdown_cell(source)
        else:  # RAW
            cell = nbformat.v4.new_raw_cell(source)
        
        # Set metadata
        if metadata:
            cell.metadata.update(metadata)
        
        # Insert cell
        if index is None:
            self.notebook.cells.append(cell)
            index = len(self.notebook.cells) - 1
        else:
            self.notebook.cells.insert(index, cell)
        
        logger.info(f"Added {cell_type.value} cell at index {index}")
        return index
    
    def remove_cell(self, index: int) -> NotebookCell:
        """
        Remove a cell from the notebook.
        
        Args:
            index: Index of cell to remove
        
        Returns:
            Removed cell
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range")
        
        cell = self.notebook.cells.pop(index)
        removed = NotebookCell.from_dict(cell)
        
        logger.info(f"Removed cell at index {index}")
        return removed
    
    def edit_cell(
        self,
        index: int,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Edit an existing cell.
        
        Args:
            index: Index of cell to edit
            source: New source code (None = keep existing)
            metadata: New metadata (None = keep existing)
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range")
        
        cell = self.notebook.cells[index]
        
        if source is not None:
            cell.source = source
        
        if metadata is not None:
            cell.metadata.update(metadata)
        
        logger.info(f"Edited cell at index {index}")
    
    def get_cell(self, index: int) -> NotebookCell:
        """
        Get a cell from the notebook.
        
        Args:
            index: Index of cell
        
        Returns:
            NotebookCell object
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range")
        
        cell_dict = self.notebook.cells[index]
        return NotebookCell.from_dict(cell_dict)
    
    def get_all_cells(self) -> List[NotebookCell]:
        """
        Get all cells from the notebook.
        
        Returns:
            List of NotebookCell objects
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        return [NotebookCell.from_dict(cell) for cell in self.notebook.cells]
    
    def clear_outputs(self, indices: Optional[List[int]] = None) -> int:
        """
        Clear outputs from code cells.
        
        Args:
            indices: List of cell indices (None = all code cells)
        
        Returns:
            Number of cells cleared
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        cleared = 0
        
        if indices is None:
            # Clear all code cells
            for cell in self.notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
                    cleared += 1
        else:
            # Clear specific cells
            for idx in indices:
                if 0 <= idx < len(self.notebook.cells):
                    cell = self.notebook.cells[idx]
                    if cell.cell_type == "code":
                        cell.outputs = []
                        cell.execution_count = None
                        cleared += 1
        
        logger.info(f"Cleared outputs from {cleared} cells")
        return cleared
    
    def find_cells_by_content(
        self,
        pattern: str,
        cell_type: Optional[CellType] = None,
        case_sensitive: bool = False
    ) -> List[int]:
        """
        Find cells containing a pattern.
        
        Args:
            pattern: Text pattern to search for
            cell_type: Filter by cell type (None = all types)
            case_sensitive: Whether search is case-sensitive
        
        Returns:
            List of matching cell indices
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        matches = []
        
        for i, cell in enumerate(self.notebook.cells):
            # Filter by type
            if cell_type and cell.cell_type != cell_type.value:
                continue
            
            # Search in source
            source = cell.source
            search_text = source if case_sensitive else source.lower()
            search_pattern = pattern if case_sensitive else pattern.lower()
            
            if search_pattern in search_text:
                matches.append(i)
        
        logger.info(f"Found {len(matches)} cells matching '{pattern}'")
        return matches
    
    def get_code_cells(self) -> List[tuple[int, str]]:
        """
        Get all code cells with their indices.
        
        Returns:
            List of (index, source) tuples
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        return [
            (i, cell.source)
            for i, cell in enumerate(self.notebook.cells)
            if cell.cell_type == "code"
        ]
    
    def get_markdown_cells(self) -> List[tuple[int, str]]:
        """
        Get all markdown cells with their indices.
        
        Returns:
            List of (index, source) tuples
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        return [
            (i, cell.source)
            for i, cell in enumerate(self.notebook.cells)
            if cell.cell_type == "markdown"
        ]
    
    def extract_code(self, output_path: Optional[Path] = None) -> str:
        """
        Extract all code from code cells into a Python script.
        
        Args:
            output_path: Path to save script (None = return string only)
        
        Returns:
            Extracted Python code
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        code_lines = []
        
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                code_lines.append(f"# Cell {i}")
                code_lines.append(cell.source)
                code_lines.append("")  # Blank line
        
        code = '\n'.join(code_lines)
        
        if output_path:
            output_path.write_text(code, encoding='utf-8')
            logger.info(f"Extracted code to {output_path}")
        
        return code
    
    def validate_notebook(self) -> Dict[str, Any]:
        """
        Validate notebook structure.
        
        Returns:
            Validation results dictionary
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        errors = []
        warnings = []
        
        # Check format version
        if self.notebook.nbformat < 4:
            warnings.append(f"Old notebook format: v{self.notebook.nbformat}")
        
        # Check cells
        for i, cell in enumerate(self.notebook.cells):
            # Check cell type
            if cell.cell_type not in ["code", "markdown", "raw"]:
                errors.append(f"Cell {i}: Invalid cell type '{cell.cell_type}'")
            
            # Check source
            if not hasattr(cell, 'source'):
                errors.append(f"Cell {i}: Missing source")
            
            # Check code cell structure
            if cell.cell_type == "code":
                if not hasattr(cell, 'outputs'):
                    warnings.append(f"Cell {i}: Code cell missing outputs")
        
        # Check metadata
        if 'kernelspec' not in self.notebook.metadata:
            warnings.append("Missing kernelspec metadata")
        
        if 'language_info' not in self.notebook.metadata:
            warnings.append("Missing language_info metadata")
        
        is_valid = len(errors) == 0
        
        result = {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "cell_count": len(self.notebook.cells)
        }
        
        logger.info(f"Validation: {'PASS' if is_valid else 'FAIL'} "
                   f"({len(errors)} errors, {len(warnings)} warnings)")
        
        return result
    
    def merge_cells(self, start: int, end: int, separator: str = "\n\n") -> int:
        """
        Merge consecutive cells into one.
        
        Args:
            start: Starting cell index
            end: Ending cell index (inclusive)
            separator: Text to join cells with
        
        Returns:
            Index of merged cell
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        if start < 0 or end >= len(self.notebook.cells) or start > end:
            raise IndexError("Invalid cell range")
        
        # Get cells to merge
        cells_to_merge = self.notebook.cells[start:end + 1]
        
        # Check all cells are same type
        cell_types = {cell.cell_type for cell in cells_to_merge}
        if len(cell_types) > 1:
            raise ValueError("Cannot merge cells of different types")
        
        # Merge sources
        merged_source = separator.join(cell.source for cell in cells_to_merge)
        
        # Create merged cell
        cell_type = cells_to_merge[0].cell_type
        if cell_type == "code":
            merged_cell = nbformat.v4.new_code_cell(merged_source)
        elif cell_type == "markdown":
            merged_cell = nbformat.v4.new_markdown_cell(merged_source)
        else:
            merged_cell = nbformat.v4.new_raw_cell(merged_source)
        
        # Remove old cells and insert merged cell
        for _ in range(end - start + 1):
            self.notebook.cells.pop(start)
        
        self.notebook.cells.insert(start, merged_cell)
        
        logger.info(f"Merged cells {start}-{end} into cell {start}")
        return start
    
    def split_cell(self, index: int, split_at: int) -> tuple[int, int]:
        """
        Split a cell into two cells.
        
        Args:
            index: Index of cell to split
            split_at: Line number to split at
        
        Returns:
            Tuple of (first_cell_index, second_cell_index)
        """
        if self.notebook is None:
            raise ValueError("No notebook loaded")
        
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range")
        
        cell = self.notebook.cells[index]
        lines = cell.source.split('\n')
        
        if split_at < 1 or split_at >= len(lines):
            raise ValueError(f"Invalid split position {split_at}")
        
        # Split source
        first_source = '\n'.join(lines[:split_at])
        second_source = '\n'.join(lines[split_at:])
        
        # Create new cells
        cell_type = cell.cell_type
        if cell_type == "code":
            first_cell = nbformat.v4.new_code_cell(first_source)
            second_cell = nbformat.v4.new_code_cell(second_source)
        elif cell_type == "markdown":
            first_cell = nbformat.v4.new_markdown_cell(first_source)
            second_cell = nbformat.v4.new_markdown_cell(second_source)
        else:
            first_cell = nbformat.v4.new_raw_cell(first_source)
            second_cell = nbformat.v4.new_raw_cell(second_source)
        
        # Replace old cell with two new cells
        self.notebook.cells.pop(index)
        self.notebook.cells.insert(index, second_cell)
        self.notebook.cells.insert(index, first_cell)
        
        logger.info(f"Split cell {index} into cells {index} and {index + 1}")
        return (index, index + 1)
