"""
Interactive Modification Agent (Corrected)

Handles targeted code modifications with AST-based validation.
This agent provides safe, intelligent code modifications.

Key Features:
- Targeted file modifications with AST parsing
- Function and class insertion at correct locations
- Syntax validation before applying changes
- Import statement management
- Refactoring with pattern matching
- Rollback support for failed modifications

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class InteractiveModificationAgent(BaseAgent):
    """
    Agent for safe, intelligent code modifications
    
    This agent uses AST parsing and syntax validation to ensure
    all code modifications are syntactically correct before applying them.
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        project_root: Optional[str] = None
    ):
        """
        Initialize InteractiveModificationAgent
        
        Args:
            llm_client: LLM client for intelligent suggestions
            prompt_manager: Prompt template manager
            project_root: Root directory of the project (defaults to cwd)
        """
        super().__init__(
            agent_type="code_modification",
            name="modification_agent",
            role="Expert Code Modification Specialist with AST-based Validation",
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.modification_history: List[Dict[str, Any]] = []
        
        logger.info(f"{self.name} initialized")
        logger.info(f"   - Project root: {self.project_root}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a modification task
        
        Args:
            task: Task with:
                - action: "modify_file" | "add_function" | "add_class" | "refactor"
                - filepath: Target file path
                - changes: Modification details
        
        Returns:
            Result dictionary with success status and details
        """
        action = task.get("action")
        filepath = task.get("filepath")
        
        if action == "modify_file":
            return await self.modify_file(filepath, task.get("changes"))
        elif action == "add_function":
            return await self.add_function(
                filepath,
                task.get("function_code"),
                task.get("location")
            )
        elif action == "add_class":
            return await self.add_class(filepath, task.get("class_code"))
        elif action == "refactor":
            return await self.refactor(
                filepath,
                task.get("pattern"),
                task.get("replacement")
            )
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}"
            }
    
    async def modify_file(
        self,
        filepath: str,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply targeted modifications to a file
        
        Args:
            filepath: Path to file (relative to project root)
            changes: Dictionary with:
                - old_code: Code to replace (exact match)
                - new_code: Replacement code
                - validate: Whether to validate syntax (default: True)
        
        Returns:
            {
                "status": "success" | "error",
                "filepath": str,
                "changes_applied": int,
                "message": str,
                "backup_path": Optional[str]
            }
        """
        try:
            full_path = self._resolve_path(filepath)
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            old_code = changes.get("old_code", "")
            new_code = changes.get("new_code", "")
            validate = changes.get("validate", True)
            
            if not old_code:
                return {
                    "status": "error",
                    "message": "old_code is required"
                }
            
            if old_code not in original_content:
                return {
                    "status": "error",
                    "message": f"old_code not found in {filepath}",
                    "suggestion": "Check for whitespace/indentation differences"
                }
            
            modified_content = original_content.replace(old_code, new_code, 1)
            
            if validate and filepath.endswith('.py'):
                syntax_check = self._validate_python_syntax(modified_content)
                if not syntax_check["valid"]:
                    return {
                        "status": "error",
                        "message": f"Syntax error: {syntax_check['error']}",
                        "line": syntax_check.get("line"),
                        "column": syntax_check.get("column")
                    }
            
            backup_path = self._create_backup(full_path, original_content)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            self._record_modification({
                "action": "modify_file",
                "filepath": str(filepath),
                "backup_path": str(backup_path),
                "changes": 1
            })
            
            logger.info(f"✅ Modified {filepath}")
            
            return {
                "status": "success",
                "filepath": str(filepath),
                "changes_applied": 1,
                "message": f"Successfully modified {filepath}",
                "backup_path": str(backup_path)
            }
            
        except Exception as e:
            logger.error(f"Error modifying {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def add_function(
        self,
        filepath: str,
        function_code: str,
        location: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new function to a Python file
        """
        try:
            full_path = self._resolve_path(filepath)
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            syntax_check = self._validate_python_syntax(function_code)
            if not syntax_check["valid"]:
                return {
                    "status": "error",
                    "message": f"Invalid function syntax: {syntax_check['error']}"
                }
            
            location = location or {"type": "end"}
            insertion_point = self._find_insertion_point(
                original_content,
                location
            )
            
            if insertion_point is None:
                return {
                    "status": "error",
                    "message": "Could not determine insertion point"
                }
            
            lines = original_content.split('\n')
            lines.insert(insertion_point, function_code)
            modified_content = '\n'.join(lines)
            
            syntax_check = self._validate_python_syntax(modified_content)
            if not syntax_check["valid"]:
                return {
                    "status": "error",
                    "message": f"File syntax error after insertion: {syntax_check['error']}"
                }
            
            backup_path = self._create_backup(full_path, original_content)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            self._record_modification({
                "action": "add_function",
                "filepath": str(filepath),
                "backup_path": str(backup_path),
                "insertion_line": insertion_point
            })
            
            logger.info(f"✅ Added function to {filepath} at line {insertion_point}")
            
            return {
                "status": "success",
                "filepath": str(filepath),
                "message": f"Function added at line {insertion_point}",
                "backup_path": str(backup_path),
                "insertion_line": insertion_point
            }
            
        except Exception as e:
            logger.error(f"Error adding function to {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def add_class(
        self,
        filepath: str,
        class_code: str,
        location: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new class to a Python file
        """
        try:
            full_path = self._resolve_path(filepath)
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            syntax_check = self._validate_python_syntax(class_code)
            if not syntax_check["valid"]:
                return {
                    "status": "error",
                    "message": f"Invalid class syntax: {syntax_check['error']}"
                }
            
            location = location or {"type": "end"}
            insertion_point = self._find_insertion_point(
                original_content,
                location
            )
            
            if insertion_point is None:
                return {
                    "status": "error",
                    "message": "Could not determine insertion point"
                }
            
            lines = original_content.split('\n')
            lines.insert(insertion_point, class_code)
            modified_content = '\n'.join(lines)
            
            syntax_check = self._validate_python_syntax(modified_content)
            if not syntax_check["valid"]:
                return {
                    "status": "error",
                    "message": f"File syntax error after insertion: {syntax_check['error']}"
                }
            
            backup_path = self._create_backup(full_path, original_content)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            self._record_modification({
                "action": "add_class",
                "filepath": str(filepath),
                "backup_path": str(backup_path),
                "insertion_line": insertion_point
            })
            
            logger.info(f"✅ Added class to {filepath} at line {insertion_point}")
            
            return {
                "status": "success",
                "filepath": str(filepath),
                "message": f"Class added at line {insertion_point}",
                "backup_path": str(backup_path),
                "insertion_line": insertion_point
            }
            
        except Exception as e:
            logger.error(f"Error adding class to {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def refactor(
        self,
        filepath: str,
        pattern: str,
        replacement: str,
        is_regex: bool = False
    ) -> Dict[str, Any]:
        """
        Refactor code using pattern matching
        """
        try:
            full_path = self._resolve_path(filepath)
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            if is_regex:
                modified_content, count = re.subn(pattern, replacement, original_content)
            else:
                count = original_content.count(pattern)
                modified_content = original_content.replace(pattern, replacement)
            
            if count == 0:
                return {
                    "status": "success",
                    "message": "No matches found for pattern",
                    "replacements": 0
                }
            
            if filepath.endswith('.py'):
                syntax_check = self._validate_python_syntax(modified_content)
                if not syntax_check["valid"]:
                    return {
                        "status": "error",
                        "message": f"Refactoring caused syntax error: {syntax_check['error']}",
                        "replacements_attempted": count
                    }
            
            backup_path = self._create_backup(full_path, original_content)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            self._record_modification({
                "action": "refactor",
                "filepath": str(filepath),
                "backup_path": str(backup_path),
                "replacements": count
            })
            
            logger.info(f"✅ Refactored {filepath} ({count} replacements)")
            
            return {
                "status": "success",
                "filepath": str(filepath),
                "replacements": count,
                "message": f"Applied {count} refactoring changes",
                "backup_path": str(backup_path)
            }
            
        except Exception as e:
            logger.error(f"Error refactoring {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def rollback(self, backup_path: str) -> Dict[str, Any]:
        """
        Rollback a modification using its backup
        """
        try:
            backup = Path(backup_path)
            
            if not backup.exists():
                return {
                    "status": "error",
                    "message": f"Backup not found: {backup_path}"
                }
            
            original_name = backup.stem.rsplit('.backup_', 1)[0]
            original_path = backup.parent / original_name
            
            with open(backup, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            
            with open(original_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            
            logger.info(f"✅ Rolled back {original_path} from {backup_path}")
            
            return {
                "status": "success",
                "message": f"Rolled back to {backup_path}",
                "restored_file": str(original_path)
            }
            
        except Exception as e:
            logger.error(f"Error rolling back {backup_path}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    # Helper methods
    
    def _resolve_path(self, filepath: str) -> Path:
        """Resolve filepath relative to project root"""
        path = Path(filepath)
        if not path.is_absolute():
            path = self.project_root / path
        return path
    
    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate Python syntax using AST
        """
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "column": e.offset
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def _find_insertion_point(
        self,
        content: str,
        location: Dict[str, Any]
    ) -> Optional[int]:
        """
        Find the line number where new code should be inserted
        """
        lines = content.split('\n')
        location_type = location.get("type", "end")
        
        if location_type == "end":
            return len(lines)
        
        elif location_type == "line":
            line_num = location.get("line", 0)
            return min(line_num, len(lines))
        
        elif location_type == "after_class":
            class_name = location.get("class_name")
            return self._find_after_class(lines, class_name)
        
        elif location_type == "before_function":
            func_name = location.get("function_name")
            return self._find_before_function(lines, func_name)
        
        return None
    
    def _find_after_class(self, lines: List[str], class_name: str) -> Optional[int]:
        """Find line after a class definition ends"""
        in_class = False
        class_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            
            if stripped.startswith(f"class {class_name}"):
                in_class = True
                class_indent = len(line) - len(stripped)
                continue
            
            if in_class:
                if stripped and not line.startswith(' ' * (class_indent + 1)):
                    return i
        
        if in_class:
            return len(lines)
        
        return None
    
    def _find_before_function(self, lines: List[str], func_name: str) -> Optional[int]:
        """Find line before a function definition"""
        for i, line in enumerate(lines):
            if line.lstrip().startswith(f"def {func_name}("):
                return i
        return None
    
    def _create_backup(self, filepath: Path, content: str) -> Path:
        """
        Create a backup of file content
        """
        import time
        timestamp = int(time.time())
        backup_path = filepath.parent / f"{filepath.name}.backup_{timestamp}"
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return backup_path
    
    def _record_modification(self, modification: Dict[str, Any]):
        """Record modification in history"""
        import time
        modification["timestamp"] = time.time()
        self.modification_history.append(modification)
        
        if len(self.modification_history) > 100:
            self.modification_history = self.modification_history[-100:]
    
    def get_modification_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent modification history
        """
        return self.modification_history[-limit:]
    
    async def add_import(
        self,
        filepath: str,
        import_statement: str,
        position: str = "top"
    ) -> Dict[str, Any]:
        """
        Add an import statement to a Python file
        """
        try:
            full_path = self._resolve_path(filepath) # Use resolved path
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            backup_path = self._create_backup(full_path, content)
            
            insert_line = self._find_import_position(lines, position)
            
            if any(import_statement in line for line in lines):
                return {
                    "status": "success",
                    "message": f"Import already exists: {import_statement}",
                    "backup_path": str(backup_path)
                }
            
            lines.insert(insert_line, import_statement)
            new_content = '\n'.join(lines)
            
            syntax_check = self._validate_python_syntax(new_content) # Use internal method
            if not syntax_check["valid"]:
                return {
                    "status": "error",
                    "message": "Invalid syntax after adding import",
                    "backup_path": str(backup_path)
                }
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self._record_modification({
                "action": "add_import",
                "filepath": str(filepath),
                "import": import_statement,
                "line": insert_line
            })
            
            logger.info(f"Added import to {filepath} at line {insert_line}")
            
            return {
                "status": "success",
                "message": f"Import added: {import_statement}",
                "backup_path": str(backup_path),
                "line_number": insert_line
            }
            
        except Exception as e:
            logger.error(f"Error adding import: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def insert_code(
        self,
        filepath: str,
        code: str,
        line_number: int,
        indent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Insert code at a specific line number
        """
        try:
            full_path = self._resolve_path(filepath) # Use resolved path
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            backup_path = self._create_backup(full_path, content)
            
            if line_number < 1 or line_number > len(lines) + 1:
                return {
                    "status": "error",
                    "message": f"Invalid line number: {line_number} (file has {len(lines)} lines)"
                }
            
            if indent is None and line_number > 1:
                prev_line = lines[line_number - 2] if line_number > 1 else ""
                indent = len(prev_line) - len(prev_line.lstrip())
            elif indent is None:
                indent = 0
            
            code_lines = code.split('\n')
            indented_code = [' ' * indent + line if line.strip() else line for line in code_lines]
            
            insert_index = line_number - 1
            for i, code_line in enumerate(indented_code):
                lines.insert(insert_index + i, code_line)
            
            new_content = '\n'.join(lines)
            
            syntax_check = self._validate_python_syntax(new_content) # Use internal method
            if not syntax_check["valid"]:
                return {
                    "status": "error",
                    "message": "Invalid syntax after inserting code",
                    "backup_path": str(backup_path)
                }
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self._record_modification({
                "action": "insert_code",
                "filepath": str(filepath),
                "line_number": line_number,
                "lines_added": len(indented_code)
            })
            
            logger.info(f"Inserted {len(indented_code)} lines at {filepath}:{line_number}")
            
            return {
                "status": "success",
                "message": f"Inserted {len(indented_code)} lines at line {line_number}",
                "backup_path": str(backup_path),
                "lines_added": len(indented_code)
            }
            
        except Exception as e:
            logger.error(f"Error inserting code: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _find_import_position(self, lines: List[str], position: str = "top") -> int:
        """
        Find the best position to insert an import statement
        """
        i = 0
        
        if lines and lines[0].startswith('#!'):
            i = 1
        
        if i < len(lines) and 'coding' in lines[i]:
            i += 1
        
        if i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = '"""' if stripped.startswith('"""') else "'''"
                if not stripped.endswith(quote) or len(stripped) <= 3:
                    i += 1
                    while i < len(lines):
                        if quote in lines[i]:
                            i += 1
                            break
                        i += 1
                else:
                    i += 1
        
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        if position == "top":
            return i
        else:
            last_import = i
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    last_import = i + 1
                    i += 1
                elif stripped and not stripped.startswith('#'):
                    break
                else:
                    i += 1
            
            return last_import