"""
Project Structure Creator for Setup Automation Tool.

Creates standardized project directory structures.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from .data_structures import StructureResult, ProjectType


class ProjectStructure:
    """Create project directory structures."""
    
    # Standard Python project structure
    STANDARD_STRUCTURE = {
        "src": {
            "{project_name}": {
                "__init__.py": '"""Main package."""\n\n__version__ = "{version}"\n',
                "main.py": "# Main module\n\ndef main():\n    print('Hello from {project_name}!')\n\nif __name__ == '__main__':\n    main()\n",
            }
        },
        "tests": {
            "__init__.py": "",
            "test_sample.py": "# Sample test\nimport pytest\n\ndef test_sample():\n    assert True\n",
        },
        "docs": {
            "index.md": "# {project_name} Documentation\n\n## Overview\n\n{description}\n",
        },
    }
    
    # Machine Learning project structure
    ML_STRUCTURE = {
        "src": {
            "{project_name}": {
                "__init__.py": '"""ML project package."""\n\n__version__ = "{version}"\n',
                "models": {
                    "__init__.py": "# Model definitions\n",
                    "model.py": "# Define your models here\n",
                },
                "data": {
                    "__init__.py": "# Data loading and preprocessing\n",
                    "dataset.py": "# Dataset classes\n",
                    "preprocessing.py": "# Data preprocessing functions\n",
                },
                "training": {
                    "__init__.py": "# Training pipeline\n",
                    "train.py": "# Training script\n",
                    "trainer.py": "# Trainer class\n",
                },
                "evaluation": {
                    "__init__.py": "# Model evaluation\n",
                    "metrics.py": "# Evaluation metrics\n",
                    "evaluate.py": "# Evaluation script\n",
                },
                "utils": {
                    "__init__.py": "# Utility functions\n",
                    "helpers.py": "# Helper functions\n",
                },
            }
        },
        "notebooks": {
            ".gitkeep": "",
        },
        "data": {
            "raw": {
                ".gitkeep": "",
            },
            "processed": {
                ".gitkeep": "",
            },
        },
        "models": {
            "saved": {
                ".gitkeep": "",
            },
            "checkpoints": {
                ".gitkeep": "",
            },
        },
        "tests": {
            "__init__.py": "",
            "test_model.py": "# Model tests\nimport pytest\n",
            "test_data.py": "# Data tests\nimport pytest\n",
        },
        "config": {
            ".gitkeep": "",
        },
        "docs": {
            "index.md": "# {project_name}\n\nMachine Learning Project\n",
        },
    }
    
    # Web application structure
    WEB_STRUCTURE = {
        "src": {
            "{project_name}": {
                "__init__.py": '"""Web application package."""\n\n__version__ = "{version}"\n',
                "app.py": "# Main application\n",
                "routes": {
                    "__init__.py": "# Route definitions\n",
                },
                "models": {
                    "__init__.py": "# Database models\n",
                },
                "templates": {},
                "static": {
                    "css": {},
                    "js": {},
                    "images": {},
                },
                "utils": {
                    "__init__.py": "# Utilities\n",
                },
            }
        },
        "tests": {
            "__init__.py": "",
            "test_routes.py": "# Route tests\nimport pytest\n",
        },
        "config": {
            "development.py": "# Development config\n",
            "production.py": "# Production config\n",
        },
        "docs": {
            "api.md": "# API Documentation\n",
        },
    }
    
    # CLI tool structure
    CLI_STRUCTURE = {
        "src": {
            "{project_name}": {
                "__init__.py": '"""CLI tool package."""\n\n__version__ = "{version}"\n',
                "cli.py": "# CLI entry point\nimport click\n\n@click.command()\ndef main():\n    click.echo('Hello from {project_name}!')\n",
                "commands": {
                    "__init__.py": "# Command modules\n",
                },
                "utils": {
                    "__init__.py": "# Utilities\n",
                },
            }
        },
        "tests": {
            "__init__.py": "",
            "test_cli.py": "# CLI tests\nimport pytest\n",
        },
        "docs": {
            "usage.md": "# Usage Guide\n",
        },
    }
    
    # Library structure
    LIBRARY_STRUCTURE = {
        "src": {
            "{project_name}": {
                "__init__.py": '"""Library package."""\n\n__version__ = "{version}"\n\nfrom .core import *\n',
                "core.py": "# Core functionality\n",
                "utils.py": "# Utility functions\n",
            }
        },
        "tests": {
            "__init__.py": "",
            "test_core.py": "# Core tests\nimport pytest\n",
            "test_utils.py": "# Utility tests\nimport pytest\n",
        },
        "examples": {
            "basic_usage.py": "# Basic usage example\n",
        },
        "docs": {
            "index.md": "# {project_name}\n\nLibrary Documentation\n",
            "api.md": "# API Reference\n",
        },
    }
    
    def __init__(self, project_path: Path):
        """
        Initialize project structure creator.
        
        Args:
            project_path: Root path for the project
        """
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
    
    def get_structure_template(self, structure_type: str) -> Dict:
        """
        Get structure template by type.
        
        Args:
            structure_type: Type of structure (standard, ml, web, cli, library)
            
        Returns:
            Dictionary defining the structure
        """
        # Convert enum to value if needed
        if isinstance(structure_type, ProjectType):
            structure_type = structure_type.value
        
        structures = {
            "standard": self.STANDARD_STRUCTURE,
            ProjectType.STANDARD.value: self.STANDARD_STRUCTURE,
            "ml": self.ML_STRUCTURE,
            ProjectType.ML.value: self.ML_STRUCTURE,
            "web": self.WEB_STRUCTURE,
            ProjectType.WEB.value: self.WEB_STRUCTURE,
            "cli": self.CLI_STRUCTURE,
            ProjectType.CLI.value: self.CLI_STRUCTURE,
            "library": self.LIBRARY_STRUCTURE,
            ProjectType.LIBRARY.value: self.LIBRARY_STRUCTURE,
        }
        
        return structures.get(structure_type, self.STANDARD_STRUCTURE)
    
    def create_structure(
        self,
        structure_type: str = "standard",
        project_name: str = "myproject",
        version: str = "0.1.0",
        description: str = "",
        create_init_files: bool = True
    ) -> StructureResult:
        """
        Create project directory structure.
        
        Args:
            structure_type: Type of structure to create
            project_name: Name of the project
            version: Project version
            description: Project description
            create_init_files: Whether to create __init__.py files
            
        Returns:
            StructureResult with creation status
        """
        start_time = datetime.now()
        created_directories = []
        created_files = []
        failed_items = []
        
        logger.info(f"Creating {structure_type} project structure at {self.project_path}")
        
        try:
            # Get template
            template = self.get_structure_template(structure_type)
            
            # Create directory tree
            dirs, files = self._create_directory_tree(
                template,
                self.project_path,
                {
                    "project_name": project_name,
                    "version": version,
                    "description": description or f"{project_name} project"
                }
            )
            
            created_directories.extend(dirs)
            created_files.extend(files)
            
            # Create additional __init__.py files if requested
            if create_init_files:
                init_files = self._create_init_files(created_directories)
                created_files.extend(init_files)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            message = f"Created {len(created_directories)} directories and {len(created_files)} files"
            logger.success(f"{message} in {duration:.2f}s")
            
            return StructureResult(
                success=True,
                created_directories=created_directories,
                created_files=created_files,
                structure_type=structure_type,
                message=message,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to create structure: {str(e)}"
            logger.error(error_msg)
            
            return StructureResult(
                success=False,
                created_directories=created_directories,
                created_files=created_files,
                failed_items=failed_items,
                structure_type=structure_type,
                message=error_msg,
                error=str(e),
                duration=duration
            )
    
    def _create_directory_tree(
        self,
        tree: Dict,
        base_path: Path,
        placeholders: Dict[str, str]
    ) -> tuple[List[Path], List[Path]]:
        """
        Recursively create directory tree from template.
        
        Args:
            tree: Dictionary defining the tree structure
            base_path: Base path to create tree in
            placeholders: Placeholder values for substitution
            
        Returns:
            Tuple of (created_directories, created_files)
        """
        directories = []
        files = []
        
        for name, content in tree.items():
            # Apply placeholders
            name = name.format(**placeholders)
            current_path = base_path / name
            
            if isinstance(content, dict):
                # It's a directory
                current_path.mkdir(parents=True, exist_ok=True)
                directories.append(current_path)
                logger.debug(f"Created directory: {current_path}")
                
                # Recursively create subdirectories
                sub_dirs, sub_files = self._create_directory_tree(
                    content,
                    current_path,
                    placeholders
                )
                directories.extend(sub_dirs)
                files.extend(sub_files)
                
            else:
                # It's a file
                file_content = content.format(**placeholders) if content else ""
                current_path.write_text(file_content, encoding="utf-8")
                files.append(current_path)
                logger.debug(f"Created file: {current_path}")
        
        return directories, files
    
    def _create_init_files(self, directories: List[Path]) -> List[Path]:
        """
        Create __init__.py files in Python package directories.
        
        Args:
            directories: List of directories
            
        Returns:
            List of created __init__.py files
        """
        init_files = []
        
        for directory in directories:
            # Only create in src/ subdirectories and tests/
            if "src" in directory.parts or "tests" in directory.parts:
                # Skip if already has __init__.py
                init_file = directory / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("", encoding="utf-8")
                    init_files.append(init_file)
                    logger.debug(f"Created __init__.py in {directory}")
        
        return init_files
    
    def add_directory(self, relative_path: str, create_init: bool = True) -> Path:
        """
        Add a new directory to the project structure.
        
        Args:
            relative_path: Relative path from project root
            create_init: Whether to create __init__.py
            
        Returns:
            Path to created directory
        """
        dir_path = self.project_path / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if create_init:
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")
                logger.debug(f"Created __init__.py in {dir_path}")
        
        logger.info(f"Added directory: {dir_path}")
        return dir_path
    
    def add_file(
        self,
        relative_path: str,
        content: str = "",
        overwrite: bool = False
    ) -> Path:
        """
        Add a new file to the project structure.
        
        Args:
            relative_path: Relative path from project root
            content: File content
            overwrite: Whether to overwrite if exists
            
        Returns:
            Path to created file
        """
        file_path = self.project_path / relative_path
        
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists() and not overwrite:
            logger.warning(f"File already exists: {file_path}")
            return file_path
        
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Added file: {file_path}")
        return file_path
    
    def get_structure_summary(self) -> Dict[str, Any]:
        """
        Get summary of current project structure.
        
        Returns:
            Dictionary with structure information
        """
        if not self.project_path.exists():
            return {"exists": False}
        
        def count_items(path: Path) -> Dict[str, int]:
            """Recursively count files and directories."""
            files = 0
            dirs = 0
            
            for item in path.iterdir():
                if item.is_dir():
                    dirs += 1
                    sub_counts = count_items(item)
                    files += sub_counts["files"]
                    dirs += sub_counts["dirs"]
                else:
                    files += 1
            
            return {"files": files, "dirs": dirs}
        
        counts = count_items(self.project_path)
        
        return {
            "exists": True,
            "path": str(self.project_path),
            "total_files": counts["files"],
            "total_directories": counts["dirs"],
            "has_src": (self.project_path / "src").exists(),
            "has_tests": (self.project_path / "tests").exists(),
            "has_docs": (self.project_path / "docs").exists(),
            "src_exists": (self.project_path / "src").exists(),
            "tests_exists": (self.project_path / "tests").exists(),
            "docs_exists": (self.project_path / "docs").exists(),
        }
    
    def validate_structure(self, required_items: Optional[List[str]] = None) -> bool:
        """
        Validate that required structure items exist.
        
        Args:
            required_items: List of required paths (relative to project root)
            
        Returns:
            True if all required items exist
        """
        if not required_items:
            # Default validation: check for basic structure
            required_items = ["src", "tests", "README.md"]
        
        for item in required_items:
            item_path = self.project_path / item
            if not item_path.exists():
                logger.warning(f"Missing required item: {item}")
                return False
        
        return True
