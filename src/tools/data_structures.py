"""
Data structures for Setup Automation Tool.

Contains dataclasses and types used throughout the setup automation system.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime


class EnvironmentType(Enum):
    """Types of Python virtual environments."""
    VENV = "venv"
    CONDA = "conda"
    VIRTUALENV = "virtualenv"
    SYSTEM = "system"


class SetupPhase(Enum):
    """Phases of project setup."""
    STRUCTURE = "structure"
    ENVIRONMENT = "environment"
    DEPENDENCIES = "dependencies"
    CONFIGURATION = "configuration"
    GIT = "git"
    VALIDATION = "validation"


class ProjectType(Enum):
    """Types of project structures."""
    STANDARD = "standard"
    ML = "ml"
    WEB = "web"
    CLI = "cli"
    LIBRARY = "library"


class ConfigFileType(Enum):
    """Types of configuration files."""
    SETUP_PY = "setup.py"
    PYPROJECT_TOML = "pyproject.toml"
    SETUP_CFG = "setup.cfg"
    MANIFEST_IN = "MANIFEST.in"
    GITIGNORE = ".gitignore"
    EDITORCONFIG = ".editorconfig"
    PYTEST_INI = "pytest.ini"
    TOX_INI = "tox.ini"
    REQUIREMENTS_TXT = "requirements.txt"
    README = "README.md"
    LICENSE = "LICENSE"


# ==================== Configuration ====================


@dataclass
class SetupConfig:
    """Configuration for setup automation."""
    
    # Project metadata
    project_name: str = "myproject"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    author_email: str = ""
    license: str = "MIT"
    python_version: Optional[str] = "3.10"
    project_path: str = "."
    
    # Environment - ALWAYS use "venv" to avoid path issues
    env_type: EnvironmentType = EnvironmentType.VENV
    venv_name: str = "venv"  # Simple, consistent name
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    extras: Dict[str, List[str]] = field(default_factory=dict)
    
    # Rest of your config remains the same...
    # Structure
    project_type: Optional[ProjectType] = ProjectType.STANDARD
    
    # Git
    init_git: bool = True
    git_remote: Optional[str] = None
    initial_commit: bool = True
    git_branch: str = "main"
    
    # Configuration files
    config_files: List[str] = field(default_factory=lambda: ["pyproject.toml", ".gitignore"])
    
    # Skip flags
    skip_structure: bool = False
    skip_venv: bool = False
    skip_dependencies: bool = False
    skip_config: bool = False
    
    # Options
    dry_run: bool = False
    verbose: bool = True
    force: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_email": self.author_email,
            "license": self.license,
            "python_version": self.python_version,
            "env_type": self.env_type,
            "dependencies": self.dependencies,
            "dev_dependencies": self.dev_dependencies,
            "project_type": self.project_type,
        }


# ==================== Environment ====================


@dataclass
class EnvironmentInfo:
    """Information about a Python environment."""
    type: EnvironmentType
    path: Path
    name: str
    python_version: str
    python_executable: Path
    is_active: bool = False
    packages: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "path": str(self.path),
            "name": self.name,
            "python_version": self.python_version,
            "python_executable": str(self.python_executable),
            "is_active": self.is_active,
            "packages": self.packages,
        }


@dataclass
class EnvResult:
    """Result of environment creation/operation."""
    success: bool
    env_info: Optional[EnvironmentInfo] = None
    message: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    commands_executed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "env_info": self.env_info.to_dict() if self.env_info else None,
            "message": self.message,
            "error": self.error,
            "duration": self.duration,
        }


# ==================== Dependencies ====================


@dataclass
class Dependency:
    """A Python package dependency."""
    name: str
    version_spec: str = ""  # e.g., ">=1.0,<2.0"
    extras: List[str] = field(default_factory=list)
    markers: str = ""  # e.g., "python_version >= '3.8'"
    
    def to_requirement_string(self) -> str:
        """Convert to pip requirement string."""
        result = self.name
        if self.extras:
            result += f"[{','.join(self.extras)}]"
        if self.version_spec:
            result += self.version_spec
        if self.markers:
            result += f"; {self.markers}"
        return result
    
    @classmethod
    def from_string(cls, requirement: str) -> "Dependency":
        """Parse from pip requirement string."""
        # Simple parsing - handle: package==1.0.0, package>=1.0, etc.
        import re
        
        # Match: name[extras]version; markers
        pattern = r'^([a-zA-Z0-9_-]+)(\[[a-zA-Z0-9,_-]+\])?(.*?)(;.*)?$'
        match = re.match(pattern, requirement.strip())
        
        if not match:
            return cls(name=requirement.strip())
        
        name = match.group(1)
        extras_str = match.group(2) or ""
        version_spec = (match.group(3) or "").strip()
        markers = (match.group(4) or "").strip("; ")
        
        extras = []
        if extras_str:
            extras = [e.strip() for e in extras_str.strip("[]").split(",")]
        
        return cls(
            name=name,
            version_spec=version_spec,
            extras=extras,
            markers=markers
        )


@dataclass
class PackageInfo:
    """Information about an installed package."""
    name: str
    version: str
    location: str
    requires: List[str] = field(default_factory=list)
    required_by: List[str] = field(default_factory=list)


@dataclass
class InstallResult:
    """Result of package installation."""
    success: bool
    installed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0
    message: str = ""
    error: Optional[str] = None
    
    @property
    def total_packages(self) -> int:
        """Total number of packages processed."""
        return len(self.installed) + len(self.failed) + len(self.skipped)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "installed": self.installed,
            "failed": self.failed,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "duration": self.duration,
            "message": self.message,
        }


@dataclass
class Conflict:
    """Dependency conflict information."""
    package: str
    required_by: List[str]
    conflicting_versions: List[str]
    description: str


# ==================== Configuration Files ====================


@dataclass
class ConfigResult:
    """Result of configuration file generation."""
    success: bool
    generated_files: List[Path] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    message: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "generated_files": [str(f) for f in self.generated_files],
            "failed_files": self.failed_files,
            "message": self.message,
            "duration": self.duration,
        }


@dataclass
class ProjectInfo:
    """Project metadata for configuration generation."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    author_email: str = ""
    license: str = "MIT"
    python_version: str = "3.10"
    homepage: str = ""
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    classifiers: List[str] = field(default_factory=list)
    
    def get_classifiers(self) -> List[str]:
        """Get package classifiers."""
        if self.classifiers:
            return self.classifiers
        
        # Default classifiers
        return [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            f"License :: OSI Approved :: {self.license} License",
            "Programming Language :: Python :: 3",
            f"Programming Language :: Python :: {self.python_version}",
        ]


# ==================== Git ====================


@dataclass
class GitResult:
    """Result of Git operation."""
    success: bool
    repo_path: Optional[Path] = None
    branch: str = "main"
    remote: Optional[str] = None
    commits: List[str] = field(default_factory=list)
    message: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "branch": self.branch,
            "remote": self.remote,
            "message": self.message,
        }


# ==================== Project Structure ====================


@dataclass
class StructureResult:
    """Result of project structure creation."""
    success: bool
    created_directories: List[Path] = field(default_factory=list)
    created_files: List[Path] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    structure_type: str = "standard"
    message: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    
    @property
    def total_items(self) -> int:
        """Total items created."""
        return len(self.created_directories) + len(self.created_files)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "created_directories": [str(d) for d in self.created_directories],
            "created_files": [str(f) for f in self.created_files],
            "structure_type": self.structure_type,
            "total_items": self.total_items,
            "message": self.message,
        }


# ==================== Complete Setup ====================


@dataclass
class SetupResult:
    """Result of complete project setup."""
    success: bool
    project_path: str
    phases_completed: List[str] = field(default_factory=list)
    phases_failed: List[str] = field(default_factory=list)
    
    # Individual results
    structure_result: Optional['StructureResult'] = None
    env_result: Optional['EnvResult'] = None
    install_result: Optional['InstallResult'] = None
    config_result: Optional['ConfigResult'] = None
    git_result: Optional['GitResult'] = None
    validation_result: Optional['ValidationResult'] = None
    
    # Summary
    message: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_complete(self) -> bool:
        """Check if all phases completed."""
        return len(self.phases_failed) == 0 and len(self.phases_completed) > 0
    
    @property
    def completion_rate(self) -> float:
        """Get completion percentage."""
        total = len(self.phases_completed) + len(self.phases_failed)
        if total == 0:
            return 0.0
        return (len(self.phases_completed) / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "project_path": str(self.project_path),
            "phases_completed": [p.value for p in self.phases_completed],
            "phases_failed": [p.value for p in self.phases_failed],
            "completion_rate": self.completion_rate,
            "message": self.message,
            "duration": self.duration,
            "created_at": self.created_at.isoformat(),
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Setup Result: {'✅ SUCCESS' if self.success else '❌ FAILED'}",
            f"Project: {self.project_path}",
            f"Completion: {self.completion_rate:.0f}%",
            f"Duration: {self.duration:.1f}s",
            f"Phases Completed: {len(self.phases_completed)}",
        ]
        
        if self.phases_failed:
            lines.append(f"Phases Failed: {len(self.phases_failed)}")
        
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        
        if self.message:
            lines.append(f"Message: {self.message}")
        
        return "\n".join(lines)


# ==================== Validation ====================


@dataclass
class ValidationResult:
    """Result of setup validation."""
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    message: str = ""
    
    @property
    def total_checks(self) -> int:
        """Total number of checks."""
        return len(self.checks_passed) + len(self.checks_failed)
    
    @property
    def pass_rate(self) -> float:
        """Percentage of checks passed."""
        if self.total_checks == 0:
            return 0.0
        return (len(self.checks_passed) / self.total_checks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "pass_rate": self.pass_rate,
            "message": self.message,
        }
