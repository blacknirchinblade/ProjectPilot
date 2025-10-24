"""
Setup Automation Tool - Automates project setup tasks

This tool handles:
- Virtual environment creation
- Dependency installation
- Project structure creation
- Configuration file generation
- Git initialization

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from loguru import logger


class SetupAutomationTool:
    """
    Automates project setup tasks including venv creation, dependency installation,
    and project structure generation.
    """
    
    def __init__(self, project_root: Path):
        """
        Initialize the setup automation tool.
        
        Args:
            project_root: Root directory of the project to set up
        """
        self.project_root = Path(project_root)
        self.platform = platform.system()
        self.venv_path = self.project_root / "venv"
        
        logger.info(f"Initialized SetupAutomationTool for: {self.project_root}")
        logger.info(f"Platform: {self.platform}")
    
    def create_virtual_environment(self, python_version: Optional[str] = None) -> Tuple[bool, str]:
        """
        Create a virtual environment for the project.
        
        Args:
            python_version: Specific Python version (e.g., "3.11", "3.10")
                          If None, uses current Python version
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Creating virtual environment at: {self.venv_path}")
            
            # Check if venv already exists
            if self.venv_path.exists():
                logger.warning(f"Virtual environment already exists at {self.venv_path}")
                return False, f"Virtual environment already exists at {self.venv_path}"
            
            # Determine Python executable
            if python_version:
                python_cmd = self._find_python_version(python_version)
                if not python_cmd:
                    return False, f"Python {python_version} not found"
            else:
                python_cmd = sys.executable
            
            logger.info(f"Using Python: {python_cmd}")
            
            # Create virtual environment
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(self.venv_path)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                error_msg = f"Failed to create venv: {result.stderr}"
                logger.error(error_msg)
                return False, error_msg
            
            logger.success(f"✓ Virtual environment created successfully")
            
            # Verify activation script exists
            activate_script = self._get_activate_script()
            if not activate_script.exists():
                return False, f"Activation script not found at {activate_script}"
            
            return True, f"Virtual environment created at {self.venv_path}"
            
        except subprocess.TimeoutExpired:
            return False, "Virtual environment creation timed out (>60s)"
        except Exception as e:
            logger.exception("Error creating virtual environment")
            return False, f"Error: {str(e)}"
    
    def install_dependencies(
        self,
        requirements_file: Optional[Path] = None,
        packages: Optional[List[str]] = None,
        upgrade_pip: bool = True
    ) -> Tuple[bool, str]:
        """
        Install project dependencies in the virtual environment.
        
        Args:
            requirements_file: Path to requirements.txt (relative or absolute)
            packages: List of package names to install directly
            upgrade_pip: Whether to upgrade pip first
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.venv_path.exists():
                return False, "Virtual environment not found. Create it first."
            
            pip_cmd = self._get_pip_command()
            installed = []
            
            # Upgrade pip
            if upgrade_pip:
                logger.info("Upgrading pip...")
                result = subprocess.run(
                    [*pip_cmd, "install", "--upgrade", "pip"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    logger.success("✓ pip upgraded successfully")
                else:
                    logger.warning(f"pip upgrade warning: {result.stderr}")
            
            # Install from requirements.txt
            if requirements_file:
                req_path = self.project_root / requirements_file if not requirements_file.is_absolute() else requirements_file
                
                if not req_path.exists():
                    return False, f"Requirements file not found: {req_path}"
                
                logger.info(f"Installing dependencies from {req_path.name}...")
                result = subprocess.run(
                    [*pip_cmd, "install", "-r", str(req_path)],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes for large dependency lists
                )
                
                if result.returncode != 0:
                    error_msg = f"Failed to install requirements: {result.stderr}"
                    logger.error(error_msg)
                    return False, error_msg
                
                logger.success(f"✓ Installed dependencies from {req_path.name}")
                installed.append(f"requirements from {req_path.name}")
            
            # Install individual packages
            if packages:
                for package in packages:
                    logger.info(f"Installing {package}...")
                    result = subprocess.run(
                        [*pip_cmd, "install", package],
                        cwd=str(self.project_root),
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    if result.returncode == 0:
                        logger.success(f"✓ Installed {package}")
                        installed.append(package)
                    else:
                        logger.error(f"Failed to install {package}: {result.stderr}")
                        return False, f"Failed to install {package}"
            
            if not installed:
                return False, "No dependencies specified to install"
            
            return True, f"Installed: {', '.join(installed)}"
            
        except subprocess.TimeoutExpired:
            return False, "Dependency installation timed out"
        except Exception as e:
            logger.exception("Error installing dependencies")
            return False, f"Error: {str(e)}"
    
    def create_project_structure(
        self,
        structure: Dict[str, any],
        create_init_files: bool = True
    ) -> Tuple[bool, str]:
        """
        Create project directory structure.
        
        Args:
            structure: Dictionary describing the directory structure
                      Example: {
                          "src": {
                              "models": {},
                              "utils": {},
                              "main.py": "# Main file content"
                          },
                          "tests": {},
                          "README.md": "# Project README"
                      }
            create_init_files: Whether to create __init__.py in Python packages
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            created_dirs = []
            created_files = []
            
            def create_structure_recursive(base_path: Path, struct: Dict):
                for name, content in struct.items():
                    path = base_path / name
                    
                    if isinstance(content, dict):
                        # Create directory
                        path.mkdir(parents=True, exist_ok=True)
                        created_dirs.append(path.relative_to(self.project_root))
                        logger.debug(f"Created directory: {path.relative_to(self.project_root)}")
                        
                        # Create __init__.py for Python packages
                        if create_init_files and not any(p.name == '__init__.py' for p in path.iterdir() if p.is_file()):
                            init_file = path / "__init__.py"
                            init_file.write_text("", encoding='utf-8')
                            created_files.append(init_file.relative_to(self.project_root))
                            logger.debug(f"Created __init__.py: {init_file.relative_to(self.project_root)}")
                        
                        # Recurse into subdirectories
                        create_structure_recursive(path, content)
                    
                    elif isinstance(content, str):
                        # Create file with content
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content, encoding='utf-8')
                        created_files.append(path.relative_to(self.project_root))
                        logger.debug(f"Created file: {path.relative_to(self.project_root)}")
                    
                    elif content is None:
                        # Create empty file
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                        created_files.append(path.relative_to(self.project_root))
                        logger.debug(f"Created empty file: {path.relative_to(self.project_root)}")
            
            create_structure_recursive(self.project_root, structure)
            
            logger.success(f"✓ Created {len(created_dirs)} directories and {len(created_files)} files")
            
            return True, f"Created {len(created_dirs)} directories and {len(created_files)} files"
            
        except Exception as e:
            logger.exception("Error creating project structure")
            return False, f"Error: {str(e)}"
    
    def generate_config_files(
        self,
        include_gitignore: bool = True,
        include_readme: bool = True,
        include_setup_py: bool = False,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Generate common configuration files.
        
        Args:
            include_gitignore: Generate .gitignore
            include_readme: Generate README.md
            include_setup_py: Generate setup.py
            project_name: Name of the project
            project_description: Short description
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            generated = []
            
            # Generate .gitignore
            if include_gitignore:
                gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
output/
data/memory/
"""
                gitignore_path = self.project_root / ".gitignore"
                gitignore_path.write_text(gitignore_content, encoding='utf-8')
                generated.append(".gitignore")
                logger.debug("Created .gitignore")
            
            # Generate README.md
            if include_readme:
                readme_content = f"""# {project_name or 'Project'}

{project_description or 'A Python project generated by AutoCoder'}

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
- Windows: `venv\\Scripts\\activate`
- Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Add usage instructions here]

## Testing

```bash
pytest tests/
```

## License

[Add license information]
"""
                readme_path = self.project_root / "README.md"
                readme_path.write_text(readme_content, encoding='utf-8')
                generated.append("README.md")
                logger.debug("Created README.md")
            
            # Generate setup.py
            if include_setup_py and project_name:
                setup_content = f"""from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="0.1.0",
    description="{project_description or 'A Python project'}",
    author="AutoCoder",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    python_requires=">=3.8",
    install_requires=[
        # Add your dependencies here
    ],
    extras_require={{
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    }},
)
"""
                setup_path = self.project_root / "setup.py"
                setup_path.write_text(setup_content, encoding='utf-8')
                generated.append("setup.py")
                logger.debug("Created setup.py")
            
            if not generated:
                return False, "No config files specified to generate"
            
            logger.success(f"✓ Generated {len(generated)} config files")
            return True, f"Generated: {', '.join(generated)}"
            
        except Exception as e:
            logger.exception("Error generating config files")
            return False, f"Error: {str(e)}"
    
    def initialize_git(self, initial_commit: bool = True) -> Tuple[bool, str]:
        """
        Initialize git repository.
        
        Args:
            initial_commit: Whether to make an initial commit
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            git_dir = self.project_root / ".git"
            
            if git_dir.exists():
                return False, "Git repository already initialized"
            
            # Initialize git
            logger.info("Initializing git repository...")
            result = subprocess.run(
                ["git", "init"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False, f"Failed to initialize git: {result.stderr}"
            
            logger.success("✓ Git repository initialized")
            
            # Make initial commit
            if initial_commit:
                # Add all files
                subprocess.run(
                    ["git", "add", "."],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Commit
                result = subprocess.run(
                    ["git", "commit", "-m", "Initial commit - AutoCoder generated project"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    logger.success("✓ Initial commit created")
                    return True, "Git initialized with initial commit"
                else:
                    logger.warning("Git initialized but initial commit failed")
                    return True, "Git initialized (no initial commit)"
            
            return True, "Git repository initialized"
            
        except FileNotFoundError:
            return False, "Git not found. Please install Git."
        except subprocess.TimeoutExpired:
            return False, "Git initialization timed out"
        except Exception as e:
            logger.exception("Error initializing git")
            return False, f"Error: {str(e)}"
    
    def get_activation_command(self) -> str:
        """
        Get the command to activate the virtual environment.
        
        Returns:
            Activation command string
        """
        if self.platform == "Windows":
            return f"{self.venv_path}\\Scripts\\activate"
        else:
            return f"source {self.venv_path}/bin/activate"
    
    def get_python_executable(self) -> Path:
        """
        Get the path to the Python executable in the venv.
        
        Returns:
            Path to Python executable
        """
        if self.platform == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    # Helper methods
    
    def _get_activate_script(self) -> Path:
        """Get path to activation script."""
        if self.platform == "Windows":
            return self.venv_path / "Scripts" / "activate.bat"
        else:
            return self.venv_path / "bin" / "activate"
    
    def _get_pip_command(self) -> List[str]:
        """Get pip command for the venv."""
        if self.platform == "Windows":
            return [str(self.venv_path / "Scripts" / "python.exe"), "-m", "pip"]
        else:
            return [str(self.venv_path / "bin" / "python"), "-m", "pip"]
    
    def _find_python_version(self, version: str) -> Optional[str]:
        """
        Find Python executable for specific version.
        
        Args:
            version: Python version (e.g., "3.11", "3.10")
        
        Returns:
            Path to Python executable or None if not found
        """
        possible_names = [
            f"python{version}",
            f"python{version.replace('.', '')}",
            "python3",
            "python"
        ]
        
        for name in possible_names:
            try:
                result = subprocess.run(
                    [name, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and version in result.stdout:
                    return name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None


# CODE_GENERATION_COMPLETE
