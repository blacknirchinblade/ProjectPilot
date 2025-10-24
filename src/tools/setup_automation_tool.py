"""
Setup Automation Tool - Main Orchestrator.
Coordinates all setup phases: structure, environment, dependencies, configuration, git.
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from loguru import logger
import subprocess
from src.tools.data_structures import (
    SetupConfig,
    SetupResult,
    SetupPhase,
    EnvironmentType,
    ProjectType,
    ConfigFileType,
    Dependency,
    ProjectInfo,
    StructureResult,
    EnvResult,
    InstallResult,
    ConfigResult,
    GitResult,
    ValidationResult
)
from src.tools.project_structure import ProjectStructure
from src.tools.environment_manager import EnvironmentManager
from src.tools.dependency_manager import DependencyManager
from src.tools.config_generator import ConfigGenerator
from src.tools.git_manager import GitManager


class SetupAutomationTool:
    """
    Main setup automation tool that orchestrates all setup phases.
    
    Phases:
    1. STRUCTURE - Create project directory structure
    2. ENVIRONMENT - Create virtual environment
    3. DEPENDENCIES - Install dependencies
    4. CONFIGURATION - Generate configuration files
    5. GIT - Initialize git repository
    6. VALIDATION - Validate setup
    """
    
    def __init__(self, config: SetupConfig):
        """
        Initialize setup automation tool.
        
        Args:
            config: Setup configuration
        """
        self.config = config
        self.project_path = Path(config.project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.structure_manager = ProjectStructure(self.project_path)
        self.env_manager = EnvironmentManager(self.project_path)
        self.git_manager = GitManager(self.project_path)
        
        # These will be initialized based on environment type
        self.dep_manager: Optional[DependencyManager] = None
        self.config_generator: Optional[ConfigGenerator] = None
    
    def setup_project(self) -> SetupResult:
        """
        Execute complete project setup.
        
        Returns:
            SetupResult with overall status
        """
        start_time = datetime.now()
        phases_completed: List[SetupPhase] = []
        phases_failed: List[SetupPhase] = []
        
        logger.info(f"Starting setup for project: {self.config.project_name}")
        logger.info(f"Project path: {self.project_path}")
        
        # Phase 1: Create project structure
        structure_result = None
        env_result = None
        install_result = None
        config_result = None
        git_result = None
        validation_result = None

        try:
            # Phase 1: Create project structure
            if not self.config.skip_structure:
                logger.info("=" * 60)
                logger.info("PHASE 1: Creating Project Structure")
                logger.info("=" * 60)
                structure_result = self._create_structure()
                if structure_result.success:
                    phases_completed.append(SetupPhase.STRUCTURE)
                    logger.success("✓ Structure creation complete")
                else:
                    phases_failed.append(SetupPhase.STRUCTURE)
                    logger.error("✗ Structure creation failed")
                    if not self.config.force:
                        raise Exception("Structure creation failed. Aborting setup.")
        
        # Phase 2: Create virtual environment
        
            if not self.config.skip_venv:
                logger.info("=" * 60)
                logger.info("PHASE 2: Creating Virtual Environment")
                logger.info("=" * 60)
                env_result = self._create_environment()
                if env_result.success:
                    phases_completed.append(SetupPhase.ENVIRONMENT)
                    logger.success("✓ Environment creation complete")
                else:
                    phases_failed.append(SetupPhase.ENVIRONMENT)
                    logger.error("✗ Environment creation failed")
                    if not self.config.force:
                        raise Exception("Environment creation failed. Aborting setup.")
        
        # Phase 3: Install dependencies

            if not self.config.skip_dependencies and self.config.dependencies or self.config.dev_dependencies:
                logger.info("=" * 60)
                logger.info("PHASE 3: Installing Dependencies")
                logger.info("=" * 60)
                install_result = self._install_dependencies(env_result)
                if install_result.success:
                    phases_completed.append(SetupPhase.DEPENDENCIES)
                    logger.success("✓ Dependency installation complete")
                else:
                    phases_failed.append(SetupPhase.DEPENDENCIES)
                    logger.error("✗ Dependency installation failed")
                    if not self.config.force:
                        raise Exception("Dependency installation failed. Aborting setup.")  
        
        # Phase 4: Generate configuration files
        
            if not self.config.skip_config and self.config.config_files:
                logger.info("=" * 60)
                logger.info("PHASE 4: Generating Configuration Files")
                logger.info("=" * 60)
                config_result = self._generate_configs()
                if config_result.success:
                    phases_completed.append(SetupPhase.CONFIGURATION)
                    logger.success("✓ Configuration generation complete")
                else:
                    phases_failed.append(SetupPhase.CONFIGURATION)
                    logger.error("✗ Configuration generation failed")
                    if not self.config.force:
                        raise Exception("Configuration generation failed. Aborting setup.")
        
            # Phase 5: Initialize git

            if self.config.init_git:
                logger.info("=" * 60)
                logger.info("PHASE 5: Initializing Git Repository")
                logger.info("=" * 60)
                git_result = self._initialize_git()
                if git_result.success:
                    phases_completed.append(SetupPhase.GIT)
                    logger.success("✓ Git initialization complete")
                else:
                    phases_failed.append(SetupPhase.GIT)
                    logger.error("✗ Git initialization failed")
                    if not self.config.force:
                        raise Exception("Git initialization failed. Aborting setup.")
        
            # Phase 6: Validation
            logger.info("=" * 60)
            logger.info("PHASE 6: Validating Setup")
            logger.info("=" * 60)
            validation_result = self._validate_setup()
            if validation_result.is_valid:
                phases_completed.append(SetupPhase.VALIDATION)
                logger.success("✓ Setup validation complete")
            else:
                phases_failed.append(SetupPhase.VALIDATION)
                logger.error("✗ Setup validation failed")

        except Exception as e:
            logger.error(f"Setup aborted during phase: {e}")
            # The error message is already in the last failed result, just build the final report
            
            # Build final result
        return self._build_result(
            start_time, phases_completed, phases_failed,
            structure_result=structure_result,
            env_result=env_result,
            install_result=install_result,
            config_result=config_result,
            git_result=git_result,
            validation_result=validation_result
        )
    
    def _create_structure(self) -> StructureResult:
        """Create project directory structure."""
        try:
            return self.structure_manager.create_structure(
                structure_type=self.config.project_type or ProjectType.STANDARD,
                project_name=self.config.project_name,
                version=self.config.version,
                description=self.config.description
            )
        except Exception as e:
            logger.error(f"Exception creating structure: {e}")
            from src.tools.data_structures import StructureResult
            return StructureResult(
                success=False,
                created_directories=[],
                created_files=[],
                structure_type=self.config.project_type or "standard",
                message=f"Failed: {e}",
                error=str(e),
                duration=0.0
            )
    
    def _create_environment(self) -> EnvResult:
        """Create virtual environment."""
        try:
            # Use the venv name from config, default to "venv"
            venv_name = getattr(self.config, 'venv_name', 'venv')
            
            return self.env_manager.create_environment(
                env_type=self.config.env_type,
                env_name=venv_name,  # Use consistent venv name
                python_version=self.config.python_version
            )
        except Exception as e:
            logger.error(f"Exception creating environment: {e}")
            from src.tools.data_structures import EnvResult
            return EnvResult(
                success=False,
                env_info=None,
                message=f"Failed: {e}",
                error=str(e),
                duration=0.0,
                commands_executed=[]
            )
    def _install_dependencies(self, env_result: Optional[EnvResult]) -> InstallResult:
        """Install project dependencies using venv + pip."""
        try:
            import platform
            import subprocess
            
            if env_result and env_result.env_info:
                # Use the actual environment path that was created
                venv_path = env_result.env_info.path
                logger.info(f"Using created environment path: {venv_path}") 
            else:
                # Use the venv name from config, default to "venv"
                venv_name = getattr(self.config, 'venv_name', 'venv')
                venv_path = self.project_path / venv_name
                logger.info(f"Using default environment path: {venv_path}")

            logger.info(f"Checking virtual environment at: {venv_path}")
            logger.info(f"Environment path exists: {venv_path.exists()}")

            
            if not venv_path.exists():
                logger.error(f"Virtual environment not found at: {venv_path}")
                project_contents = list(self.project_path.iterdir())
                logger.info(f"Project directory contents: {[str(p) for p in project_contents]}")
                
                return InstallResult(
                    success=False,
                    installed=[],
                    failed=self.config.dependencies + self.config.dev_dependencies,
                    skipped=[],
                    warnings=["Virtual environment not found"],
                    message="Virtual environment was not created properly",
                    duration=0.0
                )
            if env_result and env_result.env_info and env_result.env_info.type == EnvironmentType.CONDA:
                # Conda Environment
                # Conda environment
                if platform.system() == "Windows":
                    pip_exe = venv_path / "Scripts" / "pip.exe"
                    python_exe = venv_path / "Scripts" / "python.exe"
                else:
                    pip_exe = venv_path / "bin" / "pip"
                    python_exe = venv_path / "bin" / "python"
            else:
                # Venv environment
                if platform.system() == "Windows":
                    pip_exe = venv_path / "Scripts" / "pip.exe"
                    python_exe = venv_path / "Scripts" / "python.exe"
                else:
                    pip_exe = venv_path / "bin" / "pip"
                    python_exe = venv_path / "bin" / "python"

            logger.info(f"Looking for pip at: {pip_exe}")
            logger.info(f"Looking for python at: {python_exe}")

            # Verify executables exist
            pip_exists = pip_exe.exists()
            python_exists = python_exe.exists()
            
            logger.info(f"Pip executable exists: {pip_exists}")
            logger.info(f"Python executable exists: {python_exists}")

            if not pip_exists or not python_exists:
                logger.warning("Executables not found in standard location, searching...")
                found_executables = self._find_venv_executables(venv_path)
                if found_executables:
                    pip_exe, python_exe = found_executables
                    pip_exists = pip_exe.exists() if pip_exe else False
                    python_exists = python_exe.exists() if python_exe else False
                    logger.info(f"After search - Pip exists: {pip_exists}, Python exists: {python_exists}")


            # Choose the installation method
            if pip_exists:
                pip_command = str(pip_exe)
                logger.info(f"Using direct pip executable: {pip_command}")
            elif python_exists:
                pip_command = f'"{python_exe}" -m pip'
                logger.info(f"Using python -m pip: {pip_command}")
            else:
                error_msg = "Neither pip nor python executable found in virtual environment"
                logger.error(error_msg)
                return InstallResult(
                    success=False,
                    installed=[],
                    failed=self.config.dependencies + self.config.dev_dependencies,
                    skipped=[],
                    warnings=[error_msg],
                    message=error_msg,
                    duration=0.0
                )

            # Filter out invalid dependencies
            valid_dependencies = []
            invalid_dependencies = []
            
            # Known invalid patterns (submodules that aren't packages)
            invalid_patterns = ['.nn', '.logging']  # torch.nn, rich.logging etc.
            
            for d in self.config.dependencies + self.config.dev_dependencies:
                # Check for stdlib modules
                if d.split('==')[0].lower() in {'pathlib', 'logging', 'typing', 'math'}:
                    invalid_dependencies.append(f"{d} (stdlib)")
                    continue
                    
                # Check for submodules that aren't installable packages
                if any(pattern in d for pattern in invalid_patterns):
                    invalid_dependencies.append(f"{d} (submodule)")
                    continue
                    
                # Check for empty or malformed dependencies
                if not d.strip() or d.strip() == 'fastapi':  # fastapi appears twice in your logs
                    invalid_dependencies.append(f"{d} (malformed)")
                    continue
                    
                valid_dependencies.append(d)

            logger.info(f"Valid dependencies: {valid_dependencies}")
            logger.info(f"Invalid dependencies (skipped): {invalid_dependencies}")

            if not valid_dependencies:
                logger.info("No valid packages to install after filtering")
                return InstallResult(
                    success=True,
                    installed=[],
                    failed=[],
                    skipped=invalid_dependencies,
                    warnings=["No valid packages to install after filtering invalid dependencies"],
                    message="No valid packages to install",
                    duration=0.0
                )

            # Build the installation command
            if pip_exists:
                cmd = [str(pip_exe), "install"] + valid_dependencies
            else:
                cmd = [str(python_exe), "-m", "pip", "install"] + valid_dependencies

            logger.info(f"Running command: {' '.join(cmd)}")

            # Run the installation directly
            start_time = datetime.now()
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    shell=False
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                
                if result.returncode == 0:
                    logger.success(f"Successfully installed {len(valid_dependencies)} packages")
                    
                    return InstallResult(
                        success=True,
                        installed=valid_dependencies,
                        failed=[],
                        skipped=invalid_dependencies,
                        warnings=[],
                        message=f"Successfully installed {len(valid_dependencies)} packages",
                        duration=duration
                    )
                else:
                    logger.error(f"Installation failed with return code: {result.returncode}")
                    logger.error(f"stderr: {result.stderr}")
                    
                    # Try to install packages one by one to identify which ones fail
                    individual_results = self._install_packages_individually(
                        pip_exists, pip_exe, python_exe, valid_dependencies
                    )
                    
                    return individual_results
                    
            except subprocess.TimeoutExpired:
                duration = (datetime.now() - start_time).total_seconds()
                error_msg = "Installation timed out after 10 minutes"
                logger.error(error_msg)
                return InstallResult(
                    success=False,
                    installed=[],
                    failed=valid_dependencies,
                    skipped=invalid_dependencies,
                    warnings=[error_msg],
                    message=error_msg,
                    duration=duration
                )
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                error_msg = f"Exception during installation: {e}"
                logger.error(error_msg)
                return InstallResult(
                    success=False,
                    installed=[],
                    failed=valid_dependencies,
                    skipped=invalid_dependencies,
                    warnings=[str(e)],
                    message=error_msg,
                    duration=duration
                )

        except Exception as e:
            logger.error(f"Exception in dependency installation setup: {e}")
            all_deps = self.config.dependencies + self.config.dev_dependencies
            return InstallResult(
                success=False,
                installed=[],
                failed=[d for d in all_deps],
                skipped=[],
                warnings=[str(e)],
                message=f"Failed: {e}",
                duration=0.0
            )

    def _find_venv_executables(self, venv_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Find pip and python executables in venv directory."""
        import platform
        
        if platform.system() == "Windows":
            scripts_dir = venv_path / "Scripts"
            pip_pattern = "pip*.exe"
            python_pattern = "python*.exe"
        else:
            scripts_dir = venv_path / "bin"
            pip_pattern = "pip*"
            python_pattern = "python*"
        
        if not scripts_dir.exists():
            logger.error(f"Scripts directory not found: {scripts_dir}")
            return None, None
        
        # Find pip executable
        pip_files = list(scripts_dir.glob(pip_pattern))
        pip_exe = None
        for pip_file in pip_files:
            if pip_file.name.startswith('pip') and not pip_file.name.startswith('pip-'):
                pip_exe = pip_file
                break
        
        # Find python executable  
        python_files = list(scripts_dir.glob(python_pattern))
        python_exe = None
        for python_file in python_files:
            if python_file.name.startswith('python') and not python_file.name.startswith('python-'):
                python_exe = python_file
                break
        
        logger.info(f"Found pip executables: {[str(p) for p in pip_files]}")
        logger.info(f"Found python executables: {[str(p) for p in python_files]}")
        
        return pip_exe, python_exe

    def _log_available_executables(self, venv_path: Path):
        """Log all available executables in the venv for debugging."""
        import platform
        
        if platform.system() == "Windows":
            scripts_dir = venv_path / "Scripts"
        else:
            scripts_dir = venv_path / "bin"
        
        if scripts_dir.exists():
            all_files = list(scripts_dir.iterdir())
            logger.info(f"All files in {scripts_dir}: {[str(f) for f in all_files]}")
        else:
            logger.error(f"Scripts directory does not exist: {scripts_dir}")

    def _install_packages_individually(self, pip_exists: bool, pip_exe: Path, python_exe: Path, packages: List[str]) -> InstallResult:
        """Install packages one by one to identify failing packages."""
        logger.info("Attempting individual package installation to identify failures...")
        
        installed = []
        failed = []
        warnings = []
        start_time = datetime.now()
        
        for package in packages:
            try:
                if pip_exists:
                    cmd = [str(pip_exe), "install", package]
                else:
                    cmd = [str(python_exe), "-m", "pip", "install", package]
                
                logger.info(f"Installing individually: {package}")
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per package
                    shell=False
                )
                
                if result.returncode == 0:
                    installed.append(package)
                    logger.success(f"✓ Installed: {package}")
                else:
                    failed.append(package)
                    error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                    warnings.append(f"{package}: {error_msg}")
                    logger.error(f"✗ Failed to install {package}: {error_msg}")
                    
            except subprocess.TimeoutExpired:
                failed.append(package)
                warnings.append(f"{package}: Installation timed out")
                logger.error(f"✗ {package}: Installation timed out")
            except Exception as e:
                failed.append(package)
                warnings.append(f"{package}: {str(e)}")
                logger.error(f"✗ {package}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        success = len(failed) == 0
        
        message = f"Installed {len(installed)} packages individually"
        if failed:
            message += f", {len(failed)} failed"
        
        return InstallResult(
            success=success,
            installed=installed,
            failed=failed,
            skipped=[],
            warnings=warnings,
            message=message,
            duration=duration
        )

    def _install_dependencies_alternative(self, pip_exe: Path, dependencies: List[Dependency], dev_dependencies: List[Dependency]) -> InstallResult:
        """Alternative dependency installation method using subprocess directly."""
        try:
            import subprocess
            import sys
            
            # Use the python executable from the venv
            if pip_exe.name == "pip.exe":
                python_exe = pip_exe.parent / "python.exe"
            else:
                python_exe = pip_exe.parent / "python"
            
            installed = []
            failed = []
            
            # Install all packages at once
            all_packages = [dep.to_install_string() for dep in dependencies + dev_dependencies]
            
            if all_packages:
                cmd = [str(python_exe), "-m", "pip", "install"] + all_packages
                logger.info(f"Running alternative install: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    installed = all_packages
                    logger.success("Alternative installation successful")
                else:
                    failed = all_packages
                    logger.error(f"Alternative installation failed: {result.stderr}")
            
            return InstallResult(
                success=len(failed) == 0,
                installed=installed,
                failed=failed,
                skipped=[],
                warnings=[],
                message="Alternative installation completed",
                duration=0.0
            )
            
        except Exception as e:
            logger.error(f"Alternative installation also failed: {e}")
            all_packages = [dep.to_install_string() for dep in dependencies + dev_dependencies]
            return InstallResult(
                success=False,
                installed=[],
                failed=all_packages,
                skipped=[],
                warnings=[str(e)],
                message=f"Alternative method failed: {e}",
                duration=0.0
            )


    
    def _generate_configs(self) -> ConfigResult:
        """Generate configuration files."""
        try:
            # Create ProjectInfo
            project_info = ProjectInfo(
                name=self.config.project_name,
                version=self.config.version,
                description=self.config.description,
                author=self.config.author,
                author_email=self.config.author_email,
                license=self.config.license,
                python_version=self.config.python_version or "3.10"
            )
            
            # Initialize config generator
            self.config_generator = ConfigGenerator(project_info, self.project_path)
            
            # Convert config file strings to ConfigFileType
            config_types: List[ConfigFileType] = []
            for cf in self.config.config_files:
                if isinstance(cf, ConfigFileType):
                    config_types.append(cf)
                elif isinstance(cf, str):
                    try:
                        config_types.append(ConfigFileType.from_string(cf))
                    except ValueError:
                        logger.warning(f"Unknown config file type: {cf} Skipping...")

            return self.config_generator.generate_all(
                config_types=config_types,
                dependencies=self.config.dependencies,
                dev_dependencies=self.config.dev_dependencies or []
            )
            
        except Exception as e:
            logger.error(f"Exception generating configs: {e}")
            from src.tools.data_structures import ConfigResult
            return ConfigResult(
                success=False,
                generated_files=[],
                failed_files=[str(cf) for cf in self.config.config_files],
                message=f"Failed: {e}",
                error=str(e),
                duration=0.0
            )
    
    def _initialize_git(self) -> GitResult:
        """Initialize git repository."""
        try:
            result = self.git_manager.initialize_repository(
                initial_branch=self.config.git_branch,
                initial_commit=self.config.initial_commit,
                commit_message="Initial commit: Project setup"
            )
            
            # Add remote if specified
            if result.success and self.config.git_remote:
                remote_result = self.git_manager.add_remote("origin", self.config.git_remote)
                if not remote_result.success:
                    logger.warning(f"Failed to add remote: {remote_result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Exception initializing git: {e}")
            from src.tools.data_structures import GitResult
            return GitResult(
                success=False,
                repo_path=None,
                branch=None,
                remote=None,
                commits=[],
                message=f"Failed: {e}",
                error=str(e),
                duration=0.0
            )
    
    def _validate_setup(self) -> ValidationResult:
        """Validate the setup."""
        from src.tools.data_structures import ValidationResult
        
        checks_passed = []
        checks_failed = []
        warnings = []
        
        # Check project directory exists
        if self.project_path.exists():
            checks_passed.append("Project directory exists")
        else:
            checks_failed.append("Project directory missing")
        
        # Check venv exists (if not skipped)
        if not self.config.skip_venv:
            venv_path = self.project_path / self.config.venv_name
            if venv_path.exists():
                checks_passed.append("Virtual environment exists")
            else:
                checks_failed.append("Virtual environment missing")
                warnings.append("Run setup again or create environment manually")
        
        # Check git initialized (if requested)
        if self.config.init_git:
            if (self.project_path / ".git").exists():
                checks_passed.append("Git repository initialized")
            else:
                checks_failed.append("Git repository not initialized")
        
        # Check config files (if not skipped)
        if not self.config.skip_config:
            for cf in self.config.config_files:
                file_name = cf if isinstance(cf, str) else cf.value
                if (self.project_path / file_name).exists():
                    checks_passed.append(f"Config file exists: {file_name}")
                else:
                    checks_failed.append(f"Config file missing: {file_name}")
        
        is_valid = len(checks_failed) == 0
        message = f"Validation {'passed' if is_valid else 'failed'}: {len(checks_passed)}/{len(checks_passed) + len(checks_failed)} checks"
        
        if is_valid:
            logger.success(message)
        else:
            logger.warning(message)
            for check in checks_failed:
                logger.warning(f"  - {check}")
        
        return ValidationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            message=message
        )
    
    def _build_result(
        self,
        start_time: datetime,
        phases_completed: List[SetupPhase],
        phases_failed: List[SetupPhase],
        **phase_results
    ) -> SetupResult:
        """
        Build final SetupResult.
        
        Args:
            start_time: Setup start time
            phases_completed: List of completed phase names
            phases_failed: List of failed phase names
            **phase_results: Individual phase results
            
        Returns:
            SetupResult with overall status
        """
        duration = (datetime.now() - start_time).total_seconds()
        success = len(phases_failed) == 0
        
        message = f"Setup {'completed' if success else 'failed'}: "
        message += f"{len(phases_completed)} phases completed"
        if phases_failed:
            message += f", {len(phases_failed)} phases failed"
        message += f" in {duration:.2f}s"
        
        if success:
            logger.success("=" * 60)
            logger.success(f"✓ PROJECT SETUP COMPLETE: {self.config.project_name}")
            logger.success("=" * 60)
            logger.success(message)
        else:
            logger.error("=" * 60)
            logger.error(f"✗ PROJECT SETUP INCOMPLETE: {self.config.project_name}")
            logger.error("=" * 60)
            logger.error(message)
        
        return SetupResult(
            success=success,
            project_path=str(self.project_path),
            phases_completed=phases_completed,
            phases_failed=phases_failed,
            structure_result=phase_results.get('structure_result'),
            env_result=phase_results.get('env_result'),
            install_result=phase_results.get('install_result'),
            config_result=phase_results.get('config_result'),
            git_result=phase_results.get('git_result'),
            validation_result=phase_results.get('validation_result'),
            message=message,
            duration=duration
        )
