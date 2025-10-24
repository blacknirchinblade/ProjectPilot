"""
Dependency Manager for Setup Automation Tool.
Manages package installation via pip and conda.
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from loguru import logger

from src.tools.data_structures import (
    Dependency,
    InstallResult,
    EnvironmentType,
    PackageInfo
)


class BaseDependencyManager:
    """Base class for dependency managers."""
    
    def __init__(self, project_path: Path):
        """
        Initialize dependency manager.
        
        Args:
            project_path: Path to the project root
        """
        self.project_path = Path(project_path)
    
    def _run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 600
    ) -> Tuple[int, str, str]:
        """
        Run a shell command.
        
        Args:
            command: Command and arguments as list
            cwd: Working directory
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            # Convert command to string if it's a list and shell=True
            if isinstance(command, list):
                command_str = " ".join(str(c) for c in command)
            else:
                command_str = str(command)
                
            result = subprocess.run(
                command_str,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return 1, "", str(e)


class PipManager(BaseDependencyManager):
    """Manages pip package installation."""
    
    def __init__(self, project_path: Path, pip_executable: Optional[str] = None):
        """
        Initialize pip manager.
        
        Args:
            project_path: Path to the project root
            pip_executable: Path to pip executable (optional)
        """
        super().__init__(project_path)
        self.pip_executable = pip_executable or "pip"
    
    def install_packages(
        self,
        dependencies: List[Dependency],
        upgrade: bool = False,
        no_deps: bool = False,
        user: bool = False
    ) -> InstallResult:
        """
        Install packages using pip.
        
        Args:
            dependencies: List of dependencies to install
            upgrade: Whether to upgrade existing packages
            no_deps: Don't install dependencies
            user: Install to user site-packages
            
        Returns:
            InstallResult with installation status
        """
        start_time = datetime.now()
        installed = []
        failed = []
        skipped = []
        warnings = []
        
        logger.info(f"Installing {len(dependencies)} packages with pip")
        
        # Filter out stdlib modules
        stdlib_modules = {'pathlib', 'logging', 'typing', 'math'}
        filtered_dependencies = [
            dep for dep in dependencies 
            if dep.name.lower() not in stdlib_modules
        ]
        
        if len(filtered_dependencies) != len(dependencies):
            skipped_deps = [dep.name for dep in dependencies if dep.name.lower() in stdlib_modules]
            logger.info(f"Skipping stdlib modules: {skipped_deps}")
            skipped.extend(skipped_deps)
        
        # Install all packages at once for better performance
        if filtered_dependencies:
            try:
                # Build pip install command
                cmd_parts = [self.pip_executable, "install"]
                
                if upgrade:
                    cmd_parts.append("--upgrade")
                if no_deps:
                    cmd_parts.append("--no-deps")
                if user:
                    cmd_parts.append("--user")
                
                # Add all packages
                for dep in filtered_dependencies:
                    package_spec = dep.to_requirement_string()
                    cmd_parts.append(package_spec)
                
                logger.info(f"Running: {' '.join(cmd_parts)}")
                
                # Run installation
                code, stdout, stderr = self._run_command(cmd_parts)
                
                if code == 0:
                    # Parse installed packages from output
                    installed_packages = self._parse_installed_packages(stdout)
                    installed.extend([dep.name for dep in filtered_dependencies])
                    logger.success(f"Successfully installed {len(installed_packages)} packages")
                else:
                    failed.extend([dep.name for dep in filtered_dependencies])
                    logger.error(f"Failed to install packages: {stderr}")
                    warnings.append(f"Installation failed: {stderr[:200]}")
                    
            except Exception as e:
                failed.extend([dep.name for dep in filtered_dependencies])
                error_msg = str(e)
                logger.error(f"Exception installing packages: {error_msg}")
                warnings.append(f"Exception: {error_msg[:100]}")
        
        duration = (datetime.now() - start_time).total_seconds()
        success = len(failed) == 0
        
        message = f"Installed {len(installed)} packages"
        if failed:
            message += f", {len(failed)} failed"
        if skipped:
            message += f", {len(skipped)} skipped"
        
        logger.info(f"{message} in {duration:.2f}s")
        
        return InstallResult(
            success=success,
            installed=installed,
            failed=failed,
            skipped=skipped,
            warnings=warnings,
            message=message,
            duration=duration
        )
    
    def install_from_requirements(
        self,
        requirements_file: Path,
        upgrade: bool = False
    ) -> InstallResult:
        """
        Install packages from requirements.txt file.
        
        Args:
            requirements_file: Path to requirements.txt
            upgrade: Whether to upgrade existing packages
            
        Returns:
            InstallResult with installation status
        """
        # Convert to Path if string
        if isinstance(requirements_file, str):
            requirements_file = Path(requirements_file)
            
        start_time = datetime.now()
        
        if not requirements_file.exists():
            error = f"Requirements file not found: {requirements_file}"
            logger.error(error)
            return InstallResult(
                success=False,
                installed=[],
                failed=[],
                skipped=[],
                warnings=[error],
                message=error,
                duration=0.0
            )
        
        logger.info(f"Installing from {requirements_file}")
        
        try:
            # Build pip install command
            cmd = [self.pip_executable, "install", "-r", str(requirements_file)]
            
            if upgrade:
                cmd.append("--upgrade")
            
            # Run installation
            code, stdout, stderr = self._run_command(cmd)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if code == 0:
                # Parse installed packages from output
                installed = self._parse_installed_packages(stdout)
                message = f"Installed packages from {requirements_file.name} in {duration:.2f}s"
                logger.success(message)
                
                return InstallResult(
                    success=True,
                    installed=installed,
                    failed=[],
                    skipped=[],
                    warnings=[],
                    message=message,
                    duration=duration
                )
            else:
                error = f"Failed to install from requirements: {stderr}"
                logger.error(error)
                return InstallResult(
                    success=False,
                    installed=[],
                    failed=["requirements.txt"],
                    skipped=[],
                    warnings=[stderr[:200]],
                    message=error,
                    duration=duration
                )
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error = f"Exception installing from requirements: {e}"
            logger.error(error)
            return InstallResult(
                success=False,
                installed=[],
                failed=["requirements.txt"],
                skipped=[],
                warnings=[str(e)],
                message=error,
                duration=duration
            )
    
    def _parse_installed_packages(self, output: str) -> List[str]:
        """
        Parse installed package names from pip output.
        
        Args:
            output: Pip command output
            
        Returns:
            List of installed package names
        """
        packages = []
        for line in output.split("\n"):
            if "Successfully installed" in line:
                # Format: "Successfully installed package1-1.0 package2-2.0"
                parts = line.split("Successfully installed")[-1].strip().split()
                packages.extend([p.rsplit("-", 1)[0] for p in parts])
        return packages
    
    def uninstall_packages(
        self,
        package_names: List[str],
        yes: bool = True
    ) -> InstallResult:
        """
        Uninstall packages using pip.
        
        Args:
            package_names: List of package names to uninstall
            yes: Auto-confirm uninstallation
            
        Returns:
            InstallResult with uninstallation status
        """
        start_time = datetime.now()
        installed = []  # Successfully uninstalled
        failed = []
        warnings = []
        
        logger.info(f"Uninstalling {len(package_names)} packages")
        
        for package in package_names:
            try:
                cmd = [self.pip_executable, "uninstall", package]
                if yes:
                    cmd.append("-y")
                
                code, stdout, stderr = self._run_command(cmd)
                
                if code == 0:
                    installed.append(package)
                    logger.success(f"Uninstalled: {package}")
                else:
                    failed.append(package)
                    logger.error(f"Failed to uninstall {package}: {stderr}")
                    warnings.append(f"{package}: {stderr[:100]}")
                    
            except Exception as e:
                failed.append(package)
                logger.error(f"Exception uninstalling {package}: {e}")
                warnings.append(f"{package}: {str(e)[:100]}")
        
        duration = (datetime.now() - start_time).total_seconds()
        success = len(failed) == 0
        message = f"Uninstalled {len(installed)} packages"
        if failed:
            message += f", {len(failed)} failed"
        
        logger.info(f"{message} in {duration:.2f}s")
        
        return InstallResult(
            success=success,
            installed=installed,
            failed=failed,
            skipped=[],
            warnings=warnings,
            message=message,
            duration=duration
        )
    
    def list_installed_packages(self) -> List[PackageInfo]:
        """
        List all installed packages.
        
        Returns:
            List of PackageInfo objects
        """
        try:
            cmd = [self.pip_executable, "list", "--format=json"]
            code, stdout, stderr = self._run_command(cmd)
            
            if code == 0:
                import json
                packages_data = json.loads(stdout)
                packages = []
                
                for pkg in packages_data:
                    packages.append(PackageInfo(
                        name=pkg.get("name", ""),
                        version=pkg.get("version", ""),
                        location="",
                        requires=[],
                        required_by=[]
                    ))
                
                return packages
            return []
        except Exception as e:
            logger.error(f"Failed to list packages: {e}")
            return []
    
    def check_package_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed.
        
        Args:
            package_name: Name of the package
            
        Returns:
            True if installed, False otherwise
        """
        try:
            cmd = [self.pip_executable, "show", package_name]
            code, _, _ = self._run_command(cmd)
            return code == 0
        except Exception:
            return False


class CondaDependencyManager(BaseDependencyManager):
    """Manages conda package installation."""
    
    def __init__(self, project_path: Path, env_name: Optional[str] = None):
        """
        Initialize conda dependency manager.
        
        Args:
            project_path: Path to the project root
            env_name: Name of conda environment
        """
        super().__init__(project_path)
        self.env_name = env_name
        self.conda_available = self._check_conda_available()
    
    def _check_conda_available(self) -> bool:
        """Check if conda is available."""
        try:
            code, _, _ = self._run_command(["conda", "--version"])
            return code == 0
        except Exception:
            return False
    
    def install_packages(
        self,
        dependencies: List[Dependency],
        update: bool = False
    ) -> InstallResult:
        """
        Install packages using conda.
        
        Args:
            dependencies: List of dependencies to install
            update: Whether to update existing packages
            
        Returns:
            InstallResult with installation status
        """
        if not self.conda_available:
            error = "Conda is not available"
            return InstallResult(
                success=False,
                installed=[],
                failed=[d.name for d in dependencies],
                skipped=[],
                warnings=[error],
                message=error,
                duration=0.0
            )
        
        start_time = datetime.now()
        installed = []
        failed = []
        warnings = []
        
        logger.info(f"Installing {len(dependencies)} packages with conda")
        
        for dep in dependencies:
            try:
                # Build conda install command
                cmd = ["conda", "install"]
                
                if self.env_name:
                    cmd.extend(["-n", self.env_name])
                
                cmd.extend(["-y", dep.to_requirement_string()])
                
                if update:
                    cmd.append("--update-deps")
                
                logger.debug(f"Installing: {dep.name}")
                
                # Run installation
                code, stdout, stderr = self._run_command(cmd)
                
                if code == 0:
                    installed.append(dep.name)
                    logger.success(f"Installed: {dep.name}")
                else:
                    failed.append(dep.name)
                    logger.error(f"Failed to install {dep.name}: {stderr}")
                    warnings.append(f"{dep.name}: {stderr[:100]}")
                    
            except Exception as e:
                failed.append(dep.name)
                logger.error(f"Exception installing {dep.name}: {e}")
                warnings.append(f"{dep.name}: {str(e)[:100]}")
        
        duration = (datetime.now() - start_time).total_seconds()
        success = len(failed) == 0
        message = f"Installed {len(installed)} packages in {duration:.2f}s"
        
        return InstallResult(
            success=success,
            installed=installed,
            failed=failed,
            skipped=[],
            warnings=warnings,
            message=message,
            duration=duration
        )


class DependencyManager:
    """Main dependency manager that delegates to specific managers."""
    
    def __init__(
        self,
        project_path: Path,
        env_type: EnvironmentType = EnvironmentType.VENV,
        pip_executable: Optional[str] = None,
        conda_env_name: Optional[str] = None
    ):
        """
        Initialize dependency manager.
        
        Args:
            project_path: Path to the project root
            env_type: Type of environment
            pip_executable: Path to pip executable (for venv)
            conda_env_name: Name of conda environment (for conda)
        """
        self.project_path = Path(project_path)
        self.env_type = env_type
        
        if env_type == EnvironmentType.VENV:
            self.manager = PipManager(project_path, pip_executable)
        elif env_type == EnvironmentType.CONDA:
            self.manager = CondaDependencyManager(project_path, conda_env_name)
        else:
            self.manager = PipManager(project_path, pip_executable)
    
    def install_packages(
        self,
        dependencies: List[Dependency],
        **kwargs
    ) -> InstallResult:
        """
        Install packages.
        
        Args:
            dependencies: List of dependencies to install
            **kwargs: Additional arguments for specific managers
            
        Returns:
            InstallResult with installation status
        """
        return self.manager.install_packages(dependencies, **kwargs)
    
    def install_from_requirements(
        self,
        requirements_file: Path,
        **kwargs
    ) -> InstallResult:
        """
        Install packages from requirements.txt file.
        
        Args:
            requirements_file: Path to requirements.txt
            **kwargs: Additional arguments
            
        Returns:
            InstallResult with installation status
        """
        if isinstance(self.manager, PipManager):
            return self.manager.install_from_requirements(requirements_file, **kwargs)
        else:
            # Conda doesn't have direct requirements.txt support
            # Parse and install individually
            try:
                with open(requirements_file, 'r') as f:
                    lines = f.readlines()
                
                dependencies = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        dep = Dependency.from_string(line)
                        dependencies.append(dep)
                
                return self.manager.install_packages(dependencies)
            except Exception as e:
                logger.error(f"Failed to parse requirements: {e}")
                return InstallResult(
                    success=False,
                    installed=[],
                    failed=["requirements.txt"],
                    skipped=[],
                    warnings=[str(e)],
                    message=f"Failed to parse requirements: {e}",
                    duration=0.0
                )
    
    def parse_requirements_file(self, requirements_file: Path) -> List[Dependency]:
        """
        Parse requirements.txt file into Dependency objects.
        
        Args:
            requirements_file: Path to requirements.txt
            
        Returns:
            List of Dependency objects
        """
        dependencies = []
        
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            dep = Dependency.from_string(line)
                            dependencies.append(dep)
                        except Exception as e:
                            logger.warning(f"Failed to parse dependency '{line}': {e}")
        except Exception as e:
            logger.error(f"Failed to read requirements file: {e}")
        
        return dependencies
