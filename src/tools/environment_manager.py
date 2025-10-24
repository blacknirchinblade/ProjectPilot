"""
Environment Manager for Setup Automation Tool.
Manages virtual environment creation (venv, conda, virtualenv).
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
import platform
import re
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Tuple
import sys
from datetime import datetime
from loguru import logger

from src.tools.data_structures import (
    EnvironmentInfo,
    EnvResult,
    EnvironmentType,
    PackageInfo
)


class BaseEnvironmentManager:
    """Base class for environment managers."""
    
    def __init__(self, project_path: Path):
        """
        Initialize environment manager.
        
        Args:
            project_path: Path to the project root
        """
        self.project_path = Path(project_path).resolve()
        logger.debug(f"BaseEnvironmentManager initialized with project path: {self.project_path}")
        self.platform = platform.system()
        
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
            # Don't use shell=True to avoid path issues on Windows
            result = subprocess.run(
                command,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False  # Critical fix: don't use shell
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
        except FileNotFoundError as e:
            logger.error(f"Command not found: {command[0]}")
            return 1, "", f"Command not found: {command[0]}"
        except Exception as e:
            return 1, "", str(e)
    
    def get_python_version(self, python_executable: Path) -> str:
        """
        Get Python version from executable.
        
        Args:
            python_executable: Path to Python executable
            
        Returns:
            Version string (e.g., "3.10.18")
        """
        try:
            code, stdout, _ = self._run_command(
                [str(python_executable), "--version"]
            )
            if code == 0:
                # Output format: "Python 3.10.18"
                return stdout.strip().split()[-1]
            return "unknown"
        except Exception:
            return "unknown"


class VenvManager(BaseEnvironmentManager):
    """Manages Python venv virtual environments."""
    
    def create_environment(
        self,
        env_name: str = "venv",
        python_version: Optional[str] = None,
        system_site_packages: bool = False
    ) -> EnvResult:
        """
        Create a virtual environment using venv.
        """
        start_time = datetime.now()
        env_path = self.project_path / env_name
        commands_executed = []
        
        logger.info(f"Creating venv at {env_path}")
        logger.info(f"Project path: {self.project_path}")
        logger.info(f"Absolute project path: {self.project_path.absolute()}")
        
        try:
            python_cmd = self._find_python_executable()
            if not python_cmd:
                raise FileNotFoundError("Could not find a Python executable.")

            logger.info(f"Using Python: {python_cmd}")

            # Build venv creation command
            cmd = [python_cmd, "-m", "venv"]
            if system_site_packages:
                cmd.append("--system-site-packages")
            cmd.append(str(env_path))
            
            commands_executed.append(" ".join(cmd))
            logger.info(f"Venv command: {' '.join(cmd)}")
            
            # Create venv
            code, stdout, stderr = self._run_command(cmd)
            
            logger.info(f"Venv creation return code: {code}")
            if stdout:
                logger.info(f"Venv creation stdout: {stdout}")
            if stderr:
                logger.warning(f"Venv creation stderr: {stderr}")
            
            if code != 0:
                error_msg = f"Failed to create venv: {stderr}"
                logger.error(error_msg)
                duration = (datetime.now() - start_time).total_seconds()
                return EnvResult(
                    success=False,
                    env_info=None,
                    message=error_msg,
                    error=stderr,
                    duration=duration,
                    commands_executed=commands_executed
                )
            
            # Verify the environment was created with detailed checks
            if not env_path.exists():
                error_msg = f"Venv directory was not created at {env_path}"
                logger.error(error_msg)
                duration = (datetime.now() - start_time).total_seconds()
                return EnvResult(
                    success=False,
                    env_info=None,
                    message=error_msg,
                    error="Venv directory missing",
                    duration=duration,
                    commands_executed=commands_executed
                )
            
            # Check if essential files were created
            if self.platform == "Windows":
                required_files = [
                    env_path / "Scripts" / "python.exe",
                    env_path / "Scripts" / "pip.exe"
                ]
            else:
                required_files = [
                    env_path / "bin" / "python",
                    env_path / "bin" / "pip"
                ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                logger.warning(f"Missing files in venv: {missing_files}")
            
            # Get environment info
            env_info = self.get_environment_info(env_path)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if env_info:
                logger.success(f"Venv created successfully at {env_path}")
                return EnvResult(
                    success=True,
                    env_info=env_info,
                    message=f"Venv created successfully at {env_path}",
                    duration=duration,
                    commands_executed=commands_executed
                )
            else:
                logger.warning(f"Venv created but could not retrieve environment info")
                # Still consider it a success if the directory exists
                return EnvResult(
                    success=True,
                    env_info=None,
                    message=f"Venv created at {env_path} but environment info unavailable",
                    duration=duration,
                    commands_executed=commands_executed
                )
                
        except Exception as e:
            logger.error(f"An unexpected error occurred during venv creation: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            return EnvResult(
                success=False,
                env_info=None,
                message=f"An unexpected error occurred: {e}",
                error=str(e),
                duration=duration,
                commands_executed=commands_executed
            )
        
    def _find_python_executable(self) -> str:
        """Find a suitable Python executable."""
        # In a virtual env, sys.executable may be the app entry point.
        # The real python executable is in the bin directory of the prefix.
        
        if hasattr(sys, 'real_prefix') or hasattr(sys, 'base_prefix'):
            prefix = getattr(sys, 'real_prefix', getattr(sys, 'base_prefix', sys.prefix))
            if platform.system() == "Windows":
                python_exe = Path(prefix) / "python.exe"
                if python_exe.exists():
                    logger.debug(f"Found python executable in venv prefix: {python_exe}")
                    return str(python_exe)
                python_exe = Path(prefix) / "python3.exe"
                if python_exe.exists():
                 logger.debug(f"Found python executable in venv prefix: {python_exe}")
                 return str(python_exe)
            else:
                python_exe = Path(prefix) / "bin" / "python"
            
                if python_exe.exists():
                    logger.debug(f"Found python executable in venv: {python_exe}")
                    return str(python_exe)
                python_exe = Path(prefix) / "bin" / "python3"
                if python_exe.exists():
                    logger.debug(f"Found python executable in venv prefix: {python_exe}")
                    return str(python_exe)
            python_exe = Path(prefix) / "bin" / "python3"
                
            # Fallback to checking the PATH
            python_exe = shutil.which("python3") or shutil.which("python")
            if python_exe:
                logger.debug(f"Found python executable in PATH: {python_exe}")
                return python_exe
            
            logger.error("Could not find any python executable. Please ensure python is in your PATH. Falling back to 'python'.")
            return "python"

    def create_environment(
        self,
        env_name: str = "venv",
        python_version: Optional[str] = None,
        system_site_packages: bool = False
    ) -> EnvResult:
        """
        Create a virtual environment using venv.
        """
        env_name = "venv"  # Force consistent env name
        start_time = datetime.now()
        env_path = self.project_path / env_name
        commands_executed = []
        
        logger.info(f"Creating venv at {env_path}")
        
        try:
            python_cmd = self._find_python_executable()
            if not python_cmd:
                raise FileNotFoundError("Could not find a Python executable.")

            # Build venv creation command
            cmd = [python_cmd, "-m", "venv"]
            if system_site_packages:
                cmd.append("--system-site-packages")
            cmd.append(str(env_path))
            
            commands_executed.append(" ".join(cmd))
            
            # Create venv
            code, stdout, stderr = self._run_command(cmd)
            
            if code != 0:
                error_msg = f"Failed to create venv: {stderr}"
                if "[WinError 2]" in stderr or "Command not found" in stderr:
                    error_msg = "Failed to create venv. Make sure 'python' is in your system's PATH."
                
                logger.error(error_msg)
                duration = (datetime.now() - start_time).total_seconds()
                return EnvResult(
                    success=False,
                    env_info=None,
                    message=error_msg,
                    error=stderr,
                    duration=duration,
                    commands_executed=commands_executed
                )
            
            # Get environment info
            env_info = self.get_environment_info(env_path)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if not env_info:
                return EnvResult(
                    success=False,
                    env_info=None,
                    message="Venv created, but failed to get environment info.",
                    error="Could not retrieve details after creation.",
                    duration=duration,
                    commands_executed=commands_executed
                )

            return EnvResult(
                success=True,
                env_info=env_info,
                message=f"Venv created successfully at {env_path}",
                duration=duration,
                commands_executed=commands_executed
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during venv creation: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            return EnvResult(
                success=False,
                env_info=None,
                message=f"An unexpected error occurred: {e}",
                error=str(e),
                duration=duration,
                commands_executed=commands_executed
            )
            
            
            
        
    
    def get_environment_info(self, env_path: Path) -> EnvironmentInfo:
        """
        Get information about a venv environment.
        """
        if isinstance(env_path, str):
            env_path = Path(env_path)
            
        if self.platform == "Windows":
            python_exe = env_path / "Scripts" / "python.exe"
            pip_exe = env_path / "Scripts" / "pip.exe"
        else:
            python_exe = env_path / "bin" / "python"
            pip_exe = env_path / "bin" / "pip"

        if not python_exe.exists():
            logger.warning(f"Could not find python executable at {python_exe}. Venv may be corrupt.")
            return None
        
        python_version = self.get_python_version(python_exe)
        packages = self._get_installed_packages(pip_exe)
        
        return EnvironmentInfo(
            type=EnvironmentType.VENV,
            name=env_path.name,
            path=env_path, # Keep as Path object
            python_version=python_version,
            python_executable=python_exe, # Keep as Path object
            is_active=False,
            packages=packages,
            created_at=datetime.fromtimestamp(env_path.stat().st_ctime) if env_path.exists() else datetime.now()
        )
    
    def _get_installed_packages(self, pip_exe: Path) -> List[str]:
        """
        Get list of installed packages.
        """
        try:
            code, stdout, _ = self._run_command(
                [str(pip_exe), "list", "--format=freeze"]
            )
            if code == 0:
                return [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            return []
        except Exception:
            return []
    
    def activate_commands(self, env_path: Path) -> List[str]:
        """
        Get commands to activate the environment.
        """
        if self.platform == "Windows":
            return [
                f"call {env_path}\\Scripts\\activate.bat",
                f". {env_path}\\Scripts\\Activate.ps1"
            ]
        else:
            return [f"source {env_path}/bin/activate"]
    
    def list_environments(self) -> List[Path]:
        """
        List all venv environments in project.
        """
        envs = []
        common_names = ["venv", ".venv", "env", ".env", "virtualenv"]
        
        for name in common_names:
            env_path = self.project_path / name
            if env_path.exists() and env_path.is_dir():
                if self.platform == "Windows":
                    if (env_path / "Scripts" / "python.exe").exists():
                        envs.append(env_path)
                else:
                    if (env_path / "bin" / "python").exists():
                        envs.append(env_path)
        
        return envs


class CondaManager(BaseEnvironmentManager):
    """Manages Conda virtual environments."""
    
    def __init__(self, project_path: Path):
        """Initialize conda manager."""
        super().__init__(project_path)
        self.conda_executable = self._find_conda_executable()
        self.conda_available = self.conda_executable is not None
    
    def _find_conda_executable(self) -> Optional[str]:
        """Find the conda executable."""
        # First, try to find conda in the system's PATH
        conda_exe = shutil.which("conda")
        if conda_exe:
            logger.info(f"Found conda executable in PATH: {conda_exe}")
            return conda_exe
        
        # Add common locations for Windows if PATH fails
        if self.platform == "Windows":
            common_paths = [
                Path(os.environ.get("CONDA_PREFIX", "")) / "Scripts" / "conda.exe",
                Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "Scripts" / "conda.exe",
                Path(os.environ.get("USERPROFILE", "")) / "miniconda3" / "Scripts" / "conda.exe",
            ]
            for path in common_paths:
                if path.exists():
                    logger.info(f"Found conda executable in common location: {path}")
                    return str(path)

        logger.warning("Conda executable not found in PATH or common locations.")
        return None

    def _run_conda_command(self, command: List[str], **kwargs) -> Tuple[int, str, str]:
        """Helper to run a command with the found conda executable."""
        if not self.conda_available:
            return 1, "", "Conda is not available."
        return self._run_command([self.conda_executable] + command, **kwargs)

    def _check_conda_available(self) -> bool:
        """Check if conda is available."""
        return self.conda_available
    
    def create_environment(
        self,
        env_name: str,
        python_version: str = "3.10",
        packages: Optional[List[str]] = None
    ) -> EnvResult:
        """
        Create a conda environment.
        
        Args:
            env_name: Name of the environment
            python_version: Python version to use
            packages: Additional packages to install
            
        Returns:
            EnvResult with creation status
        """
        start_time = datetime.now()
        commands_executed = []
        
        if not self.conda_available:
            error = "Conda is not available on this system"
            logger.error(error)
            return EnvResult(
                success=False,
                env_info=None,
                message=error,
                error=error,
                duration=0.0,
                commands_executed=[]
            )
        
        logger.info(f"Creating conda environment: {env_name}")
        
        try:
            # Build conda create command
            cmd = [
                "create",
                "--prefix", str(self.project_path / env_name), # Create in project dir
                f"python={python_version}",
                "-y"
            ]
            
            if packages:
                cmd.extend(packages)
            
            commands_executed.append(" ".join([self.conda_executable] + cmd))
            
            # Create environment
            code, stdout, stderr = self._run_conda_command(cmd)
            
            if code != 0:
                error = f"Failed to create conda environment: {stderr}"
                logger.error(error)
                duration = (datetime.now() - start_time).total_seconds()
                return EnvResult(
                    success=False,
                    env_info=None,
                    message=error,
                    error=stderr,
                    duration=duration,
                    commands_executed=commands_executed
                )
            
            # Get environment info
            env_info = self.get_environment_info(env_name)
            
            duration = (datetime.now() - start_time).total_seconds()
            message = f"Created conda environment '{env_name}' in {duration:.2f}s"
            logger.success(message)
            
            return EnvResult(
                success=True,
                env_info=env_info,
                message=message,
                error=None,
                duration=duration,
                commands_executed=commands_executed
            )
            
        except Exception as e:
            error = f"Exception creating conda environment: {e}"
            logger.error(error)
            duration = (datetime.now() - start_time).total_seconds()
            return EnvResult(
                success=False,
                env_info=None,
                message=error,
                error=str(e),
                duration=duration,
                commands_executed=commands_executed
            )
    
    def get_environment_info(self, env_name: str) -> Optional[EnvironmentInfo]:
        """
        Get information about a conda environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            EnvironmentInfo or None if not found
        """
        try:
            # Check for env in the local project directory first
            env_path = self.project_path / env_name
            if not (env_path / "conda-meta").exists():
                 # Fallback to searching all conda envs
                code, stdout, _ = self._run_conda_command(
                    ["info", "--envs", "--json"]
                )
            
                if code != 0:
                    return None
            
                import json
                info = json.loads(stdout)
                envs = info.get("envs", [])
            
                # Find environment path
                env_path = None
                for path in envs:
                    if Path(path).name == env_name or env_name in path:
                        env_path = Path(path)
                        break
                else:
                    return None
            
            if not env_path:
                return None
            
            # Get Python version
            if self.platform == "Windows":
                python_exe = env_path / "python.exe"
            else:
                python_exe = env_path / "bin" / "python"
            
            python_version = self.get_python_version(python_exe)
            
            # Get installed packages
            packages = self._get_installed_packages(env_name)
            
            return EnvironmentInfo(
                type=EnvironmentType.CONDA,
                name=env_name,
                path=env_path,
                python_version=python_version,
                python_executable=python_exe,
                is_active=False,
                packages=packages,
                created_at=datetime.fromtimestamp(env_path.stat().st_ctime)
            )
            
        except Exception as e:
            logger.error(f"Failed to get conda environment info: {e}")
            return None
    
    def _get_installed_packages(self, env_name_or_path: str) -> List[str]:
        """
        Get list of installed packages in conda environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            List of package names with versions
        """
        try:
            # Use --prefix for local envs
            env_path = self.project_path / env_name_or_path
            if env_path.exists():
                cmd = ["list", "--prefix", str(env_path), "--export"]
            else: # Use --name for global envs
                cmd = ["list", "-n", env_name_or_path, "--export"]

            code, stdout, _ = self._run_conda_command(cmd)
            if code == 0:
                return [line.strip() for line in stdout.strip().split("\n") if line.strip() and not line.startswith("#")]
            return []
        except Exception:
            return []
    
    def activate_commands(self, env_name: str) -> List[str]:
        """
        Get commands to activate the conda environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            List of activation commands
        """
        return [f"conda activate {env_name}"]
    
    def list_environments(self) -> List[str]:
        """
        List all conda environments.
        
        Returns:
            List of environment names
        """
        if not self.conda_available:
            return []
        
        try:
            code, stdout, _ = self._run_conda_command(["env", "list"])
            if code == 0:
                envs = []
                for line in stdout.strip().split("\n"):
                    if line.strip() and not line.startswith("#"):
                        # Format: "name  *  /path/to/env"
                        parts = line.split()
                        if parts:
                            envs.append(parts[0])
                return envs
            return []
        except Exception:
            return []


class EnvironmentManager:
    """Main environment manager that delegates to specific managers."""
    
    def __init__(self, project_path: Path):
        """
        Initialize environment manager.
        
        Args:
            project_path: Path to the project root
        """
        self.project_path = Path(project_path)
        self.venv_manager = VenvManager(project_path)
        self.conda_manager = CondaManager(project_path)
    
    def create_environment(
        self,
        env_type: EnvironmentType = EnvironmentType.VENV,
        env_name: str = "venv",
        python_version: Optional[str] = None,
        **kwargs
    ) -> EnvResult:
        """
        Create a virtual environment with fallback logic.
        
        Tries to create the specified 'env_type'.
        If it fails (e.g., Conda not found), it falls back to 'venv'.
        """
        logger.info(f"Attempting to create {env_type.value} environment: {env_name}")
        
        if env_type == EnvironmentType.CONDA:
            result = self.conda_manager.create_environment(
                env_name=env_name,
                python_version=python_version or "3.10",
                **kwargs
            )
            if result.success:
                return result
            
            # --- FALLBACK LOGIC ---
            logger.warning("Conda creation failed. Falling back to venv.")
            logger.warning(f"Conda error: {result.error}")
            env_type = EnvironmentType.VENV
            env_name = "venv" # Use standard venv name
        if env_type == EnvironmentType.VENV:
            return self.venv_manager.create_environment(
                env_name=env_name,
                python_version=python_version,
                **kwargs
            )
        
        else:
            return EnvResult(
                success=False,
                env_info=None,
                message=f"Unsupported environment type: {env_type}",
                error=f"Only VENV and CONDA are supported",
                duration=0.0,
                commands_executed=[]
            )
    
    def get_environment_info(
        self,
        env_type: EnvironmentType,
        env_identifier: str
    ) -> Optional[EnvironmentInfo]:
        """
        Get information about an environment.
        
        Args:
            env_type: Type of environment
            env_identifier: Path (for venv) or name (for conda)
            
        Returns:
            EnvironmentInfo or None
        """
        if env_type == EnvironmentType.VENV:
            env_path = Path(env_identifier)
            if not env_path.is_absolute():
                env_path = self.project_path / env_identifier
            return self.venv_manager.get_environment_info(env_path)
        elif env_type == EnvironmentType.CONDA:
            # Check local path first
            local_path = self.project_path / env_identifier
            if local_path.exists():
                return self.conda_manager.get_environment_info(str(local_path))
            # Fallback to name
            return self.conda_manager.get_environment_info(env_identifier)
        return None
    
    def list_all_environments(self) -> dict:
        """
        List all environments (venv and conda).
        
        Returns:
            Dictionary with venv and conda environments
        """
        return {
            "venv": [str(p) for p in self.venv_manager.list_environments()],
            "conda": self.conda_manager.list_environments()
        }
