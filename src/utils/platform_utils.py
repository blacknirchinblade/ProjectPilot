"""
Platform Utils - Cross-platform compatibility utilities
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com

"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional, List
from loguru import logger


class PlatformUtils:
    """Cross-platform utilities for Windows and Linux"""
    
    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows"""
        return platform.system() == "Windows"
    
    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux"""
        return platform.system() == "Linux"
    
    @staticmethod
    def is_mac() -> bool:
        """Check if running on Mac"""
        return platform.system() == "Darwin"
    
    @staticmethod
    def get_os_name() -> str:
        """Get OS name"""
        return platform.system()
    
    @staticmethod
    def get_path_separator() -> str:
        """Get path separator for current OS"""
        return "\\" if PlatformUtils.is_windows() else "/"
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize path for current OS
        
        Args:
            path: Path string
        
        Returns:
            Normalized path
        """
        return str(Path(path).resolve())
    
    @staticmethod
    def run_command(
        command: str,
        shell: bool = True,
        cwd: Optional[str] = None,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run command cross-platform
        
        Args:
            command: Command to run
            shell: Run in shell
            cwd: Working directory
            capture_output: Capture stdout/stderr
        
        Returns:
            CompletedProcess result
        """
        try:
            logger.debug(f"Running command: {command}")
            
            # On Windows, adjust path separators in command if needed
            if PlatformUtils.is_windows():
                # Use cmd.exe for shell commands on Windows
                if shell and not command.startswith("cmd"):
                    command = f"cmd /c {command}"
            
            result = subprocess.run(
                command,
                shell=shell,
                cwd=cwd,
                capture_output=capture_output,
                text=True
            )
            
            logger.debug(f"Command exit code: {result.returncode}")
            return result
        
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            raise
    
    @staticmethod
    def get_python_executable() -> str:
        """
        Get Python executable path
        
        Returns:
            Path to Python executable
        """
        import sys
        return sys.executable
    
    @staticmethod
    def create_virtual_env(venv_path: str) -> bool:
        """
        Create virtual environment
        
        Args:
            venv_path: Path for virtual environment
        
        Returns:
            True if successful
        """
        try:
            python_exe = PlatformUtils.get_python_executable()
            command = f'"{python_exe}" -m venv "{venv_path}"'
            result = PlatformUtils.run_command(command)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error creating virtual environment: {str(e)}")
            return False
    
    @staticmethod
    def get_temp_dir() -> Path:
        """Get system temp directory"""
        import tempfile
        return Path(tempfile.gettempdir())
    
    @staticmethod
    def get_home_dir() -> Path:
        """Get user home directory"""
        return Path.home()
    
    @staticmethod
    def join_paths(*paths: str) -> str:
        """
        Join paths using OS-specific separator
        
        Args:
            *paths: Path components
        
        Returns:
            Joined path
        """
        return str(Path(*paths))
