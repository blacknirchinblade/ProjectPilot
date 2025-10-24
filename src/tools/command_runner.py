"""
Command Runner Tool - Executes shell commands and manages background tasks

This tool handles:
- Shell command execution with output capture
- pytest test execution
- Background process management (servers, watch tasks)
- Cross-platform command compatibility
- Timeout handling

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
import subprocess
import threading
import time
import signal
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from loguru import logger
import platform


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    command: str


@dataclass
class BackgroundTask:
    """Represents a background task."""
    task_id: str
    process: subprocess.Popen
    command: str
    start_time: float
    log_file: Optional[Path] = None


class CommandRunner:
    """
    Executes shell commands and manages background tasks.
    Provides cross-platform compatibility and safe execution.
    """
    
    def __init__(self, working_directory: Path, venv_path: Optional[Path] = None):
        """
        Initialize the command runner.
        
        Args:
            working_directory: Directory to execute commands in
            venv_path: Path to virtual environment (if any)
        """
        self.working_directory = Path(working_directory)
        self.venv_path = Path(venv_path) if venv_path else None
        self.platform = platform.system()
        self.background_tasks: Dict[str, BackgroundTask] = {}
        
        logger.info(f"Initialized CommandRunner")
        logger.info(f"Working directory: {self.working_directory}")
        logger.info(f"Virtual environment: {self.venv_path or 'None'}")
        logger.info(f"Platform: {self.platform}")
    
    def run_command(
        self,
        command: str | List[str],
        timeout: int = 300,
        capture_output: bool = True,
        use_venv: bool = True,
        env_vars: Optional[Dict[str, str]] = None
    ) -> CommandResult:
        """
        Execute a shell command and return the result.
        
        Args:
            command: Command to execute (string or list of arguments)
            timeout: Maximum execution time in seconds (default: 5 minutes)
            capture_output: Whether to capture stdout/stderr
            use_venv: Whether to use the virtual environment
            env_vars: Additional environment variables
        
        Returns:
            CommandResult object with execution details
        """
        start_time = time.time()
        
        try:
            # Convert command to list if string
            if isinstance(command, str):
                cmd_list = command.split()
                cmd_str = command
            else:
                cmd_list = command
                cmd_str = " ".join(command)
            
            logger.info(f"Executing command: {cmd_str}")
            
            # Prepare environment
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            
            # Add venv to PATH if requested
            if use_venv and self.venv_path:
                python_exe = self._get_venv_python()
                if python_exe.exists():
                    # Replace 'python' command with venv python
                    if cmd_list[0] in ['python', 'python3']:
                        cmd_list[0] = str(python_exe)
                    
                    # Add venv to PATH
                    venv_scripts = self.venv_path / ("Scripts" if self.platform == "Windows" else "bin")
                    env["PATH"] = f"{venv_scripts}{os.pathsep}{env['PATH']}"
            
            # Execute command
            result = subprocess.run(
                cmd_list,
                cwd=str(self.working_directory),
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=env
            )
            
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            
            if success:
                logger.success(f"✓ Command completed in {execution_time:.2f}s (exit code: {result.returncode})")
            else:
                logger.error(f"✗ Command failed in {execution_time:.2f}s (exit code: {result.returncode})")
            
            return CommandResult(
                success=success,
                return_code=result.returncode,
                stdout=result.stdout if capture_output else "",
                stderr=result.stderr if capture_output else "",
                execution_time=execution_time,
                command=cmd_str
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"✗ Command timed out after {timeout}s")
            return CommandResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                execution_time=execution_time,
                command=cmd_str
            )
        except FileNotFoundError as e:
            execution_time = time.time() - start_time
            logger.error(f"✗ Command not found: {e}")
            return CommandResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Command not found: {str(e)}",
                execution_time=execution_time,
                command=cmd_str
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception("Error executing command")
            return CommandResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Error: {str(e)}",
                execution_time=execution_time,
                command=cmd_str
            )
    
    def run_pytest(
        self,
        test_path: Optional[str] = None,
        verbose: bool = True,
        coverage: bool = False,
        markers: Optional[str] = None,
        extra_args: Optional[List[str]] = None
    ) -> CommandResult:
        """
        Run pytest with common options.
        
        Args:
            test_path: Specific test file or directory (default: "tests/")
            verbose: Enable verbose output (-v)
            coverage: Enable coverage reporting (--cov)
            markers: Run tests with specific markers (-m)
            extra_args: Additional pytest arguments
        
        Returns:
            CommandResult object
        """
        cmd = ["pytest"]
        
        # Add test path
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append("tests/")
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        
        # Add coverage
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html"])
        
        # Add markers
        if markers:
            cmd.extend(["-m", markers])
        
        # Add extra args
        if extra_args:
            cmd.extend(extra_args)
        
        logger.info(f"Running pytest: {' '.join(cmd)}")
        
        return self.run_command(cmd, timeout=600, use_venv=True)  # 10 minutes for tests
    
    def start_background_task(
        self,
        command: str | List[str],
        task_id: str,
        log_file: Optional[Path] = None,
        use_venv: bool = True
    ) -> Tuple[bool, str]:
        """
        Start a command as a background task (e.g., development server).
        
        Args:
            command: Command to execute
            task_id: Unique identifier for this task
            log_file: Optional file to log output to
            use_venv: Whether to use the virtual environment
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if task_id in self.background_tasks:
                return False, f"Task '{task_id}' already running"
            
            # Convert command to list if string
            if isinstance(command, str):
                cmd_list = command.split()
                cmd_str = command
            else:
                cmd_list = command
                cmd_str = " ".join(command)
            
            logger.info(f"Starting background task '{task_id}': {cmd_str}")
            
            # Prepare environment
            env = os.environ.copy()
            
            # Add venv to PATH if requested
            if use_venv and self.venv_path:
                python_exe = self._get_venv_python()
                if python_exe.exists() and cmd_list[0] in ['python', 'python3']:
                    cmd_list[0] = str(python_exe)
                
                venv_scripts = self.venv_path / ("Scripts" if self.platform == "Windows" else "bin")
                env["PATH"] = f"{venv_scripts}{os.pathsep}{env['PATH']}"
            
            # Setup log file
            stdout_dest = subprocess.PIPE
            stderr_dest = subprocess.PIPE
            
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_file, 'w', encoding='utf-8')
                stdout_dest = log_handle
                stderr_dest = subprocess.STDOUT
            
            # Start process
            process = subprocess.Popen(
                cmd_list,
                cwd=str(self.working_directory),
                stdout=stdout_dest,
                stderr=stderr_dest,
                text=True,
                env=env
            )
            
            # Store task
            task = BackgroundTask(
                task_id=task_id,
                process=process,
                command=cmd_str,
                start_time=time.time(),
                log_file=log_file
            )
            self.background_tasks[task_id] = task
            
            # Wait a moment to check if it started successfully
            time.sleep(0.5)
            
            if process.poll() is not None:
                # Process died immediately
                del self.background_tasks[task_id]
                return False, f"Task '{task_id}' failed to start (exit code: {process.returncode})"
            
            logger.success(f"✓ Background task '{task_id}' started (PID: {process.pid})")
            
            return True, f"Task '{task_id}' started successfully"
            
        except Exception as e:
            logger.exception(f"Error starting background task '{task_id}'")
            return False, f"Error: {str(e)}"
    
    def stop_background_task(self, task_id: str, timeout: int = 5) -> Tuple[bool, str]:
        """
        Stop a running background task.
        
        Args:
            task_id: ID of the task to stop
            timeout: Seconds to wait before force-killing
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if task_id not in self.background_tasks:
                return False, f"Task '{task_id}' not found"
            
            task = self.background_tasks[task_id]
            
            logger.info(f"Stopping background task '{task_id}'...")
            
            # Try graceful shutdown first
            task.process.terminate()
            
            try:
                task.process.wait(timeout=timeout)
                logger.success(f"✓ Task '{task_id}' stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning(f"Task '{task_id}' did not stop gracefully, force-killing...")
                task.process.kill()
                task.process.wait()
                logger.success(f"✓ Task '{task_id}' force-killed")
            
            # Cleanup
            runtime = time.time() - task.start_time
            del self.background_tasks[task_id]
            
            return True, f"Task '{task_id}' stopped (ran for {runtime:.1f}s)"
            
        except Exception as e:
            logger.exception(f"Error stopping task '{task_id}'")
            return False, f"Error: {str(e)}"
    
    def get_task_status(self, task_id: str) -> Tuple[bool, Dict]:
        """
        Get status of a background task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Tuple of (found: bool, status: dict)
        """
        if task_id not in self.background_tasks:
            return False, {}
        
        task = self.background_tasks[task_id]
        poll_result = task.process.poll()
        
        status = {
            "task_id": task_id,
            "command": task.command,
            "pid": task.process.pid,
            "running": poll_result is None,
            "exit_code": poll_result,
            "runtime": time.time() - task.start_time,
            "log_file": str(task.log_file) if task.log_file else None
        }
        
        return True, status
    
    def list_background_tasks(self) -> List[Dict]:
        """
        List all running background tasks.
        
        Returns:
            List of task status dictionaries
        """
        tasks = []
        for task_id in list(self.background_tasks.keys()):
            found, status = self.get_task_status(task_id)
            if found:
                tasks.append(status)
        
        return tasks
    
    def stop_all_tasks(self) -> int:
        """
        Stop all running background tasks.
        
        Returns:
            Number of tasks stopped
        """
        task_ids = list(self.background_tasks.keys())
        stopped = 0
        
        for task_id in task_ids:
            success, _ = self.stop_background_task(task_id)
            if success:
                stopped += 1
        
        return stopped
    
    def install_package(self, package: str, upgrade: bool = False) -> CommandResult:
        """
        Install a Python package using pip.
        
        Args:
            package: Package name (e.g., "numpy", "pandas==1.5.0")
            upgrade: Whether to upgrade if already installed
        
        Returns:
            CommandResult object
        """
        cmd = ["pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        cmd.append(package)
        
        logger.info(f"Installing package: {package}")
        
        return self.run_command(cmd, timeout=300, use_venv=True)
    
    def uninstall_package(self, package: str) -> CommandResult:
        """
        Uninstall a Python package using pip.
        
        Args:
            package: Package name
        
        Returns:
            CommandResult object
        """
        cmd = ["pip", "uninstall", "-y", package]
        
        logger.info(f"Uninstalling package: {package}")
        
        return self.run_command(cmd, timeout=60, use_venv=True)
    
    def list_packages(self) -> CommandResult:
        """
        List installed Python packages.
        
        Returns:
            CommandResult with package list in stdout
        """
        cmd = ["pip", "list"]
        
        return self.run_command(cmd, use_venv=True)
    
    # Helper methods
    
    def _get_venv_python(self) -> Path:
        """Get path to Python executable in venv."""
        if self.platform == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def __del__(self):
        """Cleanup: stop all background tasks."""
        if hasattr(self, 'background_tasks') and self.background_tasks:
            logger.info("Cleaning up background tasks...")
            self.stop_all_tasks()


# CODE_GENERATION_COMPLETE
