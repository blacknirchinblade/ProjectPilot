"""
Task Runner - Execute shell commands, Python scripts, and tests.

Provides cross-platform command execution with:
- Process management (timeout, background tasks, kill)
- Output capture (stdout/stderr streaming)
- Environment variable management
- Pytest integration
- Command history tracking

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import subprocess
import platform
import os
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class ProcessStatus(Enum):
    """Process execution status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


class OutputMode(Enum):
    """Output capture mode."""
    CAPTURE = "capture"  # Capture all output
    STREAM = "stream"    # Print output in real-time
    BOTH = "both"        # Capture and stream


@dataclass
class CommandResult:
    """Result of command execution."""
    command: str
    status: ProcessStatus
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    pid: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Check if command executed successfully."""
        return self.status == ProcessStatus.COMPLETED and self.exit_code == 0
    
    @property
    def output(self) -> str:
        """Get combined output."""
        return f"{self.stdout}\n{self.stderr}".strip()


@dataclass
class BackgroundTask:
    """Background process information."""
    task_id: str
    command: str
    process: subprocess.Popen
    start_time: datetime
    output_queue: queue.Queue
    status: ProcessStatus = ProcessStatus.RUNNING
    
    @property
    def pid(self) -> int:
        """Get process ID."""
        return self.process.pid
    
    @property
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None


class TaskRunner:
    """
    Execute shell commands and Python scripts across platforms.
    
    Features:
    - Cross-platform command execution (Windows/Linux/macOS)
    - Process timeout and background task management
    - Output streaming and capture
    - Environment variable management
    - Command history tracking
    - Pytest integration
    """
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        default_timeout: int = 300,  # 5 minutes
        env_vars: Optional[Dict[str, str]] = None
    ):
        """
        Initialize task runner.
        
        Args:
            working_dir: Default working directory for commands
            default_timeout: Default timeout in seconds
            env_vars: Additional environment variables
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.default_timeout = default_timeout
        self.env_vars = env_vars or {}
        self.platform = platform.system()
        
        # Command history
        self.command_history: List[CommandResult] = []
        
        # Background tasks
        self.background_tasks: Dict[str, BackgroundTask] = {}
        
        logger.info(f"TaskRunner initialized (platform: {self.platform}, cwd: {self.working_dir})")
    
    def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None,
        output_mode: OutputMode = OutputMode.CAPTURE,
        shell: bool = True
    ) -> CommandResult:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (None = no timeout)
            working_dir: Working directory for command
            env_vars: Additional environment variables
            output_mode: How to handle output (capture/stream/both)
            shell: Whether to use shell execution
        
        Returns:
            CommandResult with execution details
        """
        start_time = time.time()
        timeout = timeout or self.default_timeout
        cwd = Path(working_dir) if working_dir else self.working_dir
        
        # Merge environment variables
        env = os.environ.copy()
        env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)
        
        logger.info(f"Executing: {command[:100]}...")
        logger.debug(f"Working directory: {cwd}")
        logger.debug(f"Timeout: {timeout}s")
        
        try:
            # Start process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=shell,
                cwd=str(cwd),
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Capture output
            if output_mode in (OutputMode.STREAM, OutputMode.BOTH):
                stdout, stderr = self._stream_output(process, timeout)
            else:
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    execution_time = time.time() - start_time
                    
                    result = CommandResult(
                        command=command,
                        status=ProcessStatus.TIMEOUT,
                        exit_code=-1,
                        stdout=stdout or "",
                        stderr=stderr or f"Command timed out after {timeout}s",
                        execution_time=execution_time,
                        pid=process.pid,
                        error_message=f"Timeout after {timeout}s"
                    )
                    self.command_history.append(result)
                    logger.warning(f"Command timed out: {command[:50]}...")
                    return result
            
            execution_time = time.time() - start_time
            exit_code = process.returncode
            
            # Determine status
            if exit_code == 0:
                status = ProcessStatus.COMPLETED
            else:
                status = ProcessStatus.FAILED
            
            result = CommandResult(
                command=command,
                status=status,
                exit_code=exit_code,
                stdout=stdout or "",
                stderr=stderr or "",
                execution_time=execution_time,
                pid=process.pid
            )
            
            self.command_history.append(result)
            
            if result.success:
                logger.success(f"Command completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"Command failed with exit code {exit_code}")
                logger.error(f"stderr: {stderr[:200]}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            result = CommandResult(
                command=command,
                status=ProcessStatus.FAILED,
                exit_code=-1,
                stdout="",
                stderr=error_msg,
                execution_time=execution_time,
                error_message=error_msg
            )
            
            self.command_history.append(result)
            logger.error(f"Command execution failed: {error_msg}")
            return result
    
    def run_python_script(
        self,
        script_path: Path,
        args: Optional[List[str]] = None,
        python_executable: str = "python",
        timeout: Optional[int] = None,
        working_dir: Optional[Path] = None,
        output_mode: OutputMode = OutputMode.CAPTURE
    ) -> CommandResult:
        """
        Run a Python script.
        
        Args:
            script_path: Path to Python script
            args: Command line arguments
            python_executable: Python executable to use
            timeout: Timeout in seconds
            working_dir: Working directory
            output_mode: How to handle output
        
        Returns:
            CommandResult with execution details
        """
        if not script_path.exists():
            return CommandResult(
                command=str(script_path),
                status=ProcessStatus.FAILED,
                exit_code=-1,
                stdout="",
                stderr=f"Script not found: {script_path}",
                execution_time=0.0,
                error_message=f"Script not found: {script_path}"
            )
        
        # Build command
        cmd_parts = [python_executable, str(script_path)]
        if args:
            cmd_parts.extend(args)
        
        command = " ".join(cmd_parts)
        
        return self.execute_command(
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            output_mode=output_mode
        )
    
    def run_pytest(
        self,
        test_path: Optional[Path] = None,
        markers: Optional[List[str]] = None,
        verbose: bool = True,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        working_dir: Optional[Path] = None
    ) -> CommandResult:
        """
        Run pytest tests.
        
        Args:
            test_path: Path to test file/directory (None = all tests)
            markers: Test markers to filter (-m option)
            verbose: Verbose output (-v)
            capture_output: Capture output (-s if False)
            timeout: Timeout in seconds
            working_dir: Working directory
        
        Returns:
            CommandResult with test execution details
        """
        cmd_parts = ["pytest"]
        
        # Add test path
        if test_path:
            cmd_parts.append(str(test_path))
        
        # Add options
        if verbose:
            cmd_parts.append("-v")
        
        if not capture_output:
            cmd_parts.append("-s")
        
        # Add markers
        if markers:
            for marker in markers:
                cmd_parts.extend(["-m", marker])
        
        # Add color output
        cmd_parts.append("--color=yes")
        
        command = " ".join(cmd_parts)
        
        return self.execute_command(
            command=command,
            timeout=timeout or 600,  # 10 minutes default for tests
            working_dir=working_dir,
            output_mode=OutputMode.STREAM
        )
    
    def start_background_task(
        self,
        command: str,
        task_id: Optional[str] = None,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a command as a background task.
        
        Args:
            command: Command to execute
            task_id: Unique task identifier (auto-generated if None)
            working_dir: Working directory
            env_vars: Additional environment variables
        
        Returns:
            Task ID
        """
        task_id = task_id or f"task_{int(time.time())}"
        cwd = Path(working_dir) if working_dir else self.working_dir
        
        # Merge environment variables
        env = os.environ.copy()
        env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)
        
        logger.info(f"Starting background task '{task_id}': {command[:50]}...")
        
        # Start process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Create output queue
        output_queue = queue.Queue()
        
        # Start output reader threads
        threading.Thread(
            target=self._read_output,
            args=(process.stdout, output_queue),
            daemon=True
        ).start()
        
        threading.Thread(
            target=self._read_output,
            args=(process.stderr, output_queue),
            daemon=True
        ).start()
        
        # Store task
        task = BackgroundTask(
            task_id=task_id,
            command=command,
            process=process,
            start_time=datetime.now(),
            output_queue=output_queue
        )
        
        self.background_tasks[task_id] = task
        
        logger.success(f"Background task started: {task_id} (PID: {process.pid})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[ProcessStatus]:
        """
        Get status of a background task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            ProcessStatus or None if task not found
        """
        task = self.background_tasks.get(task_id)
        if not task:
            return None
        
        if task.process.poll() is None:
            return ProcessStatus.RUNNING
        else:
            task.status = ProcessStatus.COMPLETED if task.process.returncode == 0 else ProcessStatus.FAILED
            return task.status
    
    def get_task_output(self, task_id: str, clear: bool = False) -> List[str]:
        """
        Get output from a background task.
        
        Args:
            task_id: Task identifier
            clear: Clear output queue after reading
        
        Returns:
            List of output lines
        """
        task = self.background_tasks.get(task_id)
        if not task:
            return []
        
        output_lines = []
        while not task.output_queue.empty():
            try:
                line = task.output_queue.get_nowait()
                output_lines.append(line)
            except queue.Empty:
                break
        
        return output_lines
    
    def kill_task(self, task_id: str) -> bool:
        """
        Kill a background task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if task was killed, False if not found or already stopped
        """
        task = self.background_tasks.get(task_id)
        if not task or not task.is_running:
            return False
        
        logger.warning(f"Killing background task: {task_id}")
        task.process.kill()
        task.status = ProcessStatus.KILLED
        return True
    
    def get_command_history(self, limit: Optional[int] = None) -> List[CommandResult]:
        """
        Get command execution history.
        
        Args:
            limit: Maximum number of recent commands to return
        
        Returns:
            List of CommandResults
        """
        if limit:
            return self.command_history[-limit:]
        return self.command_history.copy()
    
    def clear_history(self):
        """Clear command history."""
        self.command_history.clear()
        logger.info("Command history cleared")
    
    def _stream_output(
        self,
        process: subprocess.Popen,
        timeout: int
    ) -> Tuple[str, str]:
        """
        Stream process output in real-time.
        
        Args:
            process: Process to stream from
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (stdout, stderr)
        """
        stdout_lines = []
        stderr_lines = []
        start_time = time.time()
        
        def read_stream(stream, lines_list):
            """Read from a stream."""
            for line in iter(stream.readline, ''):
                if not line:
                    break
                lines_list.append(line)
                print(line, end='')
        
        # Start reader threads
        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines)
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines)
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait with timeout
        while (time.time() - start_time) < timeout:
            if process.poll() is not None:
                break
            time.sleep(0.1)
        
        # Check for timeout
        if process.poll() is None:
            process.kill()
        
        # Wait for threads to finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        return ''.join(stdout_lines), ''.join(stderr_lines)
    
    def _read_output(self, stream, output_queue: queue.Queue):
        """
        Read output from a stream into a queue.
        
        Args:
            stream: Stream to read from
            output_queue: Queue to put output lines
        """
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                output_queue.put(line.rstrip())
        except Exception as e:
            logger.error(f"Error reading output: {e}")
