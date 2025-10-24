"""
Git Manager for Setup Automation Tool.
Manages git repository initialization and operations.
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from loguru import logger

from src.tools.data_structures import GitResult


class GitManager:
    """Manages git repository operations."""
    
    def __init__(self, project_path: Path):
        """
        Initialize git manager.
        
        Args:
            project_path: Path to the project root
        """
        self.project_path = Path(project_path)
        self.git_available = self._check_git_available()
    
    def _check_git_available(self) -> bool:
        """Check if git is available."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 30
    ) -> Tuple[int, str, str]:
        """
        Run a git command.
        
        Args:
            command: Command and arguments as list
            cwd: Working directory
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return 1, "", str(e)
    
    def initialize_repository(
        self,
        initial_branch: str = "main",
        initial_commit: bool = True,
        commit_message: str = "Initial commit"
    ) -> GitResult:
        """
        Initialize a git repository.
        
        Args:
            initial_branch: Name of the initial branch
            initial_commit: Whether to create an initial commit
            commit_message: Message for initial commit
            
        Returns:
            GitResult with initialization status
        """
        start_time = datetime.now()
        
        if not self.git_available:
            error = "Git is not available on this system"
            logger.error(error)
            return GitResult(
                success=False,
                repo_path=None,
                branch=None,
                remote=None,
                commits=[],
                message=error,
                error=error,
                duration=0.0
            )
        
        logger.info(f"Initializing git repository at {self.project_path}")
        
        try:
            # Check if already a git repo
            if (self.project_path / ".git").exists():
                message = "Git repository already exists"
                logger.warning(message)
                return GitResult(
                    success=False,
                    repo_path=str(self.project_path),
                    branch=None,
                    remote=None,
                    commits=[],
                    message=message,
                    error="Repository already initialized",
                    duration=0.0
                )
            
            # Initialize repository
            code, stdout, stderr = self._run_command(["git", "init"])
            
            if code != 0:
                error = f"Failed to initialize git: {stderr}"
                logger.error(error)
                duration = (datetime.now() - start_time).total_seconds()
                return GitResult(
                    success=False,
                    repo_path=None,
                    branch=None,
                    remote=None,
                    commits=[],
                    message=error,
                    error=stderr,
                    duration=duration
                )
            
            logger.success("Initialized git repository")
            
            # Set initial branch name
            if initial_branch != "master":
                code, _, stderr = self._run_command(
                    ["git", "branch", "-M", initial_branch]
                )
                if code != 0:
                    logger.warning(f"Failed to rename branch: {stderr}")
            
            commits = []
            
            # Create initial commit if requested
            if initial_commit:
                # Stage all files
                code, _, stderr = self._run_command(["git", "add", "."])
                if code != 0:
                    logger.warning(f"Failed to stage files: {stderr}")
                else:
                    # Commit
                    code, stdout, stderr = self._run_command(
                        ["git", "commit", "-m", commit_message]
                    )
                    if code == 0:
                        commits.append(commit_message)
                        logger.success(f"Created initial commit: {commit_message}")
                    else:
                        logger.warning(f"Failed to create initial commit: {stderr}")
            
            duration = (datetime.now() - start_time).total_seconds()
            message = f"Initialized git repository in {duration:.2f}s"
            logger.info(message)
            
            return GitResult(
                success=True,
                repo_path=str(self.project_path),
                branch=initial_branch,
                remote=None,
                commits=commits,
                message=message,
                error=None,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error = f"Exception initializing git: {e}"
            logger.error(error)
            return GitResult(
                success=False,
                repo_path=None,
                branch=None,
                remote=None,
                commits=[],
                message=error,
                error=str(e),
                duration=duration
            )
    
    def add_remote(
        self,
        name: str = "origin",
        url: str = ""
    ) -> GitResult:
        """
        Add a remote repository.
        
        Args:
            name: Name of the remote (e.g., "origin")
            url: URL of the remote repository
            
        Returns:
            GitResult with operation status
        """
        start_time = datetime.now()
        
        if not url:
            error = "Remote URL is required"
            logger.error(error)
            return GitResult(
                success=False,
                repo_path=str(self.project_path),
                branch=None,
                remote=None,
                commits=[],
                message=error,
                error=error,
                duration=0.0
            )
        
        logger.info(f"Adding remote '{name}' -> {url}")
        
        try:
            code, stdout, stderr = self._run_command(
                ["git", "remote", "add", name, url]
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if code == 0:
                message = f"Added remote '{name}' in {duration:.2f}s"
                logger.success(message)
                return GitResult(
                    success=True,
                    repo_path=str(self.project_path),
                    branch=None,
                    remote=url,
                    commits=[],
                    message=message,
                    error=None,
                    duration=duration
                )
            else:
                error = f"Failed to add remote: {stderr}"
                logger.error(error)
                return GitResult(
                    success=False,
                    repo_path=str(self.project_path),
                    branch=None,
                    remote=None,
                    commits=[],
                    message=error,
                    error=stderr,
                    duration=duration
                )
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error = f"Exception adding remote: {e}"
            logger.error(error)
            return GitResult(
                success=False,
                repo_path=str(self.project_path),
                branch=None,
                remote=None,
                commits=[],
                message=error,
                error=str(e),
                duration=duration
            )
    
    def commit_all(
        self,
        message: str,
        add_all: bool = True
    ) -> GitResult:
        """
        Commit all changes.
        
        Args:
            message: Commit message
            add_all: Whether to stage all files first
            
        Returns:
            GitResult with commit status
        """
        start_time = datetime.now()
        
        logger.info(f"Committing changes: {message}")
        
        try:
            commits = []
            
            # Stage files if requested
            if add_all:
                code, _, stderr = self._run_command(["git", "add", "."])
                if code != 0:
                    error = f"Failed to stage files: {stderr}"
                    logger.error(error)
                    duration = (datetime.now() - start_time).total_seconds()
                    return GitResult(
                        success=False,
                        repo_path=str(self.project_path),
                        branch=None,
                        remote=None,
                        commits=[],
                        message=error,
                        error=stderr,
                        duration=duration
                    )
            
            # Commit
            code, stdout, stderr = self._run_command(
                ["git", "commit", "-m", message]
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if code == 0:
                commits.append(message)
                result_message = f"Committed changes in {duration:.2f}s"
                logger.success(result_message)
                return GitResult(
                    success=True,
                    repo_path=str(self.project_path),
                    branch=None,
                    remote=None,
                    commits=commits,
                    message=result_message,
                    error=None,
                    duration=duration
                )
            else:
                # Check if it's "nothing to commit"
                if "nothing to commit" in stderr.lower() or "nothing to commit" in stdout.lower():
                    message = "No changes to commit"
                    logger.info(message)
                    return GitResult(
                        success=True,
                        repo_path=str(self.project_path),
                        branch=None,
                        remote=None,
                        commits=[],
                        message=message,
                        error=None,
                        duration=duration
                    )
                else:
                    error = f"Failed to commit: {stderr}"
                    logger.error(error)
                    return GitResult(
                        success=False,
                        repo_path=str(self.project_path),
                        branch=None,
                        remote=None,
                        commits=[],
                        message=error,
                        error=stderr,
                        duration=duration
                    )
                    
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error = f"Exception committing: {e}"
            logger.error(error)
            return GitResult(
                success=False,
                repo_path=str(self.project_path),
                branch=None,
                remote=None,
                commits=[],
                message=error,
                error=str(e),
                duration=duration
            )
    
    def get_current_branch(self) -> Optional[str]:
        """
        Get the current branch name.
        
        Returns:
            Branch name or None
        """
        try:
            code, stdout, _ = self._run_command(
                ["git", "branch", "--show-current"]
            )
            if code == 0:
                return stdout.strip()
            return None
        except Exception:
            return None
    
    def get_remote_url(self, remote_name: str = "origin") -> Optional[str]:
        """
        Get the URL of a remote.
        
        Args:
            remote_name: Name of the remote
            
        Returns:
            Remote URL or None
        """
        try:
            code, stdout, _ = self._run_command(
                ["git", "remote", "get-url", remote_name]
            )
            if code == 0:
                return stdout.strip()
            return None
        except Exception:
            return None
    
    def is_repository(self) -> bool:
        """
        Check if the current directory is a git repository.
        
        Returns:
            True if it's a git repository, False otherwise
        """
        return (self.project_path / ".git").exists()
    
    def get_status(self) -> str:
        """
        Get git status.
        
        Returns:
            Status output as string
        """
        try:
            code, stdout, _ = self._run_command(["git", "status", "--short"])
            if code == 0:
                return stdout
            return ""
        except Exception:
            return ""
    
    def configure_user(
        self,
        name: str,
        email: str,
        global_config: bool = False
    ) -> GitResult:
        """
        Configure git user name and email.
        
        Args:
            name: User name
            email: User email
            global_config: Whether to set globally or just for this repo
            
        Returns:
            GitResult with configuration status
        """
        start_time = datetime.now()
        
        logger.info(f"Configuring git user: {name} <{email}>")
        
        try:
            scope = "--global" if global_config else "--local"
            
            # Set user name
            code1, _, stderr1 = self._run_command(
                ["git", "config", scope, "user.name", name]
            )
            
            # Set user email
            code2, _, stderr2 = self._run_command(
                ["git", "config", scope, "user.email", email]
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if code1 == 0 and code2 == 0:
                message = f"Configured git user in {duration:.2f}s"
                logger.success(message)
                return GitResult(
                    success=True,
                    repo_path=str(self.project_path),
                    branch=None,
                    remote=None,
                    commits=[],
                    message=message,
                    error=None,
                    duration=duration
                )
            else:
                error = f"Failed to configure user: {stderr1 or stderr2}"
                logger.error(error)
                return GitResult(
                    success=False,
                    repo_path=str(self.project_path),
                    branch=None,
                    remote=None,
                    commits=[],
                    message=error,
                    error=error,
                    duration=duration
                )
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error = f"Exception configuring user: {e}"
            logger.error(error)
            return GitResult(
                success=False,
                repo_path=str(self.project_path),
                branch=None,
                remote=None,
                commits=[],
                message=error,
                error=str(e),
                duration=duration
            )
