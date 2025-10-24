"""
GitHub Tool - Interact with GitHub API programmatically.

Provides comprehensive GitHub operations:
- Repository management (create, delete, info)
- Branch operations (create, delete, list)
- Commit operations (create, get, list)
- Pull Request management (create, list, merge)
- Issue management (create, update, close, list)
- File operations (read, write, delete)
- Release management
- Webhook management

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import base64


class PRState(Enum):
    """Pull request states."""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class IssueState(Enum):
    """Issue states."""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


@dataclass
class Repository:
    """Represents a GitHub repository."""
    name: str
    owner: str
    full_name: str
    description: Optional[str]
    private: bool
    url: str
    clone_url: str
    default_branch: str
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    
    def __str__(self) -> str:
        """String representation."""
        visibility = "Private" if self.private else "Public"
        return (
            f"{visibility} Repository: {self.full_name}\n"
            f"  Description: {self.description or 'None'}\n"
            f"  Stars: {self.stars}, Forks: {self.forks}, Issues: {self.open_issues}\n"
            f"  Default Branch: {self.default_branch}\n"
            f"  Clone URL: {self.clone_url}"
        )


@dataclass
class PullRequest:
    """Represents a GitHub pull request."""
    number: int
    title: str
    state: str
    author: str
    created_at: str
    updated_at: str
    url: str
    head_branch: str
    base_branch: str
    mergeable: Optional[bool] = None
    merged: bool = False
    
    def __str__(self) -> str:
        """String representation."""
        status = "Merged" if self.merged else self.state.capitalize()
        return (
            f"PR #{self.number}: {self.title}\n"
            f"  Status: {status}\n"
            f"  Author: {self.author}\n"
            f"  Branches: {self.head_branch} → {self.base_branch}\n"
            f"  Created: {self.created_at}"
        )


@dataclass
class Issue:
    """Represents a GitHub issue."""
    number: int
    title: str
    state: str
    author: str
    created_at: str
    updated_at: str
    url: str
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    comments: int = 0
    
    def __str__(self) -> str:
        """String representation."""
        labels_str = ", ".join(self.labels) if self.labels else "None"
        return (
            f"Issue #{self.number}: {self.title}\n"
            f"  State: {self.state.capitalize()}\n"
            f"  Author: {self.author}\n"
            f"  Labels: {labels_str}\n"
            f"  Comments: {self.comments}"
        )


class GitHubTool:
    """
    Tool for interacting with GitHub API.
    
    Features:
    - Repository operations
    - Branch management
    - Pull request management
    - Issue management
    - File operations
    - Commit operations
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: str, owner: Optional[str] = None, repo: Optional[str] = None):
        """
        Initialize GitHub tool.
        
        Args:
            token: GitHub personal access token
            owner: Default repository owner (optional)
            repo: Default repository name (optional)
        """
        self.token = token
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        logger.info(f"GitHubTool initialized{' for ' + owner + '/' + repo if owner and repo else ''}")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make API request to GitHub.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            endpoint: API endpoint
            data: Request body data
            params: URL parameters
        
        Returns:
            Response data
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                params=params
            )
            
            response.raise_for_status()
            
            # Some endpoints return 204 No Content
            if response.status_code == 204:
                return {}
            
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"GitHub API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('message', '')}"
            except:
                pass
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    # ==================== Repository Operations ====================
    
    def create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = True
    ) -> Repository:
        """
        Create a new repository.
        
        Args:
            name: Repository name
            description: Repository description
            private: Whether repository is private
            auto_init: Initialize with README
        
        Returns:
            Repository object
        """
        data = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": auto_init
        }
        
        response = self._request("POST", "user/repos", data=data)
        
        logger.info(f"Created repository: {response['full_name']}")
        
        return self._parse_repository(response)
    
    def get_repository(self, owner: Optional[str] = None, repo: Optional[str] = None) -> Repository:
        """
        Get repository information.
        
        Args:
            owner: Repository owner (uses default if None)
            repo: Repository name (uses default if None)
        
        Returns:
            Repository object
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        response = self._request("GET", f"repos/{owner}/{repo}")
        
        return self._parse_repository(response)
    
    def delete_repository(self, owner: Optional[str] = None, repo: Optional[str] = None) -> bool:
        """
        Delete a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
        
        Returns:
            True if deleted successfully
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        self._request("DELETE", f"repos/{owner}/{repo}")
        
        logger.info(f"Deleted repository: {owner}/{repo}")
        return True
    
    def list_repositories(self, username: Optional[str] = None) -> List[Repository]:
        """
        List repositories for a user.
        
        Args:
            username: Username (uses authenticated user if None)
        
        Returns:
            List of Repository objects
        """
        if username:
            endpoint = f"users/{username}/repos"
        else:
            endpoint = "user/repos"
        
        response = self._request("GET", endpoint)
        
        return [self._parse_repository(repo) for repo in response]
    
    # ==================== Branch Operations ====================
    
    def create_branch(
        self,
        branch_name: str,
        from_branch: str = "main",
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new branch.
        
        Args:
            branch_name: New branch name
            from_branch: Source branch to branch from
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Branch information
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        # Get reference SHA from source branch
        ref_response = self._request(
            "GET",
            f"repos/{owner}/{repo}/git/ref/heads/{from_branch}"
        )
        sha = ref_response["object"]["sha"]
        
        # Create new branch
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": sha
        }
        
        response = self._request(
            "POST",
            f"repos/{owner}/{repo}/git/refs",
            data=data
        )
        
        logger.info(f"Created branch: {branch_name} from {from_branch}")
        return response
    
    def delete_branch(
        self,
        branch_name: str,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> bool:
        """
        Delete a branch.
        
        Args:
            branch_name: Branch name to delete
            owner: Repository owner
            repo: Repository name
        
        Returns:
            True if deleted successfully
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        self._request(
            "DELETE",
            f"repos/{owner}/{repo}/git/refs/heads/{branch_name}"
        )
        
        logger.info(f"Deleted branch: {branch_name}")
        return True
    
    def list_branches(
        self,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List repository branches.
        
        Args:
            owner: Repository owner
            repo: Repository name
        
        Returns:
            List of branch information
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        response = self._request("GET", f"repos/{owner}/{repo}/branches")
        
        return response
    
    # ==================== Pull Request Operations ====================
    
    def create_pull_request(
        self,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> PullRequest:
        """
        Create a pull request.
        
        Args:
            title: PR title
            head: Head branch (source)
            base: Base branch (target)
            body: PR description
            owner: Repository owner
            repo: Repository name
        
        Returns:
            PullRequest object
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body or ""
        }
        
        response = self._request(
            "POST",
            f"repos/{owner}/{repo}/pulls",
            data=data
        )
        
        logger.info(f"Created PR #{response['number']}: {title}")
        
        return self._parse_pull_request(response)
    
    def list_pull_requests(
        self,
        state: PRState = PRState.OPEN,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> List[PullRequest]:
        """
        List pull requests.
        
        Args:
            state: PR state (OPEN, CLOSED, ALL)
            owner: Repository owner
            repo: Repository name
        
        Returns:
            List of PullRequest objects
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        params = {"state": state.value}
        
        response = self._request(
            "GET",
            f"repos/{owner}/{repo}/pulls",
            params=params
        )
        
        return [self._parse_pull_request(pr) for pr in response]
    
    def merge_pull_request(
        self,
        pr_number: int,
        commit_message: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge a pull request.
        
        Args:
            pr_number: PR number
            commit_message: Merge commit message
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Merge result
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        data = {}
        if commit_message:
            data["commit_message"] = commit_message
        
        response = self._request(
            "PUT",
            f"repos/{owner}/{repo}/pulls/{pr_number}/merge",
            data=data
        )
        
        logger.info(f"Merged PR #{pr_number}")
        return response
    
    # ==================== Issue Operations ====================
    
    def create_issue(
        self,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> Issue:
        """
        Create an issue.
        
        Args:
            title: Issue title
            body: Issue body
            labels: Issue labels
            assignees: Issue assignees
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Issue object
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        data = {
            "title": title,
            "body": body or ""
        }
        
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        
        response = self._request(
            "POST",
            f"repos/{owner}/{repo}/issues",
            data=data
        )
        
        logger.info(f"Created issue #{response['number']}: {title}")
        
        return self._parse_issue(response)
    
    def list_issues(
        self,
        state: IssueState = IssueState.OPEN,
        labels: Optional[List[str]] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> List[Issue]:
        """
        List issues.
        
        Args:
            state: Issue state (OPEN, CLOSED, ALL)
            labels: Filter by labels
            owner: Repository owner
            repo: Repository name
        
        Returns:
            List of Issue objects
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        params = {"state": state.value}
        if labels:
            params["labels"] = ",".join(labels)
        
        response = self._request(
            "GET",
            f"repos/{owner}/{repo}/issues",
            params=params
        )
        
        # Filter out pull requests (they appear in issues endpoint)
        issues = [self._parse_issue(issue) for issue in response if "pull_request" not in issue]
        
        return issues
    
    def close_issue(
        self,
        issue_number: int,
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> Issue:
        """
        Close an issue.
        
        Args:
            issue_number: Issue number
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Updated Issue object
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        data = {"state": "closed"}
        
        response = self._request(
            "PATCH",
            f"repos/{owner}/{repo}/issues/{issue_number}",
            data=data
        )
        
        logger.info(f"Closed issue #{issue_number}")
        
        return self._parse_issue(response)
    
    # ==================== File Operations ====================
    
    def get_file_content(
        self,
        file_path: str,
        branch: str = "main",
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> str:
        """
        Get file content from repository.
        
        Args:
            file_path: Path to file in repository
            branch: Branch name
            owner: Repository owner
            repo: Repository name
        
        Returns:
            File content as string
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        params = {"ref": branch}
        
        response = self._request(
            "GET",
            f"repos/{owner}/{repo}/contents/{file_path}",
            params=params
        )
        
        # Decode base64 content
        content = base64.b64decode(response["content"]).decode("utf-8")
        
        logger.info(f"Retrieved file: {file_path}")
        return content
    
    def create_or_update_file(
        self,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str = "main",
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create or update a file in repository.
        
        Args:
            file_path: Path to file in repository
            content: File content
            commit_message: Commit message
            branch: Branch name
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Commit information
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        # Encode content to base64
        content_bytes = content.encode("utf-8")
        content_base64 = base64.b64encode(content_bytes).decode("utf-8")
        
        data = {
            "message": commit_message,
            "content": content_base64,
            "branch": branch
        }
        
        # Check if file exists to get SHA
        try:
            existing = self._request(
                "GET",
                f"repos/{owner}/{repo}/contents/{file_path}",
                params={"ref": branch}
            )
            data["sha"] = existing["sha"]
            action = "Updated"
        except:
            action = "Created"
        
        response = self._request(
            "PUT",
            f"repos/{owner}/{repo}/contents/{file_path}",
            data=data
        )
        
        logger.info(f"{action} file: {file_path}")
        return response
    
    def delete_file(
        self,
        file_path: str,
        commit_message: str,
        branch: str = "main",
        owner: Optional[str] = None,
        repo: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a file from repository.
        
        Args:
            file_path: Path to file in repository
            commit_message: Commit message
            branch: Branch name
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Commit information
        """
        owner = owner or self.owner
        repo = repo or self.repo
        
        if not owner or not repo:
            raise ValueError("Owner and repo must be specified")
        
        # Get file SHA
        existing = self._request(
            "GET",
            f"repos/{owner}/{repo}/contents/{file_path}",
            params={"ref": branch}
        )
        
        data = {
            "message": commit_message,
            "sha": existing["sha"],
            "branch": branch
        }
        
        response = self._request(
            "DELETE",
            f"repos/{owner}/{repo}/contents/{file_path}",
            data=data
        )
        
        logger.info(f"Deleted file: {file_path}")
        return response
    
    # ==================== Helper Methods ====================
    
    def _parse_repository(self, data: Dict[str, Any]) -> Repository:
        """Parse repository data into Repository object."""
        return Repository(
            name=data["name"],
            owner=data["owner"]["login"],
            full_name=data["full_name"],
            description=data.get("description"),
            private=data["private"],
            url=data["html_url"],
            clone_url=data["clone_url"],
            default_branch=data["default_branch"],
            stars=data["stargazers_count"],
            forks=data["forks_count"],
            open_issues=data["open_issues_count"]
        )
    
    def _parse_pull_request(self, data: Dict[str, Any]) -> PullRequest:
        """Parse pull request data into PullRequest object."""
        return PullRequest(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            author=data["user"]["login"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            url=data["html_url"],
            head_branch=data["head"]["ref"],
            base_branch=data["base"]["ref"],
            mergeable=data.get("mergeable"),
            merged=data.get("merged", False)
        )
    
    def _parse_issue(self, data: Dict[str, Any]) -> Issue:
        """Parse issue data into Issue object."""
        return Issue(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            author=data["user"]["login"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            url=data["html_url"],
            labels=[label["name"] for label in data.get("labels", [])],
            assignees=[assignee["login"] for assignee in data.get("assignees", [])],
            comments=data.get("comments", 0)
        )
