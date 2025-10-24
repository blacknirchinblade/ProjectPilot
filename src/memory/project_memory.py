"""
Project Memory - Project-Specific State and Context Management

Manages project-level information, file tracking, and incremental
progress for multi-session development workflows.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from loguru import logger
import json


class ProjectMemory:
    """
    Manages project-specific memory and state.
    
    Features:
    - Project metadata tracking
    - File change history
    - Task progress tracking
    - Dependency graph
    - Agent assignments
    - Milestone tracking
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize project memory.
        
        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root)
        
        # Project state
        self.project_info: Dict[str, Any] = {}
        self.files_tracked: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.milestones: Dict[str, Dict[str, Any]] = {}
        
        # Agent assignments
        self.agent_assignments: Dict[str, List[str]] = {}
        
        # Backward compatibility aliases
        self.files = self.files_tracked
        
        logger.info(f"ProjectMemory initialized for: {self.project_root}")
    
    # ==================== Project Information ====================
    
    def set_project_info(self, info: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Set project information.
        
        Args:
            info: Project metadata dictionary
            **kwargs: Project metadata as keyword arguments (name, description, technologies, etc.)
        """
        if info:
            self.project_info.update(info)
        if kwargs:
            self.project_info.update(kwargs)
        self.project_info["last_updated"] = datetime.now().isoformat()
        logger.info("Project info updated")
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get project information."""
        return self.project_info.copy()
    
    # ==================== File Tracking ====================
    
    def track_file(
        self,
        filepath: str,
        status: str = "created",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track file changes.
        
        Args:
            filepath: File path relative to project root
            status: File status (created, modified, deleted)
            metadata: Optional metadata (size, agent_id, etc.)
        """
        if filepath not in self.files_tracked:
            self.files_tracked[filepath] = {
                "created_at": datetime.now().isoformat(),
                "history": []
            }
        
        change = {
            "action": status,  # Keep 'action' for backward compatibility
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.files_tracked[filepath]["history"].append(change)
        self.files_tracked[filepath]["last_status"] = status
        self.files_tracked[filepath]["last_updated"] = datetime.now().isoformat()
        
        logger.debug(f"Tracked file '{filepath}' ({status})")
    
    def get_file_history(self, filepath: str) -> List[Dict[str, Any]]:
        """Get file change history."""
        return self.files_tracked.get(filepath, {}).get("history", [])
    
    def get_tracked_files(
        self,
        status_filter: Optional[str] = None
    ) -> List[str]:
        """
        Get list of tracked files.
        
        Args:
            status_filter: Filter by last status
        
        Returns:
            List of file paths
        """
        files = list(self.files_tracked.keys())
        
        if status_filter:
            files = [
                f for f in files
                if self.files_tracked[f].get("last_status") == status_filter
            ]
        
        return files
    
    def get_files(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked files with their info."""
        result = {}
        for filepath, info in self.files_tracked.items():
            result[filepath] = {
                "status": info.get("last_status", "unknown"),
                "created_at": info.get("created_at"),
                "last_updated": info.get("last_updated"),
                "history": info.get("history", [])
            }
        return result
    
    def get_files_by_status(self, status: str) -> Dict[str, Dict[str, Any]]:
        """Get files filtered by status."""
        all_files = self.get_files()
        return {
            path: info for path, info in all_files.items()
            if info["status"] == status
        }
    
    # ==================== Task Management ====================
    
    def add_task(
        self,
        task_id: str,
        description: str,
        status: str,
        title: Optional[str] = None,
        assigned_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add task to project.
        
        Args:
            task_id: Unique task identifier
            description: Task description
            status: Task status (pending, in_progress, completed, blocked)
            title: Optional task title (defaults to description)
            assigned_to: Agent ID assigned to task
            metadata: Optional metadata
        """
        self.tasks[task_id] = {
            "title": title or description,
            "description": description,
            "status": status,
            "assigned_to": assigned_to,
            "assigned_agent": assigned_to,  # Backward compatibility
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Track agent assignment
        if assigned_to:
            if assigned_to not in self.agent_assignments:
                self.agent_assignments[assigned_to] = []
            self.agent_assignments[assigned_to].append(task_id)
        
        logger.info(f"Added task '{task_id}': {title}")
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        notes: Optional[str] = None
    ):
        """
        Update task status.
        
        Args:
            task_id: Task identifier
            status: New status
            notes: Optional update notes
        """
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            if notes:
                if "notes" not in self.tasks[task_id]:
                    self.tasks[task_id]["notes"] = []
                self.tasks[task_id]["notes"].append({
                    "timestamp": datetime.now().isoformat(),
                    "content": notes
                })
            
            logger.info(f"Updated task '{task_id}' status to '{status}'")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks."""
        return self.tasks.copy()
    
    def assign_agent_to_task(self, task_id: str, agent_id: str):
        """
        Assign agent to task.
        
        Args:
            task_id: Task identifier
            agent_id: Agent identifier
        """
        if task_id in self.tasks:
            self.tasks[task_id]["assigned_to"] = agent_id
            self.tasks[task_id]["assigned_agent"] = agent_id  # Backward compatibility
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # Track agent assignment
            if agent_id not in self.agent_assignments:
                self.agent_assignments[agent_id] = []
            if task_id not in self.agent_assignments[agent_id]:
                self.agent_assignments[agent_id].append(task_id)
            
            logger.info(f"Assigned agent '{agent_id}' to task '{task_id}'")
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all tasks with specific status."""
        return [
            {"id": tid, **task}
            for tid, task in self.tasks.items()
            if task["status"] == status
        ]
    
    def get_agent_tasks(self, agent_id: str) -> List[str]:
        """Get tasks assigned to agent."""
        return self.agent_assignments.get(agent_id, [])
    
    # ==================== Dependencies ====================
    
    def add_dependency(self, task_id: str, depends_on: str) -> bool:
        """
        Add task dependency.
        
        Args:
            task_id: Task that depends on another
            depends_on: Task ID that must be completed first
        
        Returns:
            True if dependency added successfully, False if task doesn't exist
        """
        # Check if both tasks exist
        if task_id not in self.tasks or depends_on not in self.tasks:
            return False
        
        if task_id not in self.dependencies:
            self.dependencies[task_id] = []
        
        if depends_on not in self.dependencies[task_id]:
            self.dependencies[task_id].append(depends_on)
            logger.debug(f"Task '{task_id}' depends on '{depends_on}'")
        
        return True
    
    def get_dependencies(self, task_id: str) -> List[str]:
        """Get task dependencies."""
        return self.dependencies.get(task_id, [])
    
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """Get task dependencies (alias for backward compatibility)."""
        return self.get_dependencies(task_id)
    
    def can_start_task(self, task_id: str) -> bool:
        """
        Check if task can be started (all dependencies completed).
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if task can start
        """
        deps = self.get_dependencies(task_id)
        
        for dep_id in deps:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task["status"] != "completed":
                return False
        
        return True
    
    # ==================== Milestones ====================
    
    def add_milestone(
        self,
        milestone_id: str,
        description: str,
        tasks: List[str],
        title: Optional[str] = None,
        target_date: Optional[str] = None
    ) -> bool:
        """
        Add project milestone.
        
        Args:
            milestone_id: Unique milestone identifier
            description: Milestone description (or title if title not provided)
            tasks: List of task IDs in milestone
            title: Optional milestone title
            target_date: Optional target completion date
        
        Returns:
            True if milestone added successfully, False if invalid tasks
        """
        # Validate that all tasks exist
        for task_id in tasks:
            if task_id not in self.tasks:
                return False
        
        self.milestones[milestone_id] = {
            "title": title or description,
            "description": description,
            "tasks": tasks,
            "target_date": target_date,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Added milestone '{milestone_id}' with {len(tasks)} tasks")
        return True
    
    def get_milestone_progress(self, milestone_id: str) -> Dict[str, Any]:
        """
        Get milestone progress.
        
        Args:
            milestone_id: Milestone identifier
        
        Returns:
            Progress statistics
        """
        milestone = self.milestones.get(milestone_id)
        if not milestone:
            return {}
        
        tasks = milestone["tasks"]
        completed = sum(
            1 for tid in tasks
            if self.get_task(tid) and self.get_task(tid)["status"] == "completed"
        )
        
        return {
            "milestone_id": milestone_id,
            "title": milestone["title"],
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "progress_percent": (completed / len(tasks) * 100) if tasks else 0,
            "is_complete": completed == len(tasks)
        }
    
    # ==================== Statistics ====================
    
    def get_milestones(self) -> Dict[str, Dict[str, Any]]:
        """Get all milestones."""
        return self.milestones.copy()
    
    def get_project_stats(self) -> Dict[str, Any]:
        """Get comprehensive project statistics."""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.get_tasks_by_status("completed"))
        in_progress_tasks = len(self.get_tasks_by_status("in_progress"))
        pending_tasks = len(self.get_tasks_by_status("pending"))
        
        # Task status breakdown
        task_status_breakdown = {}
        for task in self.tasks.values():
            status = task["status"]
            task_status_breakdown[status] = task_status_breakdown.get(status, 0) + 1
        
        return {
            "project_info": self.project_info,
            "total_files": len(self.files_tracked),
            "files_tracked": len(self.files_tracked),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": pending_tasks,
            "task_status_breakdown": task_status_breakdown,
            "completion_percent": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "total_milestones": len(self.milestones),
            "active_agents": len(self.agent_assignments)
        }
    
    # ==================== Persistence ====================
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        """
        Save project state to file.
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not filepath:
            filepath = str(self.project_root / "project_state.json")
        
        state = {
            "project_info": self.project_info,
            "files_tracked": self.files_tracked,
            "tasks": self.tasks,
            "dependencies": self.dependencies,
            "milestones": self.milestones,
            "agent_assignments": self.agent_assignments,
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Project state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save project state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load project state from file.
        
        Args:
            filepath: Path to state file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.project_info = state.get("project_info", {})
            self.files_tracked = state.get("files_tracked", {})
            self.tasks = state.get("tasks", {})
            self.dependencies = state.get("dependencies", {})
            self.milestones = state.get("milestones", {})
            self.agent_assignments = state.get("agent_assignments", {})
            
            logger.info(f"Project state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load project state: {e}")
            return False
