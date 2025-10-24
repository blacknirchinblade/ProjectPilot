

"""
Task Tracker - Persistent task management for project generation.

Tracks all tasks from planning to completion with dependency management.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
import json
from loguru import logger


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskType(Enum):
    """Task type."""
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    REVIEW = "review"
    REFACTORING = "refactoring"


@dataclass
class Task:
    """A single tracked task."""
    
    id: str                       # Unique identifier
    type: TaskType                # Task type
    target: str                   # Target (e.g., file path)
    description: str              # Human-readable description
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    estimated_lines: Optional[int] = None
    priority: int = 1               # Lower = higher priority
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data['type'] = self.type.value
        data['status'] = self.status.value
        # Convert datetimes to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data
    
    @staticmethod
    def from_dict(data: Dict) -> 'Task':
        """Create Task from dict."""
        # Convert strings to enums
        data['type'] = TaskType(data['type'])
        data['status'] = TaskStatus(data['status'])
        # Convert ISO format to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['started_at'] = datetime.fromisoformat(data['started_at']) if data.get('started_at') else None
        data['completed_at'] = datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None
        return Task(**data)


class TaskTracker:
    """
    Tracks all tasks with dependency management and persistence.
    
    Features:
    - Persistent task list (survives interruptions)
    - Dependency management (build in correct order)
    - Progress tracking
    - Resume capability
    - Iteration support (resetting tasks)
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.tasks_file = project_path / "TODO.json"
        self.tasks: List[Task] = []
        
        # Try to load existing tasks
        if self.tasks_file.exists():
            self.load()
            logger.info(f"Loaded {len(self.tasks)} tasks from {self.tasks_file}")
        else:
            logger.info(f"Initialized new task tracker for {project_path}")
    
    def add_task(self, task: Task) -> None:
        """Add a new task."""
        self.tasks.append(task)
        logger.debug(f"Added task: {task.id} - {task.description}")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_next_task(self) -> Optional[Task]:
        """
        Get next task to execute.
        Returns task whose dependencies are all complete.
        """
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        
        if not pending_tasks:
            return None
        
        # Find task with all dependencies complete
        for task in sorted(pending_tasks, key=lambda t: t.priority):
            if self._dependencies_complete(task):
                return task
        
        return None
    
    def _dependencies_complete(self, task: Task) -> bool:
        """Check if all dependencies are complete."""
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETE:
                return False
        return True
    
    def mark_started(self, task_id: str) -> None:
        """Mark task as started."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            self.save()
            logger.info(f"Task started: {task.id} - {task.description}")
    
    def mark_complete(self, task_id: str, result: Optional[Dict] = None) -> None:
        """Mark task as complete."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETE
            task.completed_at = datetime.now()
            task.result = result
            self.save()
            logger.success(f"Task complete: {task.id} - {task.description}")
    
    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            self.save()
            logger.error(f"Task failed: {task.id} - {error}")

    # --- THIS IS THE FIX ---
    def reset_task(self, task: Task) -> None:
        """Resets a single task to pending state."""
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.result = None
        task.error = None

    def reset_task_status(self, task_ids: Optional[List[str]] = None) -> None:
        """
        Resets the status of specified tasks (or all tasks) to PENDING.
        This is used for iterative refinement.
        
        Args:
            task_ids: A list of task IDs to reset. If None, resets all tasks.
        """
        if task_ids:
            logger.info(f"Resetting {len(task_ids)} tasks for iteration...")
            for task_id in task_ids:
                task = self.get_task(task_id)
                if task:
                    self.reset_task(task)
        else:
            logger.info("Resetting all tasks for iteration...")
            for task in self.tasks:
                self.reset_task(task)
        
        self.save() # Save the reset state
    # --- END FIX ---
            
    def save(self) -> None:
        """Save tasks to JSON."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "project_path": str(self.project_path),
            "last_updated": datetime.now().isoformat(),
            "tasks": [task.to_dict() for task in self.tasks]
        }
        
        try:
            with open(self.tasks_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.tasks)} tasks to {self.tasks_file}")
        except PermissionError:
            logger.error(f"Permission denied: Could not write to {self.tasks_file}")
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")
    
    def load(self) -> None:
        """Load tasks from JSON."""
        if not self.tasks_file.exists():
            logger.warning(f"Tasks file not found: {self.tasks_file}")
            return
        
        try:
            with open(self.tasks_file) as f:
                data = json.load(f)
            
            self.tasks = [Task.from_dict(task_data) for task_data in data["tasks"]]
            logger.success(f"Loaded {len(self.tasks)} tasks from {self.tasks_file}")
        
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            self.tasks = [] # Start fresh if file is corrupt
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress statistics."""
        total = len(self.tasks)
        if total == 0:
            return {"total": 0, "completed": 0, "percentage": 0, "in_progress": 0, "failed": 0, "pending": 0}
        
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETE)
        in_progress = sum(1 for t in self.tasks if t.status == TaskStatus.IN_PROGRESS)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "pending": total - completed - in_progress - failed,
            "percentage": (completed / total) * 100 if total > 0 else 0
        }
    
    def print_progress(self) -> None:
        """Print progress summary."""
        progress = self.get_progress()
        
        print("\n" + "=" * 60)
        print("📊 TASK PROGRESS")
        print("=" * 60)
        print(f"Total tasks: {progress['total']}")
        print(f"✅ Completed: {progress['completed']}")
        print(f"🔄 In progress: {progress['in_progress']}")
        print(f"❌ Failed: {progress['failed']}")
        print(f"⏳ Pending: {progress['pending']}")
        print(f"\nProgress: {progress['percentage']:.1f}%")
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * progress['percentage'] / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"[{bar}]")
        print("=" * 60 + "\n")
    
    def can_resume(self) -> bool:
        """Check if there are tasks to resume."""
        return any(t.status == TaskStatus.PENDING for t in self.tasks)
    
    def get_resume_info(self) -> Dict[str, Any]:
        """Get information about resumable tasks."""
        if not self.can_resume():
            return {"can_resume": False}
        
        progress = self.get_progress()
        next_task = self.get_next_task()
        
        return {
            "can_resume": True,
            "progress": progress,
            "next_task": {
                "id": next_task.id,
                "description": next_task.description,
                "type": next_task.type.value
            } if next_task else None
        }


def create_tasks_from_architecture(architecture, project_path) -> TaskTracker:
    """
    Create TaskTracker with tasks for each file in architecture.
    
    Args:
        architecture: ProjectArchitecture from DynamicArchitectAgent
        project_path: Project directory (Path or str)
    
    Returns:
        TaskTracker with all code generation tasks
    """
    if isinstance(project_path, str):
        project_path = Path(project_path)
    
    tracker = TaskTracker(project_path)
    
    # Create tasks for each file
    task_map = {}  # filepath → task_id
    
    # Use the flat list of file specs from the architecture object
    all_file_specs = architecture.file_specs
    
    for spec in all_file_specs:
        task_id = f"gen_{spec.path.replace('/', '_').replace('.', '_')}"
        
        # Find dependency task IDs
        dep_task_ids = []
        for dep in spec.dependencies:
            if dep.startswith("src."):
                # Internal dependency - convert to filepath
                dep_filepath = dep.replace(".", "/") + ".py"
                dep_task_id = f"gen_{dep_filepath.replace('/', '_').replace('.', '_')}"
                
                # We need to find if this dep_task_id actually exists in our plan
                # We can't just check task_map because we might not have processed it yet
                if any(s.path == dep_filepath for s in all_file_specs):
                     dep_task_ids.append(dep_task_id)
                else:
                    logger.warning(f"Task {task_id} has a dependency '{dep}' ({dep_filepath}) that is not in the architecture plan. Ignoring.")

        task = Task(
            id=task_id,
            type=TaskType.CODE_GENERATION,
            target=spec.path,
            description=f"Generate {spec.path}: {spec.purpose}",
            dependencies=dep_task_ids,
            estimated_lines=spec.estimated_lines,
            priority=spec.priority
        )
        
        tracker.add_task(task)
        task_map[spec.path] = task_id
    
    tracker.save()
    logger.success(f"Created {len(tracker.tasks)} tasks from architecture")
    
    return tracker


def load_tracker(todo_path) -> TaskTracker:
    """
    Load existing TaskTracker from TODO.json file.
    
    Args:
        todo_path: Path to TODO.json file or project directory (Path or str)
    
    Returns:
        TaskTracker loaded from file
    
    Raises:
        FileNotFoundError: If TODO.json doesn't exist
    """
    if isinstance(todo_path, str):
        todo_path = Path(todo_path)
    
    if todo_path.is_dir():
        project_path = todo_path
    else:
        project_path = todo_path.parent
    
    tracker = TaskTracker(project_path)
    
    if not tracker.tasks_file.exists():
         raise FileNotFoundError(f"Tasks file not found: {tracker.tasks_file}")

    if not tracker.tasks:
        # This can happen if load() fails silently or file is empty
        raise FileNotFoundError(f"No tasks found in {tracker.tasks_file}")
    
    logger.info(f"Loaded tracker with {len(tracker.tasks)} tasks")
    return tracker