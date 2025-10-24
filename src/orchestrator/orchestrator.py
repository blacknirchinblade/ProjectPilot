"""
Orchestrator - Central Workflow Coordination System

This module provides comprehensive workflow orchestration for the multi-agent system:
- Workflow definition and execution
- Task dependency resolution
- Agent task routing and coordination
- Resource management and load balancing
- Error handling and retry logic
- Real-time monitoring and statistics

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import uuid
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
from collections import defaultdict, deque
from loguru import logger

from ..communication.message_bus import MessageBus, MessagePriority
from ..memory.shared_memory import SharedMemory
from ..memory.context_manager import ContextManager
from ..tools.task_tracker import TaskStatus




class WorkflowStatus(Enum):
    """Workflow execution status."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult:
    """
    Result from task execution.
    
    Attributes:
        task_id: Task identifier
        status: Execution status
        output: Task output data
        error: Error message if failed
        start_time: Execution start timestamp
        end_time: Execution end timestamp
        duration: Execution duration in seconds
        agent_id: Agent that executed task
    """
    
    def __init__(
        self,
        task_id: str,
        status: TaskStatus,
        output: Any = None,
        error: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        self.task_id = task_id
        self.status = status
        self.output = output
        self.error = error
        self.agent_id = agent_id
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.duration: Optional[float] = None
    
    def complete(self, output: Any = None, error: Optional[str] = None):
        """Mark result as complete."""
        self.end_time = datetime.now().isoformat()
        if output is not None:
            self.output = output
        if error is not None:
            self.error = error
        
        # Calculate duration
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        self.duration = (end - start).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "agent_id": self.agent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration
        }


class Task:
    """
    Individual task in a workflow.
    
    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        agent_type: Type of agent to execute task
        input_data: Task input data
        dependencies: List of task IDs this task depends on
        priority: Task priority
        status: Current execution status
        result: Task execution result
        metadata: Additional task metadata
    """
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        input_data: Any,
        dependencies: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.agent_type = agent_type
        self.input_data = input_data
        self.dependencies = dependencies or []
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.assigned_agent: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 5
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type,
            "input_data": self.input_data,
            "dependencies": self.dependencies,
            "priority": self.priority.name,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "assigned_agent": self.assigned_agent,
            "retry_count": self.retry_count
        }


class Workflow:
    """
    Workflow definition containing multiple tasks.
    
    Attributes:
        id: Unique workflow identifier
        name: Workflow name
        tasks: Dictionary of task_id -> Task
        status: Current workflow status
        metadata: Additional workflow metadata
    """
    
    def __init__(
        self,
        name: str,
        tasks: Optional[List[Task]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.status = WorkflowStatus.CREATED
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        
        # Add tasks if provided
        if tasks:
            for task in tasks:
                self.add_task(task)
    
    def add_task(self, task: Task):
        """Add task to workflow."""
        self.tasks[task.id] = task
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        completed = {
            tid for tid, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }
        
        ready_tasks = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.is_ready(completed):
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return ready_tasks
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for task in self.tasks.values()
        )
    
    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed."""
        return any(
            task.status == TaskStatus.FAILED
            for task in self.tasks.values()
        )
    
    def get_task_counts(self) -> Dict[str, int]:
        """Get counts of tasks by status."""
        counts = defaultdict(int)
        for task in self.tasks.values():
            counts[task.status.value] += 1
        return dict(counts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "task_counts": self.get_task_counts()
        }


class Orchestrator:
    """
    Central orchestrator for workflow coordination and agent management.
    
    Features:
    - Workflow execution and monitoring
    - Task dependency resolution
    - Agent task routing
    - Load balancing
    - Error handling and retry logic
    - Real-time statistics
    """
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        shared_memory: Optional[SharedMemory] = None,
        context_manager: Optional[ContextManager] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            message_bus: Message bus for agent communication
            shared_memory: Shared memory for state management
            context_manager: Context manager for workflow context
        """
        self.message_bus = message_bus or MessageBus()
        self.shared_memory = shared_memory or SharedMemory()
        self.context_manager = context_manager or ContextManager()
        
        # Workflows
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Set[str] = set()
        
        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.agent_workload: Dict[str, int] = defaultdict(int)
        
        # Task tracking
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.running_tasks: Set[str] = set()
        
        # Statistics
        self.stats = {
            "workflows_created": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "tasks_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
        
        # Callbacks
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.workflow_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("Orchestrator initialized")
    
    # ==================== Agent Registration ====================
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            capabilities: List of capabilities
            metadata: Additional agent metadata
        """
        self.registered_agents[agent_id] = {
            "agent_type": agent_type,
            "capabilities": capabilities or [],
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Update capabilities index
        self.agent_capabilities[agent_type].add(agent_id)
        for capability in (capabilities or []):
            self.agent_capabilities[capability].add(agent_id)
        
        self.agent_workload[agent_id] = 0
        
        logger.info(f"Registered agent {agent_id} (type: {agent_type})")
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Agent {agent_id} not registered")
            return
        
        agent_info = self.registered_agents[agent_id]
        
        # Remove from capabilities index
        agent_type = agent_info["agent_type"]
        self.agent_capabilities[agent_type].discard(agent_id)
        for capability in agent_info["capabilities"]:
            self.agent_capabilities[capability].discard(agent_id)
        
        # Remove agent
        del self.registered_agents[agent_id]
        del self.agent_workload[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
    
    def get_agent_for_task(self, task: Task) -> Optional[str]:
        """
        Find best agent for a task using load balancing.
        
        Args:
            task: Task to assign
        
        Returns:
            Agent ID or None if no suitable agent
        """
        # Get agents with required capability
        candidates = self.agent_capabilities.get(task.agent_type, set())
        
        if not candidates:
            logger.warning(f"No agents available for type '{task.agent_type}'")
            return None
        
        # Select agent with lowest workload
        best_agent = min(
            candidates,
            key=lambda aid: self.agent_workload.get(aid, 0)
        )
        
        return best_agent
    
    # ==================== Workflow Management ====================
    
    def create_workflow(
        self,
        name: str,
        tasks: Optional[List[Task]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            tasks: List of tasks
            metadata: Additional metadata
        
        Returns:
            Workflow ID
        """
        workflow = Workflow(name=name, tasks=tasks, metadata=metadata)
        self.workflows[workflow.id] = workflow
        self.stats["workflows_created"] += 1
        
        # Store in shared memory
        self.shared_memory.store(
            f"workflow:{workflow.id}",
            workflow.to_dict(),
            metadata={"workflow_name": name}
        )
        
        logger.info(f"Created workflow '{name}' (id: {workflow.id[:8]})")
        return workflow.id
    
    def add_task_to_workflow(
        self,
        workflow_id: str,
        task: Task
    ):
        """
        Add task to workflow.
        
        Args:
            workflow_id: Workflow identifier
            task: Task to add
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.CREATED:
            raise ValueError(f"Cannot add task to {workflow.status.value} workflow")
        
        workflow.add_task(task)
        
        # Update in shared memory
        self.shared_memory.store(
            f"workflow:{workflow_id}",
            workflow.to_dict()
        )
        
        logger.debug(f"Added task '{task.name}' to workflow {workflow_id[:8]}")
    
    def start_workflow(self, workflow_id: str):
        """
        Start workflow execution.
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.CREATED:
            raise ValueError(f"Workflow already {workflow.status.value}")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now().isoformat()
        self.active_workflows.add(workflow_id)
        
        # Update in shared memory
        self.shared_memory.store(
            f"workflow:{workflow_id}",
            workflow.to_dict()
        )
        
        # Create workflow context by adding initial context
        self.context_manager.add_to_context(
            agent_id=f"workflow_{workflow_id}",
            content=f"Workflow '{workflow.name}' started",
            metadata={"workflow_id": workflow_id, "workflow_name": workflow.name}
        )
        
        logger.info(f"Started workflow '{workflow.name}' (id: {workflow_id[:8]})")
        
        # Start task execution
        self._execute_workflow(workflow_id)
    
    def _execute_workflow(self, workflow_id: str):
        """Execute workflow by processing ready tasks."""
        workflow = self.workflows[workflow_id]
        
        # Get ready tasks
        ready_tasks = workflow.get_ready_tasks()
        
        if not ready_tasks:
            # Check if workflow is complete
            if workflow.is_complete():
                self._complete_workflow(workflow_id)
            elif workflow.has_failed_tasks():
                self._fail_workflow(workflow_id, "Tasks failed")
            else:
                # Workflow is blocked
                logger.debug(f"Workflow {workflow_id[:8]} waiting for tasks to complete")
            return
        
        # Execute ready tasks
        for task in ready_tasks:
            self._execute_task(workflow_id, task.id)
    
    def _execute_task(self, workflow_id: str, task_id: str):
        """Execute a single task."""
        workflow = self.workflows[workflow_id]
        task = workflow.tasks[task_id]
        
        # Find agent for task
        agent_id = self.get_agent_for_task(task)
        
        if not agent_id:
            task.status = TaskStatus.BLOCKED
            logger.warning(f"No agent available for task '{task.name}'")
            return
        
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = agent_id
        self.task_assignments[task_id] = agent_id
        self.running_tasks.add(task_id)
        self.agent_workload[agent_id] += 1
        
        task.result = TaskResult(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            agent_id=agent_id
        )
        
        # Send task to agent via message bus
        self.message_bus.publish(
            topic=f"agent.{agent_id}.tasks",
            sender="orchestrator",
            recipient=agent_id,
            content={
                "workflow_id": workflow_id,
                "task_id": task_id,
                "task_name": task.name,
                "input_data": task.input_data,
                "metadata": task.metadata
            },
            priority=task.priority,
            metadata={"task_type": task.agent_type}
        )
        
        self.stats["tasks_executed"] += 1
        
        logger.info(f"Executing task '{task.name}' on agent {agent_id}")
    
    def complete_task(
        self,
        workflow_id: str,
        task_id: str,
        output: Any = None,
        error: Optional[str] = None
    ):
        """
        Mark task as completed.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Task identifier
            output: Task output
            error: Error message if failed
        """
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return
        
        workflow = self.workflows[workflow_id]
        
        if task_id not in workflow.tasks:
            logger.error(f"Task {task_id} not found in workflow")
            return
        
        task = workflow.tasks[task_id]
        
        # Update task status
        if error:
            # Check if can retry
            if task.can_retry():
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.result = None  # Reset result for retry
                logger.warning(f"Task '{task.name}' failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                task.status = TaskStatus.FAILED
                task.result.status = TaskStatus.FAILED
                task.result.complete(output=output, error=error)
                self.stats["tasks_failed"] += 1
                logger.error(f"Task '{task.name}' failed permanently: {error}")
        else:
            task.status = TaskStatus.COMPLETED
            task.result.status = TaskStatus.COMPLETED
            task.result.complete(output=output, error=error)
            self.stats["tasks_completed"] += 1
            logger.info(f"Task '{task.name}' completed successfully")
            
            # Update execution time
            if task.result.duration:
                self.stats["total_execution_time"] += task.result.duration
            
            # Store result in shared memory
            self.shared_memory.store(
                f"task_result:{task_id}",
                task.result.to_dict(),
                metadata={"workflow_id": workflow_id}
            )
        
        # Update agent workload (both success and failure)
        if task.assigned_agent:
            self.agent_workload[task.assigned_agent] -= 1
        
        # Remove from running tasks (both success and failure)
        self.running_tasks.discard(task_id)
        
        # Trigger callbacks
        for callback in self.task_callbacks.get(task_id, []):
            try:
                callback(task.result)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")
        
        # Continue workflow execution
        self._execute_workflow(workflow_id)
    
    def _complete_workflow(self, workflow_id: str):
        """Mark workflow as completed."""
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now().isoformat()
        self.active_workflows.discard(workflow_id)
        self.stats["workflows_completed"] += 1
        
        # Update in shared memory
        self.shared_memory.store(
            f"workflow:{workflow_id}",
            workflow.to_dict()
        )
        
        logger.info(f"Workflow '{workflow.name}' completed (id: {workflow_id[:8]})")
        
        # Trigger callbacks
        for callback in self.workflow_callbacks.get(workflow_id, []):
            try:
                callback(workflow)
            except Exception as e:
                logger.error(f"Error in workflow callback: {e}")
    
    def _fail_workflow(self, workflow_id: str, reason: str):
        """Mark workflow as failed."""
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.FAILED
        workflow.completed_at = datetime.now().isoformat()
        workflow.metadata["failure_reason"] = reason
        self.active_workflows.discard(workflow_id)
        self.stats["workflows_failed"] += 1
        
        # Update in shared memory
        self.shared_memory.store(
            f"workflow:{workflow_id}",
            workflow.to_dict()
        )
        
        logger.error(f"Workflow '{workflow.name}' failed: {reason}")
    
    def cancel_workflow(self, workflow_id: str):
        """
        Cancel workflow execution.
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now().isoformat()
        self.active_workflows.discard(workflow_id)
        
        # Cancel running tasks
        for task in workflow.tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.CANCELLED
                self.running_tasks.discard(task.id)
                if task.assigned_agent:
                    self.agent_workload[task.assigned_agent] -= 1
        
        logger.info(f"Cancelled workflow '{workflow.name}'")
    
    def pause_workflow(self, workflow_id: str):
        """
        Pause workflow execution.
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.RUNNING:
            raise ValueError(f"Cannot pause {workflow.status.value} workflow")
        
        workflow.status = WorkflowStatus.PAUSED
        logger.info(f"Paused workflow '{workflow.name}'")
    
    def resume_workflow(self, workflow_id: str):
        """
        Resume paused workflow.
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Cannot resume {workflow.status.value} workflow")
        
        workflow.status = WorkflowStatus.RUNNING
        logger.info(f"Resumed workflow '{workflow.name}'")
        
        # Continue execution
        self._execute_workflow(workflow_id)
    
    # ==================== Queries ====================
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def get_task(self, workflow_id: str, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            return workflow.tasks.get(task_id)
        return None
    
    def get_active_workflows(self) -> List[Workflow]:
        """Get all active workflows."""
        return [
            self.workflows[wid]
            for wid in self.active_workflows
            if wid in self.workflows
        ]
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information."""
        info = self.registered_agents.get(agent_id)
        if info:
            info = dict(info)
            info["workload"] = self.agent_workload[agent_id]
        return info
    
    # ==================== Callbacks ====================
    
    def on_task_complete(self, task_id: str, callback: Callable):
        """
        Register callback for task completion.
        
        Args:
            task_id: Task identifier
            callback: Callback function(TaskResult)
        """
        self.task_callbacks[task_id].append(callback)
    
    def on_workflow_complete(self, workflow_id: str, callback: Callable):
        """
        Register callback for workflow completion.
        
        Args:
            workflow_id: Workflow identifier
            callback: Callback function(Workflow)
        """
        self.workflow_callbacks[workflow_id].append(callback)
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            "active_workflows": len(self.active_workflows),
            "total_workflows": len(self.workflows),
            "registered_agents": len(self.registered_agents),
            "running_tasks": len(self.running_tasks),
            "average_task_time": (
                self.stats["total_execution_time"] / self.stats["tasks_completed"]
                if self.stats["tasks_completed"] > 0 else 0
            )
        }
    
    def get_workflow_statistics(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics for a specific workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {}
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "task_counts": workflow.get_task_counts(),
            "total_tasks": len(workflow.tasks),
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at
        }
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        if agent_id not in self.registered_agents:
            return {}
        
        # Count tasks assigned to this agent
        tasks_assigned = sum(
            1 for task in self.task_assignments.values()
            if task == agent_id
        )
        
        return {
            "agent_id": agent_id,
            "workload": self.agent_workload[agent_id],
            "tasks_assigned": tasks_assigned,
            "agent_info": self.registered_agents[agent_id]
        }
    
    # ==================== Cleanup ====================
    
    def clear_completed_workflows(self):
        """Remove completed workflows from memory."""
        to_remove = []
        for wid, workflow in self.workflows.items():
            if workflow.status in (WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED):
                to_remove.append(wid)
        
        for wid in to_remove:
            del self.workflows[wid]
        
        logger.info(f"Cleared {len(to_remove)} completed workflows")
    
    def reset(self):
        """Reset orchestrator to initial state."""
        self.workflows.clear()
        self.active_workflows.clear()
        self.task_assignments.clear()
        self.running_tasks.clear()
        self.task_callbacks.clear()
        self.workflow_callbacks.clear()
        
        for agent_id in self.agent_workload:
            self.agent_workload[agent_id] = 0
        
        self.stats = {
            "workflows_created": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "tasks_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
        
        logger.warning("Orchestrator reset")
