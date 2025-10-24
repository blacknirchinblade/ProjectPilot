"""
Context Manager - Intelligent Context Management and Summarization

Manages context windows, handles context overflow, and provides
intelligent context compression and retrieval for agents.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class ContextManager:
    """
    Intelligent context management for agents.
    
    Features:
    - Dynamic context windowing
    - Automatic context summarization
    - Priority-based context selection
    - Context compression
    - Multi-agent context isolation
    """
    
    def __init__(
        self,
        max_context_tokens: int = 8000,
        summarization_threshold: float = 0.8,
        enable_compression: bool = True
    ):
        """
        Initialize context manager.
        
        Args:
            max_context_tokens: Maximum context size in tokens
            summarization_threshold: When to trigger summarization (0-1)
            enable_compression: Enable context compression
        """
        self.max_context_tokens = max_context_tokens
        self.summarization_threshold = summarization_threshold
        self.enable_compression = enable_compression
        
        # Context storage
        self.contexts: Dict[str, List[Dict[str, Any]]] = {}
        self.summaries: Dict[str, str] = {}
        self.priorities: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"ContextManager initialized (max_tokens={max_context_tokens})")
    
    def add_to_context(
        self,
        agent_id: str,
        content: str,
        content_type: str = "text",
        priority: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add content to agent context.
        
        Args:
            agent_id: Agent identifier
            content: Content to add
            content_type: Type of content (text, code, data)
            priority: Priority score (0-1, higher = more important)
            metadata: Optional metadata
        """
        if agent_id not in self.contexts:
            self.contexts[agent_id] = []
            self.priorities[agent_id] = {}
        
        context_item = {
            "content": content,
            "type": content_type,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        item_id = f"{agent_id}_{len(self.contexts[agent_id])}"
        self.contexts[agent_id].append(context_item)
        self.priorities[agent_id][item_id] = priority
        
        # Check if summarization needed
        if self._should_summarize(agent_id):
            self._summarize_context(agent_id)
        
        logger.debug(f"Added context for agent '{agent_id}' (priority={priority})")
    
    def get_context(
        self,
        agent_id: str,
        max_items: Optional[int] = None,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Get agent context.
        
        Args:
            agent_id: Agent identifier
            max_items: Maximum context items to return
            include_summary: Include historical summary
        
        Returns:
            Context dictionary with items and optional summary
        """
        items = self.contexts.get(agent_id, [])
        
        # Apply priority-based selection if needed
        if max_items and len(items) > max_items:
            items = self._select_by_priority(agent_id, max_items)
        
        result = {
            "agent_id": agent_id,
            "items": items,
            "item_count": len(items)
        }
        
        if include_summary and agent_id in self.summaries:
            result["summary"] = self.summaries[agent_id]
        
        return result
    
    def get_context_window(
        self,
        agent_id: str,
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent context window.
        
        Args:
            agent_id: Agent identifier
            window_size: Number of recent items
        
        Returns:
            List of recent context items
        """
        items = self.contexts.get(agent_id, [])
        return items[-window_size:] if items else []
    
    def clear_context(self, agent_id: str, keep_summary: bool = True):
        """
        Clear agent context.
        
        Args:
            agent_id: Agent identifier
            keep_summary: Keep historical summary
        """
        self.contexts[agent_id] = []
        self.priorities[agent_id] = {}
        
        if not keep_summary:
            self.summaries.pop(agent_id, None)
        
        logger.info(f"Cleared context for agent '{agent_id}'")
    
    def _should_summarize(self, agent_id: str) -> bool:
        """Check if context should be summarized."""
        items = self.contexts.get(agent_id, [])
        
        # Estimate token count (rough estimate: 1 token ≈ 4 chars)
        total_chars = sum(len(item["content"]) for item in items)
        estimated_tokens = total_chars // 4
        
        threshold = self.max_context_tokens * self.summarization_threshold
        return estimated_tokens > threshold
    
    def _summarize_context(self, agent_id: str):
        """
        Summarize and compress agent context.
        
        Note: In production, this would use the LLM to generate summaries.
        For now, we keep recent items and create a simple summary.
        """
        items = self.contexts.get(agent_id, [])
        
        if len(items) <= 5:
            return
        
        # Keep recent items (top priority)
        recent_items = items[-5:]
        old_items = items[:-5]
        
        # Create simple summary of old items
        summary_parts = []
        for item in old_items:
            summary_parts.append(f"{item['type']}: {item['content'][:100]}...")
        
        self.summaries[agent_id] = "\n".join(summary_parts)
        self.contexts[agent_id] = recent_items
        
        logger.info(f"Summarized context for agent '{agent_id}' ({len(old_items)} items compressed)")
    
    def _select_by_priority(
        self,
        agent_id: str,
        max_items: int
    ) -> List[Dict[str, Any]]:
        """Select highest priority context items."""
        items = self.contexts.get(agent_id, [])
        
        # Sort by priority (descending)
        sorted_items = sorted(
            enumerate(items),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        # Take top N items
        selected = sorted_items[:max_items]
        
        # Sort back by chronological order
        selected.sort(key=lambda x: x[0])
        
        return [item for _, item in selected]
    
    def estimate_token_count(self, agent_id: str) -> int:
        """
        Estimate token count for agent context.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Estimated token count
        """
        items = self.contexts.get(agent_id, [])
        total_chars = sum(len(item["content"]) for item in items)
        
        # Rough estimate: 1 token ≈ 4 characters
        return total_chars // 4
    
    def get_context_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get context statistics for agent."""
        items = self.contexts.get(agent_id, [])
        priorities = self.priorities.get(agent_id, {})
        
        type_counts = {}
        for item in items:
            type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
        
        return {
            "agent_id": agent_id,
            "total_items": len(items),
            "estimated_tokens": self.estimate_token_count(agent_id),
            "max_tokens": self.max_context_tokens,
            "usage_percent": (self.estimate_token_count(agent_id) / self.max_context_tokens) * 100,
            "has_summary": agent_id in self.summaries,
            "type_distribution": type_counts,
            "avg_priority": sum(item["priority"] for item in items) / len(items) if items else 0
        }
    
    def compress_context(self, agent_id: str, target_size: Optional[int] = None):
        """
        Aggressively compress context to target size.
        
        Args:
            agent_id: Agent identifier
            target_size: Target number of items (default: half current size)
        """
        if not self.enable_compression:
            logger.warning("Context compression is disabled")
            return
        
        items = self.contexts.get(agent_id, [])
        
        if not items:
            return
        
        target = target_size or len(items) // 2
        
        if len(items) <= target:
            return
        
        # Select highest priority items
        selected = self._select_by_priority(agent_id, target)
        
        # Create summary of removed items
        removed = len(items) - len(selected)
        summary = f"Context compressed: {removed} items summarized"
        
        self.contexts[agent_id] = selected
        self.summaries[agent_id] = summary
        
        logger.info(f"Compressed context for agent '{agent_id}' from {len(items)} to {len(selected)} items")
    
    def merge_contexts(
        self,
        source_agent_id: str,
        target_agent_id: str,
        priority_adjustment: float = 0.5
    ):
        """
        Merge context from one agent to another.
        
        Args:
            source_agent_id: Source agent
            target_agent_id: Target agent
            priority_adjustment: Multiply source priorities by this factor
        """
        source_items = self.contexts.get(source_agent_id, [])
        
        for item in source_items:
            adjusted_priority = item["priority"] * priority_adjustment
            self.add_to_context(
                target_agent_id,
                item["content"],
                item["type"],
                adjusted_priority,
                item.get("metadata")
            )
        
        logger.info(f"Merged {len(source_items)} items from '{source_agent_id}' to '{target_agent_id}'")
    
    def get_all_agent_contexts(self) -> List[str]:
        """Get list of all agents with contexts."""
        return list(self.contexts.keys())
    
    def reset(self):
        """Reset all contexts."""
        self.contexts.clear()
        self.summaries.clear()
        self.priorities.clear()
        logger.warning("All contexts reset")
