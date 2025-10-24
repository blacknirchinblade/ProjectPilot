"""
Shared Memory System - Advanced Multi-Agent Memory Management

This system provides comprehensive memory management for agent coordination:
- Short-term working memory (ephemeral)
- Long-term persistent memory (ChromaDB)
- Semantic search and retrieval
- Context windowing and summarization
- Multi-agent state synchronization

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from loguru import logger
import chromadb
from chromadb.config import Settings


class SharedMemory:
    """
    Advanced shared memory system for multi-agent coordination.
    
    Features:
    - Working memory (RAM-based, fast access)
    - Persistent memory (ChromaDB-based, searchable)
    - Semantic search and retrieval
    - Memory prioritization and cleanup
    - Cross-agent state synchronization
    - Context management and windowing
    """
    
    def __init__(
        self,
        persist_dir: str = "./data/memory",
        max_working_memory: int = 1000,
        enable_persistence: bool = True
    ):
        """
        Initialize shared memory system.
        
        Args:
            persist_dir: Directory for persistent storage
            max_working_memory: Max items in working memory
            enable_persistence: Enable ChromaDB persistence
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_working_memory = max_working_memory
        self.enable_persistence = enable_persistence
        
        # Working Memory (Fast, Ephemeral)
        self.working_memory: Dict[str, Any] = {}
        self.memory_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, float] = {}
        
        # Agent States
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.agent_contexts: Dict[str, List[str]] = {}
        
        # Shared Resources
        self.shared_data: Dict[str, Any] = {}
        self.locks: Set[str] = set()
        
        # Initialize ChromaDB for persistent memory
        if self.enable_persistence:
            self._init_chromadb()
        
        logger.info(f"SharedMemory initialized (persist={enable_persistence}, max_working={max_working_memory})")
    
    def _init_chromadb(self):
        """Initialize ChromaDB for persistent memory."""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma"),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create collections
            self.memory_collection = self.chroma_client.get_or_create_collection(
                name="agent_memory",
                metadata={"description": "Agent working memory"}
            )
            
            self.context_collection = self.chroma_client.get_or_create_collection(
                name="agent_context",
                metadata={"description": "Agent execution context"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.enable_persistence = False
    
    # ==================== Working Memory Operations ====================
    
    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        persistent: bool = False,  # Changed from 'persist' to 'persistent'
        persist: bool = False  # Keep for backward compatibility
    ) -> bool:
        """
        Store item in working memory.
        
        Args:
            key: Unique identifier
            value: Data to store
            metadata: Optional metadata (agent_id, type, tags, etc.)
            persistent: Also store in persistent memory
            persist: Alias for persistent (backward compatibility)
        
        Returns:
            Success status
        """
        try:
            # Store in working memory
            self.working_memory[key] = value
            self.memory_metadata[key] = metadata or {}
            self.access_count[key] = 0
            self.last_access[key] = time.time()
            
            # Add timestamp
            self.memory_metadata[key]["stored_at"] = datetime.now().isoformat()
            
            # Check memory limits
            self._cleanup_if_needed()
            
            # Persist if requested (support both parameter names)
            should_persist = persistent or persist
            if should_persist and self.enable_persistence:
                self._persist_to_chromadb(key, value, metadata)
            
            logger.debug(f"Stored '{key}' in memory (persistent={should_persist})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store '{key}': {e}")
            return False
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve item from working memory.
        
        Args:
            key: Item identifier
            default: Default value if not found
        
        Returns:
            Stored value or default
        """
        if key in self.working_memory:
            # Update access metrics
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access[key] = time.time()
            
            return self.working_memory[key]
        
        return default
    
    def delete(self, key: str) -> bool:
        """
        Delete item from memory.
        
        Args:
            key: Item identifier
        
        Returns:
            Success status
        """
        try:
            if key in self.working_memory:
                del self.working_memory[key]
                self.memory_metadata.pop(key, None)
                self.access_count.pop(key, None)
                self.last_access.pop(key, None)
                
                logger.debug(f"Deleted '{key}' from memory")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        return key in self.working_memory
    
    def clear_working_memory(self):
        """Clear all working memory."""
        self.working_memory.clear()
        self.memory_metadata.clear()
        self.access_count.clear()
        self.last_access.clear()
        logger.info("Working memory cleared")
    
    def _cleanup_if_needed(self):
        """Remove least recently used items if memory is full."""
        if len(self.working_memory) <= self.max_working_memory:
            return
        
        # Sort by last access time (oldest first)
        items_by_access = sorted(
            self.last_access.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest 20%
        remove_count = len(self.working_memory) - self.max_working_memory
        for key, _ in items_by_access[:remove_count]:
            self.delete(key)
        
        logger.info(f"Cleaned up {remove_count} items from working memory")
    
    # ==================== Persistent Memory Operations ====================
    
    def _persist_to_chromadb(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]]
    ):
        """Store item in ChromaDB for persistent memory."""
        try:
            # Convert value to text for embedding
            if isinstance(value, dict):
                text = json.dumps(value, indent=2)
            elif isinstance(value, (list, tuple)):
                text = json.dumps(list(value), indent=2)
            else:
                text = str(value)
            
            # Prepare metadata
            meta = metadata.copy() if metadata else {}
            meta["key"] = key
            meta["stored_at"] = datetime.now().isoformat()
            
            # Store in ChromaDB
            self.memory_collection.add(
                ids=[key],
                documents=[text],
                metadatas=[meta]
            )
            
            logger.debug(f"Persisted '{key}' to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to persist '{key}' to ChromaDB: {e}")
    
    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across persistent memory.
        
        Args:
            query: Search query
            top_k: Number of results (preferred parameter)
            n_results: Number of results (backward compatibility)
            filter_metadata: Metadata filters
        
        Returns:
            List of matching memory items
        """
        if not self.enable_persistence:
            logger.warning("Persistence disabled, cannot search memory")
            return []
        
        # Use top_k if provided, otherwise n_results
        num_results = top_k if n_results is None else n_results
        
        try:
            results = self.memory_collection.query(
                query_texts=[query],
                n_results=num_results,
                where=filter_metadata
            )
            
            # Format results
            memories = []
            if results["ids"] and results["ids"][0]:
                for i, id in enumerate(results["ids"][0]):
                    memories.append({
                        "id": id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            logger.debug(f"Found {len(memories)} memories for query: {query}")
            return memories
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def get_persistent(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve item from persistent memory."""
        if not self.enable_persistence:
            return None
        
        try:
            results = self.memory_collection.get(ids=[key])
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get persistent memory '{key}': {e}")
            return None
    
    # ==================== Agent State Management ====================
    
    def update_agent_state(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ):
        """
        Update agent state.
        
        Args:
            agent_id: Agent identifier
            state: State dictionary
        """
        # Replace state completely (don't add last_updated automatically)
        self.agent_states[agent_id] = state.copy()
        
        logger.debug(f"Updated state for agent '{agent_id}'")
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent state."""
        return self.agent_states.get(agent_id, {})
    
    def get_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent states."""
        return self.agent_states.copy()
    
    def clear_agent_state(self, agent_id: str):
        """Clear specific agent state."""
        self.agent_states.pop(agent_id, None)
        logger.info(f"Cleared state for agent '{agent_id}'")
    
    # ==================== Context Management ====================
    
    def add_context(
        self,
        agent_id: str,
        key: str,
        value: Any,
        persist: bool = True
    ):
        """
        Add context for an agent.
        
        Args:
            agent_id: Agent identifier
            key: Context key
            value: Context value
            persist: Store in persistent memory
        """
        if agent_id not in self.agent_contexts:
            self.agent_contexts[agent_id] = {}
        
        self.agent_contexts[agent_id][key] = value
        
        # Persist context
        if persist and self.enable_persistence:
            try:
                context_id = f"{agent_id}_context_{key}"
                context_text = json.dumps({key: value}) if not isinstance(value, str) else value
                self.context_collection.add(
                    ids=[context_id],
                    documents=[context_text],
                    metadatas=[{
                        "agent_id": agent_id,
                        "key": key,
                        "timestamp": datetime.now().isoformat()
                    }]
                )
            except Exception as e:
                logger.error(f"Failed to persist context: {e}")
        
        logger.debug(f"Added context '{key}' for agent '{agent_id}'")
    
    def get_context(
        self,
        agent_id: str,
        key: str
    ) -> Any:
        """
        Get specific agent context by key.
        
        Args:
            agent_id: Agent identifier
            key: Context key
        
        Returns:
            Context value
        """
        contexts = self.agent_contexts.get(agent_id, {})
        return contexts.get(key)
    
    def get_all_context(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get all context for an agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Dictionary of all context key-value pairs
        """
        return self.agent_contexts.get(agent_id, {}).copy()
    
    def search_context(
        self,
        agent_id: str,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search agent contexts semantically.
        
        Args:
            agent_id: Agent identifier
            query: Search query
            n_results: Number of results
        
        Returns:
            List of matching contexts
        """
        if not self.enable_persistence:
            return []
        
        try:
            where_filter = {"agent_id": agent_id} if agent_id else None
            
            results = self.context_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            contexts = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    contexts.append({
                        "id": results["ids"][0][i],
                        "context": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i]
                    })
            
            return contexts
            
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return []
    
    def clear_context(self, agent_id: str):
        """Clear agent context (alias for clear_agent_context)."""
        self.agent_contexts.pop(agent_id, None)
        logger.info(f"Cleared context for agent '{agent_id}'")
    
    def clear_agent_context(self, agent_id: str):
        """Clear specific agent context."""
        self.agent_contexts.pop(agent_id, None)
        logger.info(f"Cleared context for agent '{agent_id}'")
    
    # ==================== Shared Resources ====================
    
    def set_shared(self, key: str, value: Any):
        """Set shared resource accessible by all agents."""
        self.shared_data[key] = value
        logger.debug(f"Set shared resource '{key}'")
    
    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get shared resource."""
        return self.shared_data.get(key, default)
    
    def delete_shared(self, key: str):
        """Delete shared resource."""
        self.shared_data.pop(key, None)
    
    def acquire_lock(self, resource: str, agent_id: str) -> bool:
        """
        Acquire lock on a resource.
        
        Args:
            resource: Resource identifier
            agent_id: Agent acquiring the lock
        
        Returns:
            True if lock acquired, False if already locked
        """
        lock_key = f"{resource}"
        
        # Check if already locked
        if lock_key in self.locks:
            return False
        
        self.locks.add(lock_key)
        # Store which agent has the lock
        if not hasattr(self, 'lock_owners'):
            self.lock_owners = {}
        self.lock_owners[lock_key] = agent_id
        
        logger.debug(f"Lock acquired on '{resource}' by agent '{agent_id}'")
        return True
    
    def release_lock(self, resource: str, agent_id: str) -> bool:
        """
        Release lock on a resource.
        
        Args:
            resource: Resource identifier
            agent_id: Agent releasing the lock
        
        Returns:
            True if released, False if agent doesn't own the lock
        """
        lock_key = f"{resource}"
        
        # Check if lock exists and agent owns it
        if not hasattr(self, 'lock_owners'):
            self.lock_owners = {}
        
        if lock_key not in self.locks:
            return False
        
        # Verify the agent owns this lock
        if self.lock_owners.get(lock_key) != agent_id:
            logger.warning(f"Agent '{agent_id}' tried to release lock owned by '{self.lock_owners.get(lock_key)}'")
            return False
        
        self.locks.discard(lock_key)
        self.lock_owners.pop(lock_key, None)
        logger.debug(f"Lock released on '{resource}' by agent '{agent_id}'")
        return True
    
    def is_locked(self, resource: str) -> bool:
        """Check if resource is locked."""
        return resource in self.locks
    
    # ==================== Statistics & Monitoring ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "working_memory_count": len(self.working_memory),  # Changed from working_memory_size
            "agent_count": len(self.agent_states),  # Changed from total_agents
            "context_count": sum(len(ctx) for ctx in self.agent_contexts.values()),  # Added context_count
            "shared_resources_count": len(self.shared_data),  # Changed from shared_resources
            "locks_count": len(self.locks),  # Changed from active_locks
            "max_working_memory": self.max_working_memory,
            "memory_usage_percent": (len(self.working_memory) / self.max_working_memory) * 100 if self.max_working_memory > 0 else 0,
            "persistence_enabled": self.enable_persistence,
            "most_accessed": self._get_most_accessed(5)
        }
    
    def _get_most_accessed(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get most accessed memory items."""
        sorted_items = sorted(
            self.access_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [
            {
                "key": key,
                "access_count": count,
                "last_access": self.last_access.get(key, 0)
            }
            for key, count in sorted_items
        ]
    
    def get_memory_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all memory keys, optionally filtered by pattern.
        
        Args:
            pattern: Optional substring to filter keys
        
        Returns:
            List of matching keys
        """
        keys = list(self.working_memory.keys())
        
        if pattern:
            keys = [k for k in keys if pattern in k]
        
        return keys
    
    # ==================== Persistence Management ====================
    
    def save_snapshot(self, filepath: Optional[str] = None) -> bool:
        """
        Save memory snapshot to JSON file.
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(self.persist_dir / f"snapshot_{timestamp}.json")
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "working_memory": self.working_memory,
            "memory_metadata": self.memory_metadata,
            "agent_states": self.agent_states,
            "agent_contexts": self.agent_contexts,
            "shared_data": self.shared_data
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logger.info(f"Memory snapshot saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False
    
    def load_snapshot(self, filepath: str) -> bool:
        """
        Load memory snapshot from JSON file.
        
        Args:
            filepath: Path to snapshot file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                snapshot = json.load(f)
            
            self.working_memory = snapshot.get("working_memory", {})
            self.memory_metadata = snapshot.get("memory_metadata", {})
            self.agent_states = snapshot.get("agent_states", {})
            self.agent_contexts = snapshot.get("agent_contexts", {})
            self.shared_data = snapshot.get("shared_data", {})
            
            logger.info(f"Memory snapshot loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Snapshot file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return False
    
    def reset(self):
        """Reset entire memory system."""
        self.clear_working_memory()
        self.agent_states.clear()
        self.agent_contexts.clear()
        self.shared_data.clear()
        self.locks.clear()
        
        if self.enable_persistence:
            try:
                self.chroma_client.reset()
                self._init_chromadb()
            except Exception as e:
                logger.error(f"Failed to reset persistent memory: {e}")
        
        logger.warning("Memory system reset")
