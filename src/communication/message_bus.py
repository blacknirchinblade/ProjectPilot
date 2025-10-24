"""
Message Bus System - Event-driven Inter-Agent Communication

This module provides a comprehensive message bus for agent coordination:
- Pub/sub messaging pattern
- Priority-based message queuing
- Topic-based routing
- Message history and replay
- Event logging and monitoring

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from collections import defaultdict, deque
from loguru import logger
import json
from pathlib import Path


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class Message:
    """
    Message object for inter-agent communication.
    
    Attributes:
        id: Unique message identifier
        topic: Message topic/channel
        sender: Agent ID of sender
        recipient: Target agent ID (None for broadcast)
        content: Message content (any serializable data)
        priority: Message priority level
        metadata: Additional metadata
        timestamp: Message creation time
    """
    
    def __init__(
        self,
        topic: str,
        sender: str,
        content: Any,
        recipient: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a message.
        
        Args:
            topic: Message topic/channel
            sender: Agent ID of sender
            content: Message content
            recipient: Target agent ID (None for broadcast)
            priority: Message priority level
            metadata: Additional metadata
        """
        self.id = str(uuid.uuid4())
        self.topic = topic
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.delivered = False
        self.read = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.name,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "delivered": self.delivered,
            "read": self.read
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        msg = cls(
            topic=data["topic"],
            sender=data["sender"],
            content=data["content"],
            recipient=data.get("recipient"),
            priority=MessagePriority[data.get("priority", "NORMAL")],
            metadata=data.get("metadata", {})
        )
        msg.id = data["id"]
        msg.timestamp = data["timestamp"]
        msg.delivered = data.get("delivered", False)
        msg.read = data.get("read", False)
        return msg
    
    def __repr__(self) -> str:
        return f"Message(id={self.id[:8]}, topic={self.topic}, sender={self.sender}, priority={self.priority.name})"


class MessageBus:
    """
    Central message bus for agent communication.
    
    Features:
    - Topic-based pub/sub messaging
    - Priority queues for message ordering
    - Message history and replay
    - Subscriber management
    - Event logging
    - Message filtering and routing
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        enable_persistence: bool = False,
        persist_dir: str = "./data/messages"
    ):
        """
        Initialize message bus.
        
        Args:
            max_history: Maximum messages to keep in history
            enable_persistence: Enable message persistence to disk
            persist_dir: Directory for message persistence
        """
        self.max_history = max_history
        self.enable_persistence = enable_persistence
        self.persist_dir = Path(persist_dir)
        
        if enable_persistence:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Subscribers: topic -> set of (agent_id, callback)
        self.subscribers: Dict[str, Set[tuple]] = defaultdict(set)
        
        # Message queues: agent_id -> priority queue
        self.message_queues: Dict[str, deque] = defaultdict(deque)
        
        # Message history
        self.message_history: deque = deque(maxlen=max_history)
        
        # Topic history
        self.topic_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_read": 0,
            "topics": set(),
            "active_agents": set()
        }
        
        logger.info(f"MessageBus initialized (history={max_history}, persistence={enable_persistence})")
    
    # ==================== Publishing ====================
    
    def publish(
        self,
        topic: str,
        sender: str,
        content: Any,
        recipient: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Publish a message to a topic.
        
        Args:
            topic: Message topic
            sender: Agent ID of sender
            content: Message content
            recipient: Target agent (None for broadcast)
            priority: Message priority
            metadata: Additional metadata
        
        Returns:
            Message ID
        """
        # Create message
        message = Message(
            topic=topic,
            sender=sender,
            content=content,
            recipient=recipient,
            priority=priority,
            metadata=metadata
        )
        
        # Add to history
        self.message_history.append(message)
        self.topic_history[topic].append(message)
        
        # Update stats
        self.stats["messages_sent"] += 1
        self.stats["topics"].add(topic)
        self.stats["active_agents"].add(sender)
        
        # Deliver to subscribers or specific recipient
        if recipient:
            self._deliver_to_agent(recipient, message)
        else:
            self._broadcast_to_subscribers(topic, message)
        
        # Persist if enabled
        if self.enable_persistence:
            self._persist_message(message)
        
        logger.debug(f"Published message {message.id[:8]} to topic '{topic}' from {sender}")
        return message.id
    
    def _broadcast_to_subscribers(self, topic: str, message: Message):
        """Broadcast message to all subscribers of a topic."""
        if topic not in self.subscribers:
            logger.debug(f"No subscribers for topic '{topic}'")
            return
        
        for agent_id, callback in self.subscribers[topic]:
            self._deliver_to_agent(agent_id, message, callback)
    
    def _deliver_to_agent(
        self,
        agent_id: str,
        message: Message,
        callback: Optional[Callable] = None
    ):
        """Deliver message to a specific agent."""
        # Add to agent's queue
        self.message_queues[agent_id].append(message)
        
        # Sort by priority (higher priority first)
        self.message_queues[agent_id] = deque(
            sorted(
                self.message_queues[agent_id],
                key=lambda m: m.priority.value,
                reverse=True
            )
        )
        
        message.delivered = True
        self.stats["messages_delivered"] += 1
        
        # Call callback if provided
        if callback:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in callback for agent {agent_id}: {e}")
        
        logger.debug(f"Delivered message {message.id[:8]} to {agent_id}")
    
    # ==================== Subscribing ====================
    
    def subscribe(
        self,
        topic: str,
        agent_id: str,
        callback: Optional[Callable] = None
    ):
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic to subscribe to
            agent_id: Agent ID
            callback: Optional callback function(message)
        """
        self.subscribers[topic].add((agent_id, callback))
        self.stats["active_agents"].add(agent_id)
        logger.info(f"Agent {agent_id} subscribed to topic '{topic}'")
    
    def unsubscribe(self, topic: str, agent_id: str):
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            agent_id: Agent ID
        """
        # Remove all entries for this agent on this topic
        self.subscribers[topic] = {
            (aid, cb) for aid, cb in self.subscribers[topic]
            if aid != agent_id
        }
        logger.info(f"Agent {agent_id} unsubscribed from topic '{topic}'")
    
    def unsubscribe_all(self, agent_id: str):
        """
        Unsubscribe agent from all topics.
        
        Args:
            agent_id: Agent ID
        """
        for topic in list(self.subscribers.keys()):
            self.unsubscribe(topic, agent_id)
        
        logger.info(f"Agent {agent_id} unsubscribed from all topics")
    
    def get_subscriptions(self, agent_id: str) -> List[str]:
        """
        Get all topics an agent is subscribed to.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            List of topic names
        """
        topics = []
        for topic, subscribers in self.subscribers.items():
            for aid, _ in subscribers:
                if aid == agent_id:
                    topics.append(topic)
                    break
        return topics
    
    # ==================== Receiving ====================
    
    def receive(
        self,
        agent_id: str,
        mark_as_read: bool = True
    ) -> Optional[Message]:
        """
        Receive next message from agent's queue.
        
        Args:
            agent_id: Agent ID
            mark_as_read: Mark message as read
        
        Returns:
            Next message or None
        """
        if not self.message_queues[agent_id]:
            return None
        
        message = self.message_queues[agent_id].popleft()
        
        if mark_as_read:
            message.read = True
            self.stats["messages_read"] += 1
        
        logger.debug(f"Agent {agent_id} received message {message.id[:8]}")
        return message
    
    def receive_all(
        self,
        agent_id: str,
        mark_as_read: bool = True
    ) -> List[Message]:
        """
        Receive all messages from agent's queue.
        
        Args:
            agent_id: Agent ID
            mark_as_read: Mark messages as read
        
        Returns:
            List of messages
        """
        messages = []
        while self.message_queues[agent_id]:
            msg = self.receive(agent_id, mark_as_read)
            if msg:
                messages.append(msg)
        
        return messages
    
    def peek(self, agent_id: str) -> Optional[Message]:
        """
        Peek at next message without removing it.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Next message or None
        """
        if not self.message_queues[agent_id]:
            return None
        
        return self.message_queues[agent_id][0]
    
    def has_messages(self, agent_id: str) -> bool:
        """
        Check if agent has pending messages.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if messages pending
        """
        return len(self.message_queues[agent_id]) > 0
    
    def get_message_count(self, agent_id: str) -> int:
        """
        Get number of pending messages for agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Number of pending messages
        """
        return len(self.message_queues[agent_id])
    
    # ==================== History & Replay ====================
    
    def get_history(
        self,
        topic: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get message history with optional filtering.
        
        Args:
            topic: Filter by topic
            agent_id: Filter by sender or recipient
            limit: Maximum messages to return
        
        Returns:
            List of messages
        """
        if topic:
            messages = list(self.topic_history.get(topic, []))
        else:
            messages = list(self.message_history)
        
        # Filter by agent
        if agent_id:
            messages = [
                m for m in messages
                if m.sender == agent_id or m.recipient == agent_id
            ]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def replay_messages(
        self,
        agent_id: str,
        topic: Optional[str] = None,
        from_timestamp: Optional[str] = None
    ) -> int:
        """
        Replay historical messages to an agent.
        
        Args:
            agent_id: Target agent ID
            topic: Filter by topic
            from_timestamp: Replay from this timestamp
        
        Returns:
            Number of messages replayed
        """
        messages = self.get_history(topic=topic)
        
        # Filter by timestamp
        if from_timestamp:
            messages = [m for m in messages if m.timestamp >= from_timestamp]
        
        # Replay to agent
        count = 0
        for message in messages:
            self._deliver_to_agent(agent_id, message)
            count += 1
        
        logger.info(f"Replayed {count} messages to {agent_id}")
        return count
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get message by ID from history.
        
        Args:
            message_id: Message ID
        
        Returns:
            Message or None
        """
        for message in self.message_history:
            if message.id == message_id:
                return message
        return None
    
    # ==================== Statistics & Monitoring ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "messages_sent": self.stats["messages_sent"],
            "messages_delivered": self.stats["messages_delivered"],
            "messages_read": self.stats["messages_read"],
            "total_topics": len(self.stats["topics"]),
            "active_agents": len(self.stats["active_agents"]),
            "history_size": len(self.message_history),
            "pending_messages": sum(len(q) for q in self.message_queues.values()),
            "topics": list(self.stats["topics"]),
            "agents": list(self.stats["active_agents"])
        }
    
    def get_topic_statistics(self, topic: str) -> Dict[str, Any]:
        """
        Get statistics for a specific topic.
        
        Args:
            topic: Topic name
        
        Returns:
            Topic statistics
        """
        messages = self.topic_history.get(topic, [])
        
        return {
            "topic": topic,
            "message_count": len(messages),
            "subscriber_count": len(self.subscribers.get(topic, set())),
            "subscribers": [aid for aid, _ in self.subscribers.get(topic, set())],
            "recent_messages": len([m for m in messages if not m.read])
        }
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent statistics
        """
        sent = len([m for m in self.message_history if m.sender == agent_id])
        received = len([m for m in self.message_history if m.recipient == agent_id])
        
        return {
            "agent_id": agent_id,
            "messages_sent": sent,
            "messages_received": received,
            "pending_messages": len(self.message_queues[agent_id]),
            "subscriptions": self.get_subscriptions(agent_id)
        }
    
    # ==================== Persistence ====================
    
    def _persist_message(self, message: Message):
        """Persist message to disk."""
        try:
            date = datetime.now().strftime("%Y%m%d")
            filepath = self.persist_dir / f"messages_{date}.jsonl"
            
            with open(filepath, 'a') as f:
                f.write(json.dumps(message.to_dict()) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to persist message: {e}")
    
    def load_history(self, filepath: str) -> int:
        """
        Load message history from file.
        
        Args:
            filepath: Path to history file
        
        Returns:
            Number of messages loaded
        """
        try:
            count = 0
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    message = Message.from_dict(data)
                    self.message_history.append(message)
                    self.topic_history[message.topic].append(message)
                    count += 1
            
            logger.info(f"Loaded {count} messages from {filepath}")
            return count
        
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return 0
    
    # ==================== Cleanup ====================
    
    def clear_queue(self, agent_id: str):
        """
        Clear all messages for an agent.
        
        Args:
            agent_id: Agent ID
        """
        self.message_queues[agent_id].clear()
        logger.info(f"Cleared message queue for {agent_id}")
    
    def clear_topic_history(self, topic: str):
        """
        Clear history for a topic.
        
        Args:
            topic: Topic name
        """
        self.topic_history[topic].clear()
        logger.info(f"Cleared history for topic '{topic}'")
    
    def clear_all(self):
        """Clear all messages and history."""
        self.message_queues.clear()
        self.message_history.clear()
        self.topic_history.clear()
        self.subscribers.clear()
        
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_read": 0,
            "topics": set(),
            "active_agents": set()
        }
        
        logger.warning("Cleared all message bus data")
    
    def reset(self):
        """Reset message bus to initial state."""
        self.clear_all()
        logger.warning("Message bus reset")
