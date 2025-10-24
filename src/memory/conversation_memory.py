"""
Conversation Memory - Manage Multi-Turn Conversations

Handles conversation history, context windows, and summarization
for natural multi-turn interactions with agents.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class ConversationMemory:
    """
    Manages conversation history and context for agent interactions.
    
    Features:
    - Message history tracking
    - Conversation threading
    - Context window management
    - Automatic summarization
    - Role-based message filtering
    """
    
    def __init__(
        self,
        max_messages: int = 50,
        context_window: int = 10,
        context_window_size: Optional[int] = None,  # Backward compatibility
        enable_summarization: bool = True
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to store
            context_window: Number of recent messages in context
            context_window_size: Alias for context_window (backward compatibility)
            enable_summarization: Enable automatic summarization
        """
        self.max_messages = max_messages
        # Support both parameter names
        self.context_window = context_window_size if context_window_size is not None else context_window
        self.enable_summarization = enable_summarization
        
        # Conversation storage
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.summaries: Dict[str, str] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Default conversation for backward compatibility (single conversation mode)
        self.default_conversation = "default"
        
        # Single conversation mode properties (backward compatibility)
        self.messages: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.created_at = datetime.now().isoformat()
        
        # Expose context_window_size for backward compatibility
        self.context_window_size = self.context_window
        
        logger.info(f"ConversationMemory initialized (max={max_messages}, window={self.context_window})")
    
    # ==================== BACKWARD COMPATIBILITY METHODS ====================
    # Methods for both single and multi-conversation modes
    
    def add_message(
        self,
        role_or_conv_id: str,
        content_or_role: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add message - supports both single and multi-conversation modes.
        
        Single conversation mode (2 args):
            add_message(role, content, metadata=None)
        
        Multi conversation mode (3+ args):
            add_message(conversation_id, role, content, metadata=None)
        """
        if content is None:
            # Single conversation mode: add_message(role, content)
            role = role_or_conv_id
            content = content_or_role
            conversation_id = self.default_conversation
        else:
            # Multi conversation mode: add_message(conversation_id, role, content)
            conversation_id = role_or_conv_id
            role = content_or_role
        
        # Use existing implementation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            self.metadata[conversation_id] = {
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[conversation_id].append(message)
        self.metadata[conversation_id]["message_count"] += 1
        self.metadata[conversation_id]["last_updated"] = datetime.now().isoformat()
        
        # Update single conversation mode properties
        if conversation_id == self.default_conversation:
            self.messages = self.conversations[conversation_id]
        
        # Trim if exceeds max
        if len(self.conversations[conversation_id]) > self.max_messages:
            self._trim_conversation(conversation_id)
        
        logger.debug(f"Added {role} message to conversation '{conversation_id}'")
    
    def get_messages(
        self,
        conversation_id: Optional[str] = None,
        last_n: Optional[int] = None,
        role: Optional[str] = None,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages - supports both single and multi-conversation modes.
        
        Single conversation mode:
            get_messages(last_n=None, role=None)
        
        Multi conversation mode:
            get_messages(conversation_id, last_n=None, role_filter=None)
        """
        # Determine conversation ID
        if conversation_id is None:
            conversation_id = self.default_conversation
        
        messages = self.conversations.get(conversation_id, [])
        
        # Filter by role (support both parameter names)
        role_to_filter = role_filter or role
        if role_to_filter:
            messages = [m for m in messages if m["role"] == role_to_filter]
        
        # Get last N
        if last_n:
            messages = messages[-last_n:]
        
        return messages
    
    def get_context_window(
        self,
        conversation_id: Optional[str] = None,
        window_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages within context window.
        
        Args:
            conversation_id: Conversation identifier (None for default)
            window_size: Override default window size
        
        Returns:
            List of recent messages
        """
        if conversation_id is None:
            conversation_id = self.default_conversation
        
        window = window_size if window_size is not None else self.context_window
        return self.get_messages(conversation_id, last_n=window)
    
    def get_summary(self, conversation_id: Optional[str] = None) -> str:
        """Get conversation summary."""
        if conversation_id is None:
            conversation_id = self.default_conversation
        return self.summaries.get(conversation_id, "")
    
    def set_summary(self, summary: str, conversation_id: Optional[str] = None):
        """Set conversation summary."""
        if conversation_id is None:
            conversation_id = self.default_conversation
        self.summaries[conversation_id] = summary
        if conversation_id == self.default_conversation:
            self.summary = summary
        logger.debug(f"Set summary for conversation '{conversation_id}'")
    
    def clear_conversation(self, reset_summary: bool = False, conversation_id: Optional[str] = None):
        """Clear conversation."""
        if conversation_id is None:
            conversation_id = self.default_conversation
        
        self.conversations[conversation_id] = []
        if conversation_id == self.default_conversation:
            self.messages = []
        
        if reset_summary:
            self.summaries[conversation_id] = ""
            if conversation_id == self.default_conversation:
                self.summary = ""
        
        logger.info(f"Cleared conversation '{conversation_id}'")
    
    def export_conversation(
        self,
        format: str = "text",
        conversation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Export conversation in specified format.
        
        Args:
            format: Export format (text, json, markdown)
            conversation_id: Conversation identifier (None for default)
        
        Returns:
            Formatted conversation string or None if invalid format
        """
        if conversation_id is None:
            conversation_id = self.default_conversation
        
        messages = self.get_messages(conversation_id)
        
        if format == "json":
            import json
            return json.dumps(messages, indent=2)
        
        elif format == "markdown":
            lines = [f"# Conversation: {conversation_id}\n"]
            for msg in messages:
                lines.append(f"**{msg['role']}:** ({msg['timestamp']}):")
                lines.append(f"{msg['content']}\n")
            return "\n".join(lines)
        
        elif format == "text":
            lines = []
            for msg in messages:
                lines.append(f"{msg['role']}: {msg['content']}")
            return "\n".join(lines)
        
        else:
            return None
    
    def _trim_conversation(self, conversation_id: str):
        """Trim conversation to max messages, optionally summarizing."""
        messages = self.conversations[conversation_id]
        
        if len(messages) > self.max_messages:
            # Keep recent messages
            old_messages = messages[:-self.max_messages]
            self.conversations[conversation_id] = messages[-self.max_messages:]
            
            # Update single-conversation mode messages
            if conversation_id == self.default_conversation:
                self.messages = self.conversations[conversation_id]
            
            # Create summary if enabled
            if self.enable_summarization:
                summary_text = f"Previous {len(old_messages)} messages summarized"
                self.set_summary(summary_text, conversation_id)
            
            logger.info(f"Trimmed conversation '{conversation_id}' to {self.max_messages} messages")
    
    def get_all_conversations(self) -> List[str]:
        """Get all conversation IDs."""
        return list(self.conversations.keys())
    
    def get_conversation_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics."""
        if conversation_id is None:
            conversation_id = self.default_conversation
        
        messages = self.conversations.get(conversation_id, [])
        meta = self.metadata.get(conversation_id, {})
        
        role_counts = {}
        for msg in messages:
            role = msg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "role_distribution": role_counts,
            "created_at": meta.get("created_at"),
            "last_updated": meta.get("last_updated"),
            "has_summary": conversation_id in self.summaries
        }
