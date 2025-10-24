"""
Conversation Manager - Multi-Turn Dialogue System

Manages conversations between user and AutoCoder:
- Maintains conversation history
- Tracks context across turns
- Handles follow-up questions
- Preserves state between interactions

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import json
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


@dataclass
class Message:
    """Represents a single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a conversation session."""
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ConversationManager:
    """
    Manages multi-turn conversations with context preservation.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        max_history: int = 20,
        save_path: Optional[Path] = None
    ):
        """
        Initialize the conversation manager.
        
        Args:
            model_name: LLM model to use
            max_history: Maximum messages to keep in history
            save_path: Path to save conversation history
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5)
        self.max_history = max_history
        self.save_path = save_path or Path("data/conversations")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.active_conversation: Optional[Conversation] = None
        
        logger.info(f"Initialized ConversationManager")
    
    def start_conversation(
        self,
        initial_query: str,
        context: Optional[Dict] = None,
        conversation_id: Optional[str] = None  # Add this parameter
    ) -> Conversation:
        """
        Start a new conversation.
        
        Args:
            initial_query: User's first message
            context: Initial context (project info, etc.)
            conversation_id: Optional custom conversation ID
        
        Returns:
            New Conversation object
        """
        # Use provided conversation_id or generate one
        if conversation_id is None:
            conversation_id = f"temp_{uuid.uuid4()}"
        
        conversation = Conversation(
            conversation_id=conversation_id,
            context=context or {}
        )
        
        # Add initial message
        user_message = Message(
            role="user",
            content=initial_query
        )
        conversation.messages.append(user_message)
        
        self.active_conversation = conversation
        
        logger.info(f"Started conversation: {conversation_id}")
        return conversation
    
    def add_message(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a message to the conversation.
        
        Args:
            conversation: Conversation to add to
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        
        # Trim history if too long
        if len(conversation.messages) > self.max_history:
            # Keep system message + recent messages
            conversation.messages = conversation.messages[-self.max_history:]
        
        logger.debug(f"Added {role} message ({len(content)} chars)")
    
    def get_context(self, conversation: Conversation) -> Dict:
        """
        Get conversation context for LLM.
        
        Args:
            conversation: Conversation to extract context from
        
        Returns:
            Context dictionary
        """
        context = {
            "conversation_id": conversation.conversation_id,
            "message_count": len(conversation.messages),
            "duration": (datetime.now() - conversation.created_at).total_seconds(),
            "last_user_message": None,
            "last_assistant_message": None
        }
        
        # Get last messages
        for msg in reversed(conversation.messages):
            if msg.role == "user" and not context["last_user_message"]:
                context["last_user_message"] = msg.content
            elif msg.role == "assistant" and not context["last_assistant_message"]:
                context["last_assistant_message"] = msg.content
            
            if context["last_user_message"] and context["last_assistant_message"]:
                break
        
        # Merge with conversation context
        context.update(conversation.context)
        
        return context
    
    def continue_conversation(
        self,
        conversation: Conversation,
        user_message: str
    ) -> str:
        """
        Continue the conversation with a new user message.
        
        Args:
            conversation: Active conversation
            user_message: New user message
        
        Returns:
            Assistant's response
        """
        logger.info(f"Continuing conversation: {conversation.conversation_id}")
        
        try:
            # Add user message
            self.add_message(conversation, "user", user_message)
            
            # Build message history for LLM
            messages = self._build_message_history(conversation)
            
            # Get response
            response = self.llm.invoke(messages)
            assistant_message = response.content
            
            # Add assistant message
            self.add_message(conversation, "assistant", assistant_message)
            
            # Save conversation
            self._save_conversation(conversation)
            
            logger.info(f"Generated response ({len(assistant_message)} chars)")
            return assistant_message
            
        except Exception as e:
            logger.error(f"Failed to continue conversation: {e}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            self.add_message(conversation, "assistant", error_msg)
            return error_msg
    
    def summarize_conversation(self, conversation: Conversation) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            conversation: Conversation to summarize
        
        Returns:
            Summary text
        """
        logger.info("Generating conversation summary")
        
        try:
            # Build conversation text
            conv_text = "\n\n".join([
                f"{msg.role.upper()}: {msg.content}"
                for msg in conversation.messages
            ])
            
            prompt = f"""
Summarize this conversation between a user and AutoCoder (an AI code generation system).

CONVERSATION:
{conv_text}

Provide a concise summary covering:
1. What the user wanted
2. Key decisions made
3. Main code generated
4. Current status

Keep it brief (3-5 sentences).
"""
            
            messages = [
                SystemMessage(content="You are a helpful assistant that summarizes conversations."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            summary = response.content
            
            logger.info("Summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Summary not available"
    
    def export_conversation(
        self,
        conversation: Conversation,
        format: str = "json"
    ) -> str:
        """
        Export conversation to various formats.
        
        Args:
            conversation: Conversation to export
            format: Export format (json, markdown, text)
        
        Returns:
            Exported content as string
        """
        if format == "json":
            return self._export_json(conversation)
        elif format == "markdown":
            return self._export_markdown(conversation)
        else:  # text
            return self._export_text(conversation)
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a saved conversation.
        
        Args:
            conversation_id: ID of conversation to load
        
        Returns:
            Conversation object or None
        """
        file_path = self.save_path / f"{conversation_id}.json"
        
        if not file_path.exists():
            logger.warning(f"Conversation not found: {conversation_id}")
            return None
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Reconstruct conversation
            conversation = Conversation(
                conversation_id=data["conversation_id"],
                context=data["context"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"])
            )
            
            # Reconstruct messages
            for msg_data in data["messages"]:
                message = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=msg_data.get("metadata", {})
                )
                conversation.messages.append(message)
            
            logger.info(f"Loaded conversation: {conversation_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return None
    
    def list_conversations(self) -> List[Dict]:
        """
        List all saved conversations.
        
        Returns:
            List of conversation info dictionaries
        """
        conversations = []
        
        for file_path in self.save_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                conversations.append({
                    "id": data["conversation_id"],
                    "created_at": data["created_at"],
                    "message_count": len(data["messages"]),
                    "last_message": data["messages"][-1]["content"][:100] if data["messages"] else ""
                })
            except:
                continue
        
        # Sort by creation time (newest first)
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        
        return conversations
    
    # Helper methods
    
    def _build_message_history(self, conversation: Conversation) -> List:
        """Build message history for LLM."""
        messages = [
            SystemMessage(content=self._get_system_prompt(conversation))
        ]
        
        # Add conversation messages
        for msg in conversation.messages:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        return messages
    
    def _get_system_prompt(self, conversation: Conversation) -> str:
        """Get system prompt with context."""
        prompt = """You are AutoCoder, an AI-powered code generation assistant.

You help users:
- Design and generate code
- Answer questions about their project
- Make changes and improvements
- Provide guidance and explanations

Be helpful, concise, and accurate. Remember the conversation context."""
        
        if conversation.context:
            prompt += f"\n\nCONTEXT:\n"
            for key, value in conversation.context.items():
                prompt += f"- {key}: {value}\n"
        
        return prompt
    
    def _save_conversation(self, conversation: Conversation):
        """Save conversation to disk."""
        try:
            file_path = self.save_path / f"{conversation.conversation_id}.json"
            
            data = {
                "conversation_id": conversation.conversation_id,
                "context": conversation.context,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata
                    }
                    for msg in conversation.messages
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved conversation: {conversation.conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def _export_json(self, conversation: Conversation) -> str:
        """Export as JSON."""
        data = {
            "id": conversation.conversation_id,
            "created": conversation.created_at.isoformat(),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "time": msg.timestamp.isoformat()
                }
                for msg in conversation.messages
            ]
        }
        return json.dumps(data, indent=2)
    
    def _export_markdown(self, conversation: Conversation) -> str:
        """Export as Markdown."""
        lines = [
            f"# Conversation: {conversation.conversation_id}",
            f"**Created**: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ]
        
        for msg in conversation.messages:
            role_label = "👤 User" if msg.role == "user" else "🤖 Assistant"
            lines.append(f"## {role_label}")
            lines.append(f"*{msg.timestamp.strftime('%H:%M:%S')}*")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_text(self, conversation: Conversation) -> str:
        """Export as plain text."""
        lines = [f"Conversation: {conversation.conversation_id}", "=" * 60, ""]
        
        for msg in conversation.messages:
            role = msg.role.upper()
            time = msg.timestamp.strftime('%H:%M:%S')
            lines.append(f"[{time}] {role}:")
            lines.append(msg.content)
            lines.append("")
        
        return "\n".join(lines)


# CODE_GENERATION_COMPLETE
