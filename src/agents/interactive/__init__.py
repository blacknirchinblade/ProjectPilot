"""
Interactive Agents Module

This module contains agents that interact with users to gather information,
manage conversations, and process feedback.
Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from .clarification_agent import ClarificationAgent
from .data_structures import (
    Question,
    QuestionCategory,
    Priority,
    Answer,
    ProjectContext
)
from .template_loader import QuestionTemplateLoader, get_template_loader

# New interactive agents
from .user_interaction_agent import (
    UserInteractionAgent,
    ChangeRequest,
    UserFeedback,
    ApprovalRequest,
    ChangeType,
    ApprovalResult
)
from .conversation_manager import (
    ConversationManager,
    Conversation,
    Message
)
from .feedback_handler import (
    FeedbackHandler,
    FeedbackData,
    UserPreference
)

__all__ = [
    'ClarificationAgent',
    'Question',
    'QuestionCategory',
    'Priority',
    'Answer',
    'ProjectContext',
    'QuestionTemplateLoader',
    'get_template_loader',
    # New agents
    "UserInteractionAgent",
    "ConversationManager",
    "FeedbackHandler",
    # Data structures
    "ChangeRequest",
    "UserFeedback",
    "ApprovalRequest",
    "Conversation",
    "Message",
    "FeedbackData",
    "UserPreference",
    # Enums
    "ChangeType",
    "ApprovalResult"
]
