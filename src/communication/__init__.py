"""
Communication module for inter-agent messaging.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from .message_bus import MessageBus, Message, MessagePriority

__all__ = ['MessageBus', 'Message', 'MessagePriority']
