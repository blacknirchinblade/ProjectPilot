"""
UI Module for AutoCoder Streamlit Application

Contains components and utilities for the web interface.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from .components import (
    render_header,
    render_sidebar,
    render_agent_animation,
    render_progress_tracker,
    render_code_preview,
    render_project_card,
    render_metrics_dashboard,
    render_file_tree,
    render_loading_animation,
    render_success_message,
    render_error_message,
    render_agent_selector
)
from .project_manager import ProjectManager

__all__ = [
    "render_header",
    "render_sidebar",
    "render_agent_animation",
    "render_progress_tracker",
    "render_code_preview",
    "render_project_card",
    "render_metrics_dashboard",
    "render_file_tree",
    "render_loading_animation",
    "render_success_message",
    "render_error_message",
    "render_agent_selector",
    "ProjectManager"
]
