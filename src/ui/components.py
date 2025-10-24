"""
UI Components for Streamlit Application

Reusable components for the AutoCoder Streamlit interface.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import streamlit as st
from typing import Dict, List, Any
from datetime import datetime


def render_header():
    """Render the main header with animations."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AutoCoder</h1>
        <p>AI-Powered Code Generation System</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=AutoCoder", use_column_width=True)
        st.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Home": "home",
            "✨ Generate": "generate",
            "📚 History": "history",
            "📤 Upload": "upload",
            "⚙️ Settings": "settings"
        }
        
        for label, page_id in pages.items():
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id


def render_agent_animation(agent_name: str, status: str, progress: int = 0):
    """
    Render an animated agent card.
    
    Args:
        agent_name: Name of the agent
        status: Current status message
        progress: Progress percentage (0-100)
    """
    icon_map = {
        "Coding Agent": "💻",
        "Database Agent": "💾",
        "Testing Agent": "🧪",
        "Documentation Agent": "📝",
        "Planning Agent": "📋",
        "Review Agent": "🔍",
        "Refactoring Agent": "🔄"
    }
    
    icon = icon_map.get(agent_name, "🤖")
    
    st.markdown(f"""
    <div class="agent-card active">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 2rem;">{icon}</span>
                <strong style="margin-left: 0.5rem;">{agent_name}</strong>
            </div>
            <div>
                <span class="status-badge status-success">{progress}%</span>
            </div>
        </div>
        <div style="margin-top: 0.5rem;">
            <small>{status}</small>
        </div>
        <div style="margin-top: 0.5rem; background: rgba(255,255,255,0.2); border-radius: 4px; height: 4px;">
            <div style="background: white; height: 100%; width: {progress}%; border-radius: 4px; transition: width 0.3s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_progress_tracker(steps: List[Dict[str, Any]], current_step: int = 0):
    """
    Render a progress tracker for multi-step processes.
    
    Args:
        steps: List of step dictionaries with 'name' and 'status'
        current_step: Index of current step
    """
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    
    cols = st.columns(len(steps))
    for idx, (col, step) in enumerate(zip(cols, steps)):
        with col:
            status_icon = "✅" if idx < current_step else "⏳" if idx == current_step else "⭕"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 2rem;">{status_icon}</div>
                <div style="font-size: 0.8rem; margin-top: 0.25rem;">{step['name']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Progress bar
    progress = (current_step / len(steps)) * 100
    st.progress(int(progress))
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_code_preview(code: str, language: str = "python", max_lines: int = 50):
    """
    Render a code preview with syntax highlighting.
    
    Args:
        code: Code to display
        language: Programming language for syntax highlighting
        max_lines: Maximum number of lines to display initially
    """
    lines = code.split('\n')
    
    if len(lines) > max_lines:
        preview_code = '\n'.join(lines[:max_lines])
        remaining = len(lines) - max_lines
        
        with st.expander(f"📝 Code Preview (showing {max_lines} of {len(lines)} lines)"):
            st.code(preview_code, language=language, line_numbers=True)
            
            if st.button(f"Show all {remaining} remaining lines"):
                st.code(code, language=language, line_numbers=True)
    else:
        st.code(code, language=language, line_numbers=True)


def render_project_card(project: Dict[str, Any]):
    """
    Render a project card with actions.
    
    Args:
        project: Project dictionary with metadata
    """
    st.markdown(f"""
    <div class="project-card">
        <h3>📁 {project.get('name', 'Untitled Project')}</h3>
        <p style="color: #666; font-size: 0.9rem;">
            {project.get('description', 'No description')}
        </p>
        <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
            <span class="status-badge status-success">
                {project.get('file_count', 0)} files
            </span>
            <span class="status-badge status-warning">
                {project.get('date', 'Unknown date')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Resume", key=f"resume_{project.get('id')}", use_container_width=True):
            st.session_state.current_project = project
            st.success("Project loaded!")
    
    with col2:
        if st.button("📥 Download", key=f"download_{project.get('id')}", use_container_width=True):
            st.info("Preparing download...")
    
    with col3:
        if st.button("🗑️ Delete", key=f"delete_{project.get('id')}", use_container_width=True):
            st.warning("Delete confirmation needed")


def render_metrics_dashboard(metrics: Dict[str, Any]):
    """
    Render a metrics dashboard.
    
    Args:
        metrics: Dictionary of metric name -> value pairs
    """
    cols = st.columns(len(metrics))
    
    for col, (metric_name, metric_value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metric_value}</div>
                <div class="metric-label">{metric_name}</div>
            </div>
            """, unsafe_allow_html=True)


def render_file_tree(root_path: str, files: List[str]):
    """
    Render a file tree view.
    
    Args:
        root_path: Root directory path
        files: List of file paths relative to root
    """
    st.markdown(f"### 📁 {root_path}")
    
    for file in files:
        indent = file.count('/') * 2
        file_name = file.split('/')[-1]
        icon = "📄" if '.' in file_name else "📁"
        
        st.markdown(f"{' ' * indent}{icon} `{file_name}`")


def render_loading_animation(message: str = "Processing..."):
    """
    Render a loading animation with message.
    
    Args:
        message: Loading message to display
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div style="font-size: 3rem; animation: pulse 1.5s infinite;">⚡</div>
        <div style="margin-top: 1rem; font-size: 1.2rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_success_message(message: str, details: str = ""):
    """
    Render a success message with animation.
    
    Args:
        message: Success message
        details: Additional details
    """
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #10AC84 0%, #0EAD69 100%); 
                color: white; padding: 1.5rem; border-radius: 12px; 
                margin: 1rem 0; animation: slideIn 0.5s ease-out;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">✅</div>
        <div style="font-size: 1.2rem; font-weight: 600;">{message}</div>
        {f'<div style="margin-top: 0.5rem; opacity: 0.9;">{details}</div>' if details else ''}
    </div>
    """, unsafe_allow_html=True)


def render_error_message(message: str, details: str = ""):
    """
    Render an error message.
    
    Args:
        message: Error message
        details: Additional error details
    """
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EE5A6F 0%, #D64754 100%); 
                color: white; padding: 1.5rem; border-radius: 12px; 
                margin: 1rem 0;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">❌</div>
        <div style="font-size: 1.2rem; font-weight: 600;">{message}</div>
        {f'<div style="margin-top: 0.5rem; opacity: 0.9; font-size: 0.9rem;">{details}</div>' if details else ''}
    </div>
    """, unsafe_allow_html=True)


def render_agent_selector(selected_agents: List[str] = None) -> List[str]:
    """
    Render an agent selector with checkboxes.
    
    Args:
        selected_agents: List of pre-selected agent names
    
    Returns:
        List of selected agent names
    """
    if selected_agents is None:
        selected_agents = []
    
    agents = {
        "💻 Coding Agent": "Essential for code generation",
        "💾 Database Agent": "SQLAlchemy models and migrations",
        "🧪 Testing Agent": "Unit and integration tests",
        "📝 Documentation Agent": "Docstrings and README",
        "📋 Planning Agent": "Architecture planning",
        "🔍 Review Agent": "Code review and quality checks",
        "🔄 Refactoring Agent": "Code optimization"
    }
    
    st.markdown("### 🤖 Select Agents")
    
    selected = []
    for agent_name, description in agents.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.checkbox(agent_name, value=agent_name in selected_agents, key=f"agent_{agent_name}"):
                selected.append(agent_name)
        with col2:
            st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
    
    return selected
