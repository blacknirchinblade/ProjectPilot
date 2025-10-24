import streamlit as st
from src.agents.interactive.conversation_manager import ConversationManager
from src.ui.project import Project # Make sure Project is imported
from typing import List

def render_home_page():
    """Render the home page with overview."""
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 👋 Welcome to ProjectPilot!")
        st.markdown("""
        **ProjectPilot** is your unified AI/ML project generation copilot. Effortlessly create production-ready AI/ML projects with:
        
        ### ✨ Features:
        - 🧠 Multi-agent architecture for smart project design
        - 💾 SQLAlchemy model & database generation
        - 🔄 Automated migrations and backend setup
        - 🧪 Test and documentation generation
        - 🎨 Beautiful, ready-to-use Streamlit dashboards
        - 📦 One-click project download & management
        - 📚 Project history and resume
        """)
        
        if st.button("🚀 Start Generating Code", type="primary", use_container_width=True):
            st.session_state.page = "✨ Generate Code"
            st.rerun()
    
    with col2:
        st.markdown("### 🎯 Available Agents")
        agents = [
            {"name": "Coding Agent", "icon": "💻", "status": "Ready"},
            {"name": "Database Agent", "icon": "💾", "status": "Ready"},
            {"name": "Testing Agent", "icon": "🧪", "status": "Ready"},
            {"name": "Documentation Agent", "icon": "📝", "status": "Ready"},
            {"name": "Planning Agent", "icon": "📋", "status": "Ready"},
            {"name": "Review Agent", "icon": "🔍", "status": "Ready"},
            {"name": "Refactoring Agent", "icon": "🔄", "status": "Ready"},
        ]
        
        for agent in agents:
            st.markdown(f"""
            <div class="agent-card">
                {agent['icon']} <strong>{agent['name']}</strong><br>
                <small>Status: {agent['status']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent projects
    st.markdown("---")
    st.markdown("## 📚 Recent Projects")
    
    projects: List[Project] = st.session_state.project_manager.get_recent_projects(5)
    
    if 'project_chats' not in st.session_state:
        st.session_state.project_chats = {}
    if 'active_chat_project' not in st.session_state:
        st.session_state.active_chat_project = None

    if projects:
        cols = st.columns(min(len(projects), 3))
        for idx, project in enumerate(projects):
            with cols[idx % 3]:
                
                # --- THIS IS THE FIX ---
                # Changed project['key'] to project.key
                st.markdown(f"**{project.name}**\n{project.description}")
                
                if st.button(f"💬 Project Chat", key=f"chatbtn_{project.id}"):
                    st.session_state.active_chat_project = project.id
                
                # Show chat if this project is active
                if st.session_state.active_chat_project == project.id:
                    st.markdown("---")
                    st.markdown(f"### 💬 Chat about: {project.name}")
                    
                    # Load or create conversation manager for this project
                    if project.id not in st.session_state.project_chats:
                        st.session_state.project_chats[project.id] = ConversationManager()
                        # Start conversation with project context
                        st.session_state.project_chats[project.id].start_conversation(
                            initial_query=f"Let's discuss the project: {project.name}. {project.description}",
                            context=project.to_dict() # Convert dataclass to dict for context
                        )
                    
                    conv_mgr = st.session_state.project_chats[project.id]
                    conversation = conv_mgr.active_conversation
                    
                    # Display chat history
                    for msg in conversation.messages:
                        if msg.role == 'user':
                            st.markdown(f"**You:** {msg.content}")
                        else:
                            st.markdown(f"**AI:** {msg.content}")
                    
                    # Input for new message
                    user_input = st.text_input("Type your question or message...", key=f"chat_input_{project.id}")
                    if st.button("Send", key=f"sendbtn_{project.id}") and user_input:
                        conv_mgr.add_message(conversation, role="user", content=user_input)
                        
                        # Get AI response (simulate for now)
                        ai_response = f"[AI response to: {user_input}]"  # Replace with real LLM call
                        
                        conv_mgr.add_message(conversation, role="assistant", content=ai_response)
                        st.rerun() # Use st.rerun() which is safer
                # --- END FIX ---
                
    else:
        st.info("No projects yet. Start by generating your first project!")