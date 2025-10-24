from asyncore import loop
import streamlit as st
from src.ui.project_manager import ProjectManager
from src.agents.interactive.conversation_manager import ConversationManager
from src.agents.interview.complex_interview_agent import ComplexInterviewAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.ui.project import Project # Import the dataclass
from pathlib import Path
import asyncio
import traceback
from loguru import logger
from typing import List
from src.agents.interactive.user_interaction_agent import UserInteractionAgent


def _load_project_code(project_path: Path, max_chars: int = 20000) -> str:
    """
    Walks the src/ directory of a project, concatenates all .py files,
    and truncates to a max character limit to use as LLM context.
    """
    src_path = project_path / "src"
    if not src_path.exists():
        return "Source code directory (src/) not found."

    all_code = []
    total_chars = 0

    for file_path in src_path.rglob("*.py"):
        if total_chars >= max_chars:
            all_code.append("\n... [PROJECT CONTEXT TRUNCATED] ...")
            break
        
        try:
            relative_path = file_path.relative_to(project_path)
            content = file_path.read_text(encoding="utf-8")
            
            # Fix: Cannot use backslash in f-string expression.
            # Perform the replacement before creating the f-string.
            path_str = str(relative_path).replace('\\', '/')
            header = f"--- FILE: {path_str} ---"
            file_content = f"{header}\n{content}\n\n"
            
            if total_chars + len(file_content) > max_chars:
                # Add a truncated version if this file is too long
                remaining_chars = max_chars - total_chars
                all_code.append(file_content[:remaining_chars] + "\n... [FILE TRUNCATED] ...\n\n")
                total_chars = max_chars
            else:
                all_code.append(file_content)
                total_chars += len(file_content)
        
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
    
    if not all_code:
        return "No Python files found in src/ directory."

    return "".join(all_code)


def render_history_page():
    """Render project history page."""
    st.markdown("## 📚 Project History")

    projects: List[Project] = st.session_state.project_manager.get_all_projects()
    if not projects:
        st.info("No projects in history. Generate your first project to get started!")
        return

    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search projects:", placeholder="Search by name or description...")
    with col2:
        sort_by = st.selectbox("Sort by:", ["Date (newest)", "Date (oldest)", "Name"])
    with col3:
        filter_type = st.selectbox("Filter:", ["All", "Simple", "Full Project", "Database"])

    # Initialize chat states if they don't exist
    if 'project_chats' not in st.session_state:
        st.session_state.project_chats = {}
    if 'active_chat_project' not in st.session_state:
        st.session_state.active_chat_project = None
    if 'active_interview_project' not in st.session_state:
        st.session_state.active_interview_project = None
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = {}

    # Display projects
    for project in projects:
        with st.expander(f"📁 {project.name} - {project.date}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Type:** {project.project_type}")
                st.markdown(f"**Description:** {project.description}")
                st.markdown(f"**Files:** {project.file_count}")

            with col2:
                # --- CHAT BUTTON ---
                if st.button("💬 Project Chat", key=f"chat_history_{project.id}"):
                    st.session_state.active_chat_project = project.id if st.session_state.active_chat_project != project.id else None
                    st.session_state.active_interview_project = None # Close other accordion
                    st.rerun()

                # --- NEW INTERVIEW PREP BUTTON ---
                if st.button("🎤 Interview Prep", key=f"interview_{project.id}"):
                    st.session_state.active_interview_project = project.id if st.session_state.active_interview_project != project.id else None
                    st.session_state.active_chat_project = None # Close other accordion
                    st.rerun()

                if st.button("📥 Download", key=f"download_{project.id}"):
                    st.info("Download functionality coming soon!")

                if st.button("🗑️ Delete", key=f"delete_{project.id}"):
                    st.session_state.project_manager.delete_project(project.id)
                    st.rerun()

            # --- RENDER CHAT INTERFACE (if active) ---
            if st.session_state.active_chat_project == project.id:
                render_project_chat(project)

            # --- RENDER INTERVIEW INTERFACE (if active) ---
            if st.session_state.active_interview_project == project.id:
                render_interview_prep(project)


def render_project_chat(project: Project):
    """Renders the chat interface for a specific project."""
    st.markdown("---")
    st.markdown(f"### 💬 Chat about: {project.name}")

    # Load or create conversation manager for this project
    if project.id not in st.session_state.project_chats:
        # Each project's chat history is saved in its own directory
        chat_save_path = Path(f"data/projects/{project.id}/chat_history")
        st.session_state.project_chats[project.id] = ConversationManager(save_path=chat_save_path)

        # Try to load existing conversation or start a new one
        loaded_conv = st.session_state.project_chats[project.id].load_conversation(project.id)
        if not loaded_conv:
            st.session_state.project_chats[project.id].start_conversation(
                initial_query=f"Let's discuss the project: {project.name}. {project.description}",
                context=project.to_dict()
            )

    conv_mgr = st.session_state.project_chats[project.id]
    conversation = conv_mgr.active_conversation

    # Display chat history
    with st.container(height=300, border=True):
        if conversation and conversation.messages:
            for msg in conversation.messages:
                with st.chat_message(msg.role):
                    st.markdown(msg.content)

    # Input for new message
    if prompt := st.chat_input("Ask a question about this project...", key=f"chat_input_history_{project.id}"):
        conv_mgr.add_message(conversation, role="user", content=prompt)

        # Get AI response
        with st.spinner("AI is thinking..."):
            # In a real app, this would call the UserInteractionAgent
            # For now, we simulate a response.
            llm_client = GeminiClient()
            prompt_manager = PromptManager()
            # ai_response = f"[Simulated AI response to: {prompt}]"
            # This is where you would run the real agent
            from src.agents.interactive.user_interaction_agent import UserInteractionAgent
            interaction_agent = UserInteractionAgent(
                llm_client=llm_client, 
                prompt_manager=prompt_manager,
                project_path=Path(f"output/{project.id}"),
                conversation=conversation
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_response = loop.run_until_complete(interaction_agent.handle_user_request(prompt))
            loop.close()

            st.session_state.project_chats[project.id].add_message(conversation, role="assistant", content=ai_response)

        # Save the conversation and rerun the UI
        conv_mgr._save_conversation(conversation)
        st.rerun()


def render_interview_prep(project: Project):
    """Renders the interview prep interface for a specific project."""
    st.markdown("---")
    st.markdown(f"### 🎤 Interview Prep for: {project.name}")
    st.info("Generate interview questions based on this project's code and description.")

    if st.button("🚀 Generate Questions", key=f"gen_interview_{project.id}", use_container_width=True, type="primary"):
        with st.spinner("🤖 Analyzing project code and generating questions..."):
            try:
                # 1. Initialize agent
                llm_client = GeminiClient()
                prompt_manager = PromptManager()
                agent = ComplexInterviewAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                
                # 2. --- NEW: Load actual project code ---
                # Use the actual project path from the project object
                project_path = Path(project.path)
                code_context = _load_project_code(project_path)
                
                if "not found" in code_context:
                    st.error(f"Could not load project code from {project_path / 'src'}. Using description only.")

                # Create a rich topic string
                project_topic = f"""
                Project Name: {project.name}
                Description: {project.description}
                Project Type: {project.project_type}

                --- PROJECT SOURCE CODE ---
                {code_context}
                """
                # --- END NEW ---

                # 3. Run agent
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                questions_list = loop.run_until_complete(
                    agent.generate_interview_questions(
                        role="AI/ML Engineer",
                        topic=project_topic, # Pass the rich topic
                        difficulty="Medium",
                        num_questions=15
                    )
                )
                loop.close()
                
                if questions_list and "Error" not in questions_list[0].get("category", ""):
                    st.success(f"Generated {len(questions_list)} questions!")
                    # 4. Format as Markdown
                    md_content = f"# Interview Prep for {project.name}\n\n"
                    md_content += f"**Project Description:** {project.description}\n\n---\n\n"
                    
                    for i, q in enumerate(questions_list, 1):
                        md_content += f"## ❓ Question {i}: {q.get('question', 'N/A')}\n\n"
                        md_content += f"**Category:** {q.get('category', 'N/A')} | **Difficulty:** {q.get('difficulty', 'N/A')}\n\n"
                        md_content += "**🔑 Key Points to Cover:**\n"
                        for point in q.get('answer_key', []):
                            md_content += f"- {point}\n"
                        md_content += "\n**🔄 Follow-up Questions:**\n"
                        for followup in q.get('follow_ups', []):
                            md_content += f"- *{followup}*\n"
                        md_content += "\n---\n"
                    
                    # 5. Save the .md file
                    doc_path = Path(f"output/{project.id}/docs")
                    doc_path.mkdir(parents=True, exist_ok=True)
                    file_path = doc_path / "generated_interview_prep.md"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(md_content)
                    
                    st.info(f"💾 Questions saved to `{file_path}`")
                    
                    # 6. Store in session state for display
                    st.session_state.generated_questions[project.id] = md_content

                else:
                    st.error(f"Agent failed to generate questions: {questions_list[0].get('question', 'Unknown error')}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())
    
    # Display the generated questions if they exist in session state
    if project.id in st.session_state.generated_questions:
        st.markdown("---")
        with st.container(height=400, border=True):
            st.markdown(st.session_state.generated_questions[project.id])