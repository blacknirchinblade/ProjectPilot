"""
AutoCoder - Streamlit Web UI

A beautiful, interactive web interface for AI-powered code generation with:
- Animated agent visualizations
- Project generation and management
- Download generated projects
- Project history and resume capability
- Upload and work with existing projects
- Real-time progress tracking

Author: GaneshS271123
gmail:ganeshnaik214@gmail.com

"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import zipfile
import shutil
from datetime import datetime
import asyncio
from typing import Dict, List, Any, Optional
import warnings
import os
from loguru import logger
import traceback
import uuid

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st_lottie = None
    logger.warning("streamlit_lottie not installed. Animations will be disabled.")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# --- UI Components & Pages ---
from src.ui.components import render_project_card
from src.ui.pages.home_page import render_home_page
from src.ui.pages.history_page import render_history_page
from src.ui.pages.upload_page import render_upload_page
from src.ui.pages.interview_page import render_interview_page
from src.ui.pages.notebook_page import render_notebook_page
from src.ui.pages.settings_page import render_settings_page
from src.ui.pages.dataset_page import render_dataset_generation_page

# --- UI Utilities & Styling ---
from src.ui.styles import CUSTOM_CSS
from src.ui.utils import show_lottie_animation, LOTTIE_ANIMATIONS, strip_markdown_code_fences
from src.ui.project_manager import ProjectManager
from src.ui.project import Project


# --- Core LLM & Memory ---
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.memory.shared_memory import SharedMemory


# --- Core Agents ---
from src.agents.coding.coding_agent import CodingAgent
from src.agents.interactive.clarification_agent import ClarificationAgent
from src.agents.prompt_engineering.prompt_engineer_agent import PromptEngineerAgent
from src.agents.testing.testing_agent import TestingAgent
from src.agents.documentation.advanced_doc_agent import AdvancedDocumentationAgent
from src.agents.review.review_orchestrator import ReviewOrchestrator
from src.agents.refactoring.refactoring_agent import RefactoringAgent
from src.agents.interactive.user_interaction_agent import UserInteractionAgent
from src.agents.interactive.conversation_manager import ConversationManager, Conversation
from src.agents.architecture.dynamic_architect_agent import DynamicArchitectAgent, ProjectArchitecture
from src.agents.debugging.debugging_agent import DebuggingAgent

# --- Specialized Agents ---
from src.agents.specialized.database_agent import DatabaseAgent
from src.agents.specialized.api_agent import APIAgent
from src.agents.specialized.streamlit_agent import StreamlitAgent
from src.agents.specialized.genai_agent import GenAIAgent
from src.agents.specialized.deployment_agent import DeploymentAgent
from src.agents.dataset.dataset_agent import DatasetAgent
from src.agents.document.document_analyzer_agent import DocumentAnalyzerAgent
from src.agents.notebook.notebook_agent import NotebookAgent
from src.agents.interview.complex_interview_agent import ComplexInterviewAgent


# --- Reviewer Agents ---
from src.agents.review.readability_reviewer import ReadabilityReviewer
from src.agents.review.logic_flow_reviewer import LogicFlowReviewer
from src.agents.review.code_connectivity_reviewer import CodeConnectivityReviewer
from src.agents.review.project_connectivity_reviewer import ProjectConnectivityReviewer
from src.agents.review.performance_reviewer import PerformanceReviewer
from src.agents.review.security_reviewer import SecurityReviewer
from src.agents.review.scoring_system import ScoringSystem

# --- Tools ---
from src.tools.dependency_manager import DependencyManager
from src.tools.problem_checker import ProblemChecker
from src.tools.setup_automation_tool import SetupAutomationTool, SetupConfig
from src.tools.data_structures import ProjectType, EnvironmentType, ConfigFileType
from src.tools.search_refactor_tool import SearchRefactorTool
from src.tools.project_structure import ProjectStructure
from src.tools.todo_manager import TodoManager
from src.tools.cross_file_impact_analyzer import CrossFileImpactAnalyzer
from src.tools.test_failure_handler import TestFailureHandler
from src.tools.task_runner import TaskRunner
from src.tools.task_tracker import TaskTracker, Task, TaskType, create_tasks_from_architecture, load_tracker
from src.agents.interactive.modification_agent import InteractiveModificationAgent
from src.utils.logger import setup_logger

# -----Configuration and setup----- #
st.set_page_config(page_title="ProjectPilot - AI Code Generator", page_icon="🤖", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'project_manager' not in st.session_state:
    st.session_state.project_manager = ProjectManager()
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'active_agents' not in st.session_state:
    st.session_state.active_agents = []
if 'clarification_answers' not in st.session_state:
    st.session_state.clarification_answers = {}
if 'clarification_questions' not in st.session_state: # Ensure this exists
    st.session_state.clarification_questions = None
if 'project_specification' not in st.session_state: # Ensure this exists
     st.session_state.project_specification = None
if 'project_chats' not in st.session_state: # Ensure this exists for chat
     st.session_state.project_chats = {}
if 'generated_project_id' not in st.session_state:
    st.session_state.generated_project_id = {}
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None




def generate_tree_structure(path, prefix: str = ""):
    """Return a string representing the tree structure for the given path.
    Accepts a Path or a string path. Returns empty string on error.
    """
    try:
        p = Path(path)
        if not p.exists():
            return ""
        
        # Get items, ignoring common noise
        items = sorted(

            [
                item for item in p.iterdir() 
                if item.name not in ['.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'output', 'data', 'logs', 'chat_history']
            ], 
            key=lambda x: (x.is_file(), x.name.lower())
            
        )
    except Exception as e:
        logger.warning(f"Could not generate tree structure for {path}: {e}")
        return ""

    lines = []
    for idx, item in enumerate(items):
        is_last = idx == (len(items) - 1)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{item.name}")
        if item.is_dir():
            extension = "    " if is_last else "│   "
            subtree = generate_tree_structure(item, prefix + extension)
            if subtree:
                lines.append(subtree)

    return "\n".join(lines)

if 'active_agents' not in st.session_state:
    st.session_state.active_agents = []


def main():
    """Main application entry point."""


    if 'current_project_id' in st.session_state and st.session_state.current_project_id:
        
        project_id = st.session_state.current_project_id
        if project_id in st.session_state.project_chats:
            pass
    try:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            show_lottie_animation(LOTTIE_ANIMATIONS["ai_brain"], height=120, key="header_brain")
        with col2:
            st.markdown("""
            <div class="main-header">
                <h1>🧠 ProjectPilot</h1>
                <p>Your Unified AI/ML Project Generation Copilot</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            show_lottie_animation(LOTTIE_ANIMATIONS["rocket"], height=120, key="header_rocket")

        with st.sidebar:
            st.markdown("### 🎯 Navigation")
            page_options = [
                "🏠 Home", 
                "✨ Generate Code", 
                "📚 Project History", 
                "🎤 Interview Questions", 
                "📤 Upload Project",
                "📊 Notebook EDA",
                "🧬 Generate Dataset",
                "⚙️ Settings"
            ]
            page = st.radio(
                "Select Page",
                page_options,
                label_visibility="collapsed"
            )
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            projects = st.session_state.project_manager.get_all_projects()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Projects", len(projects))
            with col2:
                st.metric("Agents", "25+")
            st.markdown("---")
            if st.session_state.active_agents:
                st.markdown("### 🤖 Active Agents")
                for agent in st.session_state.active_agents:
                    st.markdown(f"""
                    <div class="agent-card active">
                        <strong>{agent['name']}</strong><br>
                        <small>{agent['status']}</small>
                    </div>
                    """, unsafe_allow_html=True)

        if page == "🏠 Home":
            render_home_page()
        elif page == "✨ Generate Code":
            render_generate_page()
        elif page == "📚 Project History":
            render_history_page()
        elif page == "📤 Upload Project":
            render_upload_page()
        elif page == "🎤 Interview Questions":
            render_interview_page()
        elif page == "📊 Notebook EDA": # This page is not defined in the radio list
            render_notebook_page()
        elif page == "🧬 Generate Dataset":
            render_dataset_generation_page()                    
        elif page == "⚙️ Settings":
            render_settings_page()
            
    except Exception as e:
        st.error(f"An error occurred in main: {e}")
        st.code(traceback.format_exc())





def render_generate_page():
    """Render the code generation page."""
    st.markdown("## ✨ Generate Code")
    
    # Simplified tabs - focus on ML/AI projects
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI/ML Project Generator",
        "📄 Simple Code Generation",
        "💾 Database Schema",
        "🔌 API Endpoints"
    ])
    
    with tab1:
        render_ai_ml_project_generator()
    
    with tab2:
        render_simple_generation()
    
    with tab3:
        render_database_generation()

    with tab4:
        render_api_generation()


def render_simple_generation():
    """Render simple code generation interface."""
    st.markdown("### Generate a Single File")
    
    # Input form
    with st.form("simple_generation_form"):
        description = st.text_area(
            "Describe what you want to generate:",
            placeholder="e.g., A calculator class with add, subtract, multiply, divide methods",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            file_name = st.text_input("File name:", value="generated.py")
        with col2:
            include_tests = st.checkbox("Include tests (coming soon)", value=False, disabled=True)
        
        submitted = st.form_submit_button("🚀 Generate", type="primary", use_container_width=True)
    
    if submitted and description:
        with st.spinner("Generating code..."):
            st.session_state.active_agents = [
                {"name": "Coding Agent", "status": "Analyzing requirements..."}
            ]
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate generation steps
            steps = [
                ("Analyzing requirements", 20),
                ("Planning code structure", 40),
                ("Generating code", 60),
                ("Validating syntax", 80),
                ("Finalizing", 100)
            ]
            
            for step, progress in steps:
                status_text.text(f"⚡ {step}...")
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            # Generate code
            try:
                status_text.text("🤖 Initializing Coding Agent...")
                
                # Initialize agents
                llm_client = GeminiClient()
                prompt_manager = PromptManager()
                agent = CodingAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                
                status_text.text("🔄 Creating event loop...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                status_text.text("⚡ Generating code with AI...")
                try:
                    result = loop.run_until_complete(agent.execute_task({
                        "task_type": "generate_module", # Using legacy task type for simple generation
                        "data": {
                            "module_name": Path(file_name).stem,
                            "purpose": description,
                            "requirements": description
                        }
                    }))
                finally:
                    loop.close()
                
                status_text.text("✅ Processing results...")
                if result["status"] == "success":
                    code = result["code"]
                    
                    st.session_state.active_agents = []
                    
                    st.success("✅ Code generated successfully!")
                    
                    st.markdown("### 📝 Generated Code")
                    st.code(code, language="python", line_numbers=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lines", len(code.split('\n')))
                    with col2:
                        st.metric("Characters", len(code))
                    with col3:
                        st.metric("Attempts", result.get("attempts", 1))
                    
                    st.download_button(
                        label="📥 Download Code",
                        data=code,
                        file_name=file_name,
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "simple",
                        "description": description,
                        "file_name": file_name,
                        "code": code
                    })
                else:
                    st.error(f"❌ Generation failed: {result.get('message', 'Unknown error')}")
                    st.error(f"Debug info: {result}")
            
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")
                st.code(traceback.format_exc(), language="python")
                st.session_state.active_agents = []


def render_full_project_generation():
    """Render full project generation interface."""
    st.markdown("### 🏗️ Generate Complete Project")
    
    st.info("🚀 Generate a complete, production-ready project with backend API, database, tests, Docker, and documentation!")
    
    with st.form("full_project_form"):
        # Project basics
        st.markdown("#### 📋 Project Information")
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input(
                "Project Name:",
                value="my_api_project",
                help="Use lowercase with underscores (e.g., task_manager_api)"
            )
            project_description = st.text_area(
                "Description:",
                value="A REST API for task management",
                height=80
            )
        
        with col2:
            framework = st.selectbox(
                "Framework:",
                ["FastAPI", "Flask"]
            )
            database = st.selectbox(
                "Database:",
                ["SQLite", "PostgreSQL", "MySQL"]
            )
        
        # Features
        st.markdown("#### ✨ Features to Include")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_auth = st.checkbox("🔐 Authentication (JWT)", value=True)
            include_docker = st.checkbox("🐳 Docker", value=True)
        
        with col2:
            include_tests = st.checkbox("🧪 Unit Tests", value=True)
            include_alembic = st.checkbox("📊 Database Migrations", value=True)
        
        with col3:
            include_cors = st.checkbox("🌐 CORS", value=True)
            include_docs = st.checkbox("📝 Documentation", value=True)
        
        # Models/Resources
        st.markdown("#### 📦 Data Models")
        num_models = st.number_input(
            "Number of models:",
            min_value=1,
            max_value=5,
            value=2,
            help="E.g., User, Task, Project"
        )
        
        models = []
        for i in range(int(num_models)):
            st.markdown(f"**Model {i+1}**")
            col_a, col_b, col_c = st.columns([2, 1, 2])
            
            with col_a:
                model_name = st.text_input(
                    "Model Name:",
                    value=["User", "Task", "Project", "Comment", "Tag"][i],
                    key=f"model_name_{i}"
                )
            
            with col_b:
                num_fields = st.number_input(
                    "Fields:",
                    min_value=2,
                    max_value=10,
                    value=4,
                    key=f"model_fields_{i}"
                )
            
            with col_c:
                has_api = st.checkbox(
                    "Generate API routes",
                    value=True,
                    key=f"model_api_{i}"
                )
            
            if model_name:
                models.append({
                    "name": model_name,
                    "num_fields": int(num_fields),
                    "has_api": has_api
                })
        
        st.markdown("#### 4️⃣ (Optional) Upload Documents for Context")
        st.session_state.uploaded_files = st.file_uploader(
            "Upload research papers, specs, or other documents to provide context to the AI.",
            accept_multiple_files=True,
            type=['pdf', 'md', 'txt', 'html']
        )

        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button(
            "🚀 Generate Complete Project",
            type="primary",
            use_container_width=True
        )
    
    if submitted and project_name and models:
        # This part is now handled by the render_ai_ml_project_generator
        # This function (render_full_project_generation) is now OBSOLETE
        # and replaced by render_ai_ml_project_generator
        st.warning("This generation method is deprecated. Please use the 'AI/ML Project Generator' tab.")
        pass


def generate_default_fields(model_name: str, num_fields: int) -> List[Dict]:
    """Generate default fields for a model."""
    base_fields = [
        {"name": "id", "type": "Integer", "nullable": False},
        {"name": "created_at", "type": "DateTime", "nullable": False},
        {"name": "updated_at", "type": "DateTime", "nullable": False},
    ]
    
    # Add model-specific fields
    if model_name.lower() == "user":
        custom_fields = [
            {"name": "email", "type": "String", "nullable": False},
            {"name": "username", "type": "String", "nullable": False},
            {"name": "hashed_password", "type": "String", "nullable": False},
            {"name": "is_active", "type": "Boolean", "nullable": False},
        ]
    elif model_name.lower() == "task":
        custom_fields = [
            {"name": "title", "type": "String", "nullable": False},
            {"name": "description", "type": "Text", "nullable": True},
            {"name": "status", "type": "String", "nullable": False},
            {"name": "priority", "type": "Integer", "nullable": False},
        ]
    else:
        custom_fields = [
            {"name": "name", "type": "String", "nullable": False},
            {"name": "description", "type": "Text", "nullable": True},
            {"name": "status", "type": "String", "nullable": False},
        ]
    
    # Limit to requested number of fields
    return base_fields + custom_fields[:num_fields - len(base_fields)]


def generate_main_app(framework, project_name, models, include_auth, include_cors, database):
    """Generate main application file."""
    # This is a simplified placeholder. The actual generation is done by agents.
    if framework == "FastAPI":
        return f"""
from fastapi import FastAPI
app = FastAPI(title="{project_name}")

@app.get("/")
def read_root():
    return {{"message": "Welcome to {project_name}"}}
"""
    else: # Flask
        return f"""
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, {project_name}!'
"""


def generate_requirements(framework, database, include_auth, include_tests, include_alembic):
    reqs = []
    # Framework
    if framework == "FastAPI":
        reqs.extend([
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.4.0"
        ])
    else:
        reqs.extend([
            "flask>=3.0.0",
            "flask-restful>=0.3.10"
        ])
    # Database
    reqs.append("sqlalchemy>=2.0.0")
    if database == "PostgreSQL":
        reqs.append("psycopg2-binary>=2.9.0" if framework == "Flask" else "asyncpg>=0.29.0")
    elif database == "MySQL":
        reqs.append("pymysql>=1.1.0" if framework == "Flask" else "aiomysql>=0.2.0")
    else:  # SQLite
        reqs.append("aiosqlite>=0.19.0" if framework == "FastAPI" else "")
    # Auth
    if include_auth:
        if framework == "FastAPI":
            reqs.extend([
                "python-jose[cryptography]>=3.3.0",
                "passlib[bcrypt]>=1.7.4",
                "python-multipart>=0.0.6"
            ])
        else:
            reqs.extend([
                "flask-jwt-extended>=4.5.0",
                "bcrypt>=4.1.0"
            ])
    # Migrations
    if include_alembic:
        reqs.append("alembic>=1.12.0")
    # Testing
    if include_tests:
        reqs.extend([
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "httpx>=0.25.0" if framework == "FastAPI" else "pytest-flask>=1.3.0"
        ])
    # Utilities
    reqs.extend([
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0"
    ])
    return "\n".join(sorted(filter(None, reqs)))


def generate_env_example(database, include_auth):
    '''Generate .env.example file.'''
    content = f"""# Database Configuration\nDATABASE_URL={'sqlite:///./app.db' if database == 'SQLite' else 'postgresql://user:password@localhost/dbname' if database == 'PostgreSQL' else 'mysql://user:password@localhost/dbname'}\n\n# Application Settings\nAPP_NAME=My API\nDEBUG=True\n"""
    if include_auth:
        content += """# Security\nSECRET_KEY=your-super-secret-key-change-this-in-production\nALGORITHM=HS256\nACCESS_TOKEN_EXPIRE_MINUTES=30\n"""
    return content


def generate_gitignore():
    """Generate .gitignore file."""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
.venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local

# Database
*.db
*.sqlite3

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Project specific
output/
data/memory/
"""

def generate_dockerfile(framework):
    """Generate Dockerfile."""
    if framework == "FastAPI":
        return """FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"""
    else:
        return """FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"flask\", \"run\", \"--host\", \"0.0.0.0\"]\n"""


def generate_docker_compose(project_name, database):
    """Generate docker-compose.yml."""
    db_service = ""
    if database == "PostgreSQL":
        db_service = f'''
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: {project_name}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
'''
    elif database == "MySQL":
        db_service = f'''
  mysql:
    image: mysql:8
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: {project_name}
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
'''
    
    return f'''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${{DATABASE_URL}}
    depends_on:
      - {"postgres" if database == "PostgreSQL" else "mysql" if database == "MySQL" else "none"}
{db_service}
{"volumes:" if database in ["PostgreSQL", "MySQL"] else ""}
{f"  {'postgres' if database == 'PostgreSQL' else 'mysql'}_data:" if database in ["PostgreSQL", "MySQL"] else ""}
'''


def generate_readme(project_name, description, framework, database, models, include_auth, include_docker, include_tests):
    """Generate README.md content."""
    models_list = "\n".join(
        [f"- **{m.get('name','Model')}** - {m.get('num_fields',0)} fields" for m in models]
    )

    readme = f"""# {project_name}

{description}

**Auto-generated by AutoCoder**

## 🚀 Features

- ✅ {framework} REST API
- ✅ {database} Database
- ✅ SQLAlchemy ORM
- {'✅ JWT Authentication' if include_auth else ''}
- {'✅ Docker Support' if include_docker else ''}
- {'✅ Unit Tests' if include_tests else ''}
- ✅ {len([m for m in models if m.get('has_api')])} API Resources

## 📦 Models

{models_list}

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

"""

    if include_docker:
        readme += """
### Docker

```bash
docker-compose up -d
```
"""

    readme += f"""
## 📚 API Documentation

{'- Interactive Docs: http://localhost:8000/docs' if framework.lower() == 'fastapi' else ''}
{'- ReDoc: http://localhost:8000/redoc' if framework.lower() == 'fastapi' else ''}
- Health Check: http://localhost:{'8000' if framework.lower() == 'fastapi' else '5000'}/health

## 🧪 Testing

"""
    if include_tests:
        readme += """
```bash
pytest
pytest --cov
```
"""
    else:
        readme += "Testing setup coming soon!\n"

    readme += f"""
## 📁 Project Structure

```
{project_name}/
├── app/
│   ├── models/          # Database models
│   ├── routes/          # API routes
│   ├── schemas/         # Pydantic schemas
│   ├── core/            # Core configuration
│   └── main.py          # Application entry point
├── tests/               # Unit tests
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
{'├── Dockerfile          # Docker configuration' if include_docker else ''}
{'├── docker-compose.yml  # Docker Compose' if include_docker else ''}
└── README.md            # This file
```

## 📝 License
"""

    return readme


    output = []
    extension = "    " if prefix else ""
    items = list(path.iterdir())
    for idx, item in enumerate(items):
        is_last_item = idx == len(items) - 1
        if item.is_dir():
            output.append(generate_tree_structure(item, prefix + extension, is_last_item))
        else:
            connector = "└── " if is_last_item else "├── "
            output.append(f"{prefix}{extension}{connector}{item.name}")
    return "\n".join(output)


def render_database_generation():
    """Render database schema generation interface."""
    st.markdown("### Generate Database Schema")
    
    with st.form("database_generation_form"):
        model_name = st.text_input("Model Name:", value="User")
        table_name = st.text_input("Table Name:", value="users")
        
        st.markdown("#### Fields")
        num_fields = st.number_input("Number of fields:", min_value=1, max_value=20, value=5)
        
        fields = []
        for i in range(int(num_fields)):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                field_name = st.text_input(f"Field {i+1} name:", key=f"field_name_{i}")
            with col2:
                field_type = st.selectbox(
                    f"Type:",
                    ["String", "Integer", "Boolean", "DateTime", "Text", "Float", "ForeignKey"],
                    key=f"field_type_{i}"
                )
            with col3:
                nullable = st.checkbox("Null?", key=f"nullable_{i}")
            
            if field_name:
                fields.append({
                    "name": field_name,
                    "type": field_type,
                    "nullable": nullable
                })
        
        submitted = st.form_submit_button("🚀 Generate Model", type="primary", use_container_width=True)

    if submitted and model_name and fields:
        try:
            with st.spinner("Generating SQLAlchemy model..."):
                llm_client = GeminiClient()
                prompt_manager = PromptManager()
                agent = DatabaseAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = None
                try:
                    result = loop.run_until_complete(agent.generate_sqlalchemy_model({
                        "model_name": model_name,
                        "table_name": table_name,
                        "fields": fields,
                        "relationships": [],
                        "indexes": [],
                        "constraints": []
                    }))
                finally:
                    loop.close()
                
                if result and result["status"] == "success":
                    code = result["code"]
                    st.success("✅ Model generated successfully!")
                    st.markdown("### 📝 Generated Model")
                    st.code(code, language="python", line_numbers=True)
                    
                    st.download_button(
                        label="📥 Download Model",
                        data=code,
                        file_name=f"{model_name.lower()}_model.py",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    error_message = result.get('message', 'Unknown error') if result else 'Operation failed without a result'
                    st.error(f"❌ Generation failed: {error_message}")
        
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

def render_api_generation():
    """Render API generation interface."""
    st.markdown("### Generate REST API")
    
    st.info("🚀 Generate production-ready FastAPI or Flask REST APIs with authentication, database integration, and comprehensive documentation!")
    
    # API Type Selection
    api_type = st.radio(
        "Select API Type:",
        ["FastAPI Application", "Flask Application", "API Routes Only", "Pydantic Schemas", "JWT Authentication"],
        horizontal=True
    )
    llm_client = GeminiClient()
    prompt_manager = PromptManager()
    
    if api_type == "FastAPI Application":
        render_fastapi_generation(llm_client, prompt_manager)
    elif api_type == "API Routes Only":
        render_routes_generation(llm_client, prompt_manager)
    elif api_type == "Pydantic Schemas":
        render_schemas_generation(llm_client, prompt_manager)
    elif api_type == "JWT Authentication":
        render_auth_generation(llm_client, prompt_manager)


def render_fastapi_generation(llm_client, prompt_manager):
    """Render FastAPI application generation form."""
    st.markdown("#### 🚀 FastAPI Application")

    with st.form("fastapi_generation_form"):
        col1, col2 = st.columns(2)

        with col1:
            app_name = st.text_input("Application Name:", value="TaskAPI")
            description = st.text_area("Description:", value="A task management API", height=80)
            auth_type = st.selectbox("Authentication:", ["none", "jwt", "oauth2", "api_key"])

        with col2:
            database = st.selectbox("Database:", ["sqlite", "postgresql", "mysql"])
            include_cors = st.checkbox("Enable CORS", value=True)

        st.markdown("##### Endpoints")
        num_endpoints = st.number_input("Number of endpoints:", min_value=0, max_value=10, value=2)

        endpoints = []
        if num_endpoints > 0:
            for i in range(int(num_endpoints)):
                col_a, col_b, col_c = st.columns([1, 2, 3])
                with col_a:
                    method = st.selectbox(f"Method {i+1}:", ["GET", "POST", "PUT", "DELETE", "PATCH"], key=f"api_method_{i}")
                with col_b:
                    path = st.text_input(f"Path {i+1}:", value=("/items" if i == 0 else "/items/{id}"), key=f"api_path_{i}")
                with col_c:
                    desc = st.text_input(f"Description {i+1}:", value=("List items" if i == 0 else "Get item"), key=f"api_desc_{i}")
                if path:
                    endpoints.append({"method": method, "path": path, "description": desc})

        # Collect config for NameError
        config = {
            "database_type": database,
            "include_auth": auth_type != "none"
        }
        submitted = st.form_submit_button("🚀 Generate FastAPI App")

    if submitted:
        with st.spinner("🤖 Generating FastAPI application..."):
            agent = APIAgent(llm_client, prompt_manager)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(agent.generate_fastapi_app({
                    "app_name": app_name,
                    "description": description,
                    "endpoints": [],
                    "auth_type": auth_type,
                    "database": config.get("database_type", "none"),
                    "include_cors": True
                }))
            finally:
                loop.close()

            if result.get("status") == "success":
                code = result.get("code", "")
                st.success(f"✅ {app_name} generated successfully!")
                st.code(code, language="python")
                st.download_button("📥 Download FastAPI App", data=code, file_name=f"{app_name.lower()}_fastapi.py", mime="text/plain")
            else:
                st.error(f"❌ Generation failed: {result.get('message', 'Unknown error')}")


def render_flask_generation():
    """Render Flask application generation form."""
    st.markdown("#### 🍶 Flask Application")
    st.info("Flask generation coming soon!")


def render_routes_generation(llm_client, prompt_manager):
    """Render API routes generation form."""
    st.markdown("#### 🛣️ API Routes")
    
    with st.form("routes_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            resource_name = st.text_input("Resource Name:", value="User")
            framework = st.selectbox("Framework:", ["fastapi", "flask"])
            with col1:
                resource_name = st.text_input("Resource Name:", value="User")
                framework = st.selectbox("Framework:", ["fastapi", "flask"])
    
            with col2:
                operations = st.multiselect(
                    "CRUD Operations:",
                    options=["create", "read", "update", "delete", "list"],
                    default=["create", "read", "update", "delete", "list"]
                )

            
        submitted = st.form_submit_button("🚀 Generate Model", type="primary", use_container_width=True)
    if submitted and resource_name and operations:
        with st.spinner("Generating API routes..."):
            try:
                agent = APIAgent(llm_client, prompt_manager)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = None # Initialize
                try:
                    result = loop.run_until_complete(agent.generate_routes({
                        "resource_name": resource_name,
                        "operations": operations,
                        "framework": framework
                    }))
                finally:
                    loop.close()
                
                if result and result["status"] == "success":
                    st.success(f"✅ {framework.capitalize()} routes for {resource_name} generated!")
                    st.code(result["code"], language="python", line_numbers=True)
                    st.download_button(
                        label=f"📥 Download {result['file_name']}",
                        data=result["code"],
                        file_name=result['file_name'],
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    error_message = result.get('message', 'Unknown error') if result else 'Operation failed'
                    st.error(f"❌ Failed: {error_message}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def render_schemas_generation(llm_client, prompt_manager):
    """Render Pydantic schemas generation form."""
    st.markdown("#### 📋 Pydantic Schemas")
    
    with st.form("schemas_generation_form"):
        model_name = st.text_input("Model Name:", value="Product")
        
        st.markdown("##### Fields")
        num_fields = st.number_input("Number of fields:", min_value=1, max_value=15, value=5)
        
        fields = []
        for i in range(int(num_fields)):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                field_name = st.text_input(f"Field {i+1}:", key=f"schema_field_{i}")
            with col2:
                field_type = st.selectbox(
                    f"Type:",
                    ["str", "int", "float", "bool", "datetime", "list", "dict"],
                    key=f"schema_type_{i}"
                )
            with col3:
                required = st.checkbox("Required", value=True, key=f"schema_req_{i}")
            
            if field_name:
                fields.append({
                    "name": field_name,
                    "type": field_type,
                    "required": required
                })
        
        include_examples = st.checkbox("Include example values", value=True)
        
        submitted = st.form_submit_button(
            "🚀 Generate Schemas",
            type="primary",
            use_container_width=True
        )
    
    if submitted and fields:
        with st.spinner("Generating Pydantic schemas..."):
            try:
                agent = APIAgent(llm_client, prompt_manager)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(agent.generate_schemas({
                        "model_name": model_name,
                        "fields": fields,
                        "include_examples": include_examples
                    }))
                finally:
                    loop.close()
                
                if result["status"] == "success":
                    st.success(f"✅ Schemas for {model_name} generated!")
                    st.code(result["code"], language="python", line_numbers=True)
                    
                    st.download_button(
                        label=f"📥 Download {result['file_name']}",
                        data=result["code"],
                        file_name=result['file_name'],
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error(f"❌ Failed: {result.get('message')}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def render_auth_generation(llm_client, prompt_manager):
    """Render authentication generation form."""
    st.markdown("#### 🔐 JWT Authentication")
    
    with st.form("auth_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            auth_type = st.selectbox(
                "Authentication Type:",
                ["jwt", "oauth2", "api_key"]
            )
        
        with col2:
            framework = st.selectbox(
                "Framework:",
                ["fastapi", "flask"]
            )
        
        st.markdown("""
        **What will be generated:**
        - Token generation and validation
        - Password hashing (bcrypt)
        - Login/logout endpoints
        - Protected route decorators
        - User authentication dependencies
        """)
        
        submitted = st.form_submit_button(
            "🚀 Generate Authentication",
            type="primary",
            use_container_width=True
        )
    
    if submitted:
        with st.spinner("Generating authentication code..."):
            try:
                agent = APIAgent(llm_client, prompt_manager)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(agent.generate_auth({
                        "auth_type": auth_type,
                        "framework": framework
                    }))
                finally:
                    loop.close()
                
                if result["status"] == "success":
                    st.success(f"✅ {auth_type.upper()} authentication generated!")
                    st.code(result["code"], language="python", line_numbers=True)
                    
                    st.download_button(
                        label=f"📥 Download {result['file_name']}",
                        data=result["code"],
                        file_name=result['file_name'],
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error(f"❌ Failed: {result.get('message')}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def render_ai_ml_project_generator():
    #  Do not reference project_description before it is defined
    """
    Unified AI/ML Project Generator
    
    Single interface for ALL ML/AI projects:
    - Computer Vision (Image Classification, Object Detection, etc.)
    - NLP (Text Classification, NER, Sentiment Analysis, etc.)
    - Time Series (Forecasting, Anomaly Detection, etc.)
    - GenAI (RAG Systems, LLM Fine-tuning, Agents, etc.)
    
    User describes project → Clarification Agent asks questions → Generate complete project
    """
    st.markdown("### 🤖 AI/ML Project Generator")
    
    st.info("""
    🎯 **Describe your AI/ML project and get a complete, production-ready codebase!**
    
    **What you get:**
    - ✅ ML Model Training Code (PyTorch/TensorFlow)
    - ✅ FastAPI Backend (Model Serving + REST API)
    - ✅ Streamlit Dashboard (Interactive UI)
    - ✅ Docker Deployment (Multi-container)
    - ✅ Complete Documentation
    
    **How it works:**
    1. Describe your project in natural language
    2. Select components you need (checkboxes below)
    3. Our AI asks clarifying questions
    4. Get complete, ready-to-deploy project!
    """)
    
    # Main project description
    st.markdown("### 1️⃣ Describe Your Project")
    
    project_description = st.text_area(
        "What do you want to build? (Be as detailed as possible)",
        placeholder="""Examples:
• "Build a CIFAR-10 image classifier using ResNet-18 with data augmentation, trained for 50 epochs"
• "Create a sentiment analysis model for movie reviews using BERT, with FastAPI serving and Streamlit UI"
• "Develop a RAG system for customer support using LangChain, ChromaDB, and OpenAI GPT-4"
• "Build a time series forecasting model for stock prices using LSTM with 30-day prediction window"
• "Create an object detection system for cars and pedestrians using YOLOv8"
        """,
        height=250,
        help="Describe your project: dataset, model type, task, performance goals, etc."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input(
            "Project Name:",
            value="my_ai_project",
            help="Name for your project (used for directories, Docker images, etc.)"
        )
    
    with col2:
        # Auto-detect project category from description
        if project_description:
            description_lower = project_description.lower()
            if any(word in description_lower for word in ['rag', 'retrieval', 'langchain', 'vector', 'embeddings']):
                default_category = "GenAI - RAG System"
            elif any(word in description_lower for word in ['fine-tun', 'lora', 'qlora', 'llm', 'gpt', 'llama']):
                default_category = "GenAI - LLM Fine-tuning"
            elif any(word in description_lower for word in ['yolo', 'detection', 'detect', 'bounding box']):
                default_category = "Computer Vision - Object Detection"
            elif any(word in description_lower for word in ['image', 'classification', 'resnet', 'cnn', 'vision']):
                default_category = "Computer Vision - Image Classification"
            elif any(word in description_lower for word in ['ner', 'entity', 'token']):
                default_category = "NLP - Named Entity Recognition"
            elif any(word in description_lower for word in ['sentiment', 'text class', 'bert', 'nlp']):
                default_category = "NLP - Text Classification"
            elif any(word in description_lower for word in ['time series', 'forecast', 'lstm', 'stock', 'prediction']):
                default_category = "Time Series - Forecasting"
            else:
                default_category = "Custom Project"
        else:
            default_category = "Custom Project"
        
        project_category = st.selectbox(
            "Project Category (auto-detected):",
            [
                "Computer Vision - Image Classification",
                "Computer Vision - Object Detection",
                "Computer Vision - Segmentation",
                "NLP - Text Classification",
                "NLP - Named Entity Recognition",
                "NLP - Question Answering",
                "Time Series - Forecasting",
                "Time Series - Anomaly Detection",
                "GenAI - RAG System",
                "GenAI - LLM Fine-tuning",
                "GenAI - AI Agent",
                "Recommendation System",
                "Custom Project"
            ],
            index=[
                "Computer Vision - Image Classification",
                "Computer Vision - Object Detection",
                "Computer Vision - Segmentation",
                "NLP - Text Classification",
                "NLP - Named Entity Recognition",
                "NLP - Question Answering",
                "Time Series - Forecasting",
                "Time Series - Anomaly Detection",
                "GenAI - RAG System",
                "GenAI - LLM Fine-tuning",
                "GenAI - AI Agent",
                "Recommendation System",
                "Custom Project"
            ].index(default_category),
            help="Auto-detected from your description, but you can change it"
        )
    
    st.markdown("---")
    
    # Component Selection
    st.markdown("### 2️⃣ Select Components to Include")
    
    st.markdown("##### ✅ Check the components you want in your project:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🎯 Core Components**")
        enable_model = st.checkbox("ML/AI Model", value=True, disabled=True, help="Training code for your model")
        enable_fastapi = st.checkbox("FastAPI Backend", value=True, help="REST API for model serving")
        enable_streamlit = st.checkbox("Streamlit UI", value=True, help="Interactive web dashboard")
        enable_tests = st.checkbox("Unit Tests", value=True, help="Pytest test suite")
    
    with col2:
        st.markdown("**💾 Data & Storage**")
        enable_database = st.checkbox("Database", value=False, help="PostgreSQL/MongoDB for data storage")
        enable_redis = st.checkbox("Redis Cache", value=False, help="Caching layer for predictions")
        enable_vector_db = st.checkbox("Vector Database", value="GenAI" in project_category, help="ChromaDB/Pinecone for RAG", disabled="GenAI - RAG" in project_category)
        enable_data_pipeline = st.checkbox("Data Pipeline", value=False, help="ETL/data processing pipeline")
    
    with col3:
        st.markdown("**🐳 Deployment**")
        enable_docker = st.checkbox("Docker Compose", value=True, help="Multi-container deployment")
        enable_kubernetes = st.checkbox("Kubernetes", value=False, help="K8s manifests for production")
        enable_ci_cd = st.checkbox("CI/CD Pipeline", value=True, help="GitHub Actions workflow")
        enable_monitoring = st.checkbox("Monitoring", value=False, help="Prometheus + Grafana")
    
    with col4:
        st.markdown("**⚙️ Advanced**")
        enable_gpu = st.checkbox("GPU Support", value=False, help="CUDA configuration for training/inference")
        enable_api_auth = st.checkbox("API Authentication", value=False, help="JWT auth for API endpoints")
        enable_logging = st.checkbox("Advanced Logging", value=True, help="Structured logging with loguru")
        enable_docs = st.checkbox("Auto Documentation", value=True, help="Sphinx/MkDocs documentation")
    
    st.markdown("---")
    
    # Configuration options based on selections
    st.markdown("### 3️⃣ Configuration Options")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Database configuration
        if enable_database:
            database_type = st.selectbox(
                "Database Type:",
                ["postgresql", "mongodb", "mysql", "sqlite"],
                help="Which database to use for data storage"
            )
        else:
            database_type = None
        
        # Framework preference
        ml_framework = st.selectbox(
            "ML Framework:",
            ["pytorch", "tensorflow", "auto-detect"],
            index=2,
            help="Preferred deep learning framework (auto-detect analyzes your description)"
        )
    
    with config_col2:
        # Initialize GenAI variables
        llm_provider = None
        vector_store = None
        base_model = None
        finetuning_method = None
        cloud_provider = None
        
        # GenAI specific options
        if "GenAI" in project_category:
            if "RAG" in project_category:
                llm_provider = st.selectbox(
                    "LLM Provider:",
                    ["openai", "anthropic", "cohere", "huggingface", "local (Ollama)"],
                    help="Which LLM API to use"
                )
                vector_store = st.selectbox(
                    "Vector Store:",
                    ["chromadb", "faiss", "pinecone", "weaviate", "qdrant"],
                    help="Vector database for embeddings"
                )
            elif "Fine-tuning" in project_category:
                base_model = st.selectbox(
                    "Base Model:",
                    ["llama-2-7b", "mistral-7b", "phi-2", "gpt-3.5-turbo", "custom"],
                    help="Base LLM to fine-tune"
                )
                finetuning_method = st.selectbox(
                    "Fine-tuning Method:",
                    ["lora", "qlora", "full", "prefix-tuning"],
                    help="Parameter-efficient fine-tuning method"
                )
        
        # Cloud deployment
        if enable_kubernetes:
            cloud_provider = st.selectbox(
                "Cloud Provider (optional):",
                ["none", "aws", "gcp", "azure"],
                help="Generate cloud-specific configurations"
            )
    
    st.markdown("---")
    st.markdown("### 4️⃣ (Optional) Add Context Files")
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        
    st.session_state.uploaded_files = st.file_uploader(
        "Upload research papers, specs, or other documents to provide context to the AI.",
        accept_multiple_files=True,
        type=['pdf', 'md', 'txt', 'html']
    )
    
    # Generation preview
    st.markdown("### 📊 Generation Preview")
    
    if project_description:
        components_count = sum([
            enable_model, enable_fastapi, enable_streamlit, enable_database,
            enable_redis, enable_vector_db, enable_docker, enable_kubernetes,
            enable_ci_cd, enable_monitoring, enable_tests, enable_docs
        ])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Components", components_count)
        with col2:
            estimated_files = 15 + (components_count * 5)
            st.metric("Est. Files", f"{estimated_files}+")
        with col3:
            estimated_lines = 1000 + (components_count * 300)
            st.metric("Est. Lines", f"{estimated_lines:,}+")
        with col4:
            estimated_time = 1 + (components_count // 3)
            st.metric("Est. Time", f"{estimated_time}-{estimated_time+1}min")
        
        # Preview what will be generated
        with st.expander("📁 Preview Project Structure"):
            st.markdown("```")
            st.markdown(f"{project_name}/")
            if enable_model:
                st.markdown("├── src/")
                st.markdown("│   ├── models/        # ML model architecture")
                st.markdown("│   ├── training/      # Training scripts")
                st.markdown("│   └── data/          # Data loaders")
            if enable_fastapi:
                st.markdown("├── api/")
                st.markdown("│   ├── main.py        # FastAPI app")
                st.markdown("│   ├── routes/        # API endpoints")
                st.markdown("│   └── schemas/       # Pydantic models")
            if enable_streamlit:
                st.markdown("├── frontend/")
                st.markdown("│   ├── app.py         # Streamlit dashboard")
                st.markdown("│   ├── pages/         # Multi-page app")
                st.markdown("│   └── components/    # UI components")
            if enable_docker:
                st.markdown("├── deployment/")
                st.markdown("│   ├── docker-compose.yml")
                st.markdown("│   ├── Dockerfile")
                if enable_kubernetes:
                    st.markdown("│   └── k8s/           # Kubernetes manifests")
            pass
    if "clarification_questions" not in st.session_state:
        st.session_state.clarification_questions = None
    if "clarification_answers" not in st.session_state:
        st.session_state.clarification_answers = {}
    if "project_specification" not in st.session_state:
        st.session_state.project_specification = None

    
    # Ensure a project_id exists in the session state for uploads.
    if "project_id" not in st.session_state:
        st.session_state.project_id = f"temp_{uuid.uuid4()}"
    project_id = st.session_state.project_id
 
    
    # Handle generation - Generate questions ONCE when button clicked
    generate_button = st.button("🚀 Generate Project", type="primary", use_container_width=True, key="generate_project")
    if generate_button and project_description:
        st.markdown("---")
        st.markdown("## 🤔 Clarification & Generation")
        
        # Collect all configuration
        config = {
            "project_name": project_name,
            "description": project_description,
            "category": project_category,
            "project_id": f"{project_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "components": {
                "model": enable_model,
                "fastapi": enable_fastapi,
                "streamlit": enable_streamlit,
                "database": enable_database,
                "redis": enable_redis,
                "vector_db": enable_vector_db,
                "docker": enable_docker,
                "kubernetes": enable_kubernetes,
                "ci_cd": enable_ci_cd,
                "monitoring": enable_monitoring,
                "tests": enable_tests,
                "docs": enable_docs,
                "gpu": enable_gpu,
                "api_auth": enable_api_auth,
                "logging": enable_logging,
                "data_pipeline": enable_data_pipeline
            },
            "database_type": database_type,
            "ml_framework": ml_framework,
            "llm_provider": llm_provider,
            "vector_store": vector_store,
            "base_model": base_model,
            "finetuning_method": finetuning_method,
            "cloud_provider": cloud_provider
        }
        
        try:
            # Step 1: Clarification Phase
            st.markdown("### Step 1: Clarification Questions")
            
            # Show cute thinking animation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                show_lottie_animation(LOTTIE_ANIMATIONS["thinking"], height=150, key="thinking_anim")
            
            with st.spinner("🤔 AI is analyzing your project and preparing intelligent questions..."):
                llm_client = GeminiClient()
                prompt_manager = PromptManager()    
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # --- NEW RAG FEATURE LOGIC ---
                    if st.session_state.uploaded_files:
                        st.info(f"Analyzing {len(st.session_state.uploaded_files)} uploaded documents for context...")
                        doc_agent = DocumentAnalyzerAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                        temp_dir = Path("temp_data") / project_id
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        paper_summaries = []
                        for uploaded_file in st.session_state.uploaded_files:
                            temp_file_path = temp_dir / uploaded_file.name
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            analysis_result = loop.run_until_complete(
                                doc_agent.analyze_document(str(temp_file_path))
                            )
                            if analysis_result.get("status") == "success":
                                paper_summaries.append(f"--- START CONTEXT: {uploaded_file.name} ---\n{analysis_result.get('summary', 'No summary available.')}\n--- END CONTEXT ---")
                        
                        # Prepend summaries to the project description
                        if paper_summaries:
                            context_header = "\n\n--- EXTERNAL CONTEXT FROM UPLOADED FILES ---\n"
                            config["description"] = context_header + "\n\n".join(paper_summaries) + "\n\n--- ORIGINAL PROJECT DESCRIPTION ---\n" + project_description
                            st.success(f"✅ Added context from {len(paper_summaries)} documents.")
                    # --- END RAG FEATURE LOGIC ---

                    clarification_agent = ClarificationAgent()
                    # Prepare enhanced specification for smart question generation
                    specification = {
                        "project_name": project_name,
                        "description": config["description"],
                        "category": project_category,
                        "selected_components": {
                            "model": enable_model,
                            "fastapi": enable_fastapi,
                            "streamlit": enable_streamlit,
                            "database": enable_database,
                            "redis": enable_redis,
                            "vector_db": enable_vector_db,
                            "docker": enable_docker,
                            "kubernetes": enable_kubernetes,
                            "ci_cd": enable_ci_cd,
                            "monitoring": enable_monitoring,
                            "tests": enable_tests,
                            "docs": enable_docs,
                            "gpu": enable_gpu,
                            "api_auth": enable_api_auth,
                            "logging": enable_logging,
                            "data_pipeline": enable_data_pipeline
                        },
                        "config": {
                            "database_type": config.get("database_type", "postgresql"),
                            "ml_framework": config.get("ml_framework", "auto-detect"),
                            "llm_provider": config.get("llm_provider", "openai"),
                            "vector_store": config.get("vector_store", "chromadb"),
                            "cloud_provider": config.get("cloud_provider", "none")
                        }
                    }
                    
                    # Detect ambiguities based on description and components
                    ambiguities = []
                    
                    # Check for missing critical information
                    if "image" in project_description.lower() or "vision" in project_description.lower():
                        if "dataset" not in project_description.lower():
                            ambiguities.append({
                                "category": "data",
                                "description": "Dataset not specified",
                                "impact": "high",
                                "suggestions": ["CIFAR-10", "ImageNet", "COCO", "Custom dataset"]
                            })
                        if "model" not in project_description.lower() and "architecture" not in project_description.lower():
                            ambiguities.append({
                                "category": "model",
                                "description": "Model architecture not specified",
                                "impact": "high",
                                "suggestions": ["ResNet", "VGG", "EfficientNet", "YOLO", "Let agent decide"]
                            })
                    
                    if "rag" in project_description.lower() or "retrieval" in project_description.lower():
                        if "document" not in project_description.lower() and "data" not in project_description.lower():
                            ambiguities.append({
                                "category": "data",
                                "description": "Document source not specified",
                                "impact": "high",
                                "suggestions": ["PDF files", "Web pages", "Database", "API", "Custom"]
                            })
                        if "chunk" not in project_description.lower():
                            ambiguities.append({
                                "category": "configuration",
                                "description": "Chunking strategy not specified",
                                "impact": "medium",
                                "suggestions": ["Fixed size (512 tokens)", "Semantic chunking", "Paragraph-based","RecursiveCharacterTextSplitter", "Let agent decide"]
                            })
                    
                    if "api" in project_description.lower() or enable_fastapi:
                        if "endpoint" not in project_description.lower():
                            ambiguities.append({
                                "category": "api",
                                "description": "API endpoints not specified",
                                "impact": "medium",
                                "suggestions": ["Auto-generate CRUD", "Prediction only", "Custom endpoints", "Let agent decide"]
                            })
                    
                    if enable_database and not database_type:
                        ambiguities.append({
                            "category": "database",
                            "description": "Database type preference",
                            "impact": "medium",
                            "suggestions": ["PostgreSQL", "MongoDB", "MySQL", "SQLite"]
                        })
                    
                    # Generate smart, context-aware questions using LLM
                    st.info("🧠 Analyzing your project requirements...")
                    questions = loop.run_until_complete(
                        clarification_agent.generate_smart_questions(specification, ambiguities)
                    )
                    
                    # Store questions in session state so they persist across reruns
                    st.session_state.clarification_questions = questions
                    st.session_state.project_specification = specification
                    
                    if questions and len(questions) > 0:
                        st.success(f"✅ Generated {len(questions)} intelligent questions based on your project!")
                        st.info("💡 **Please answer these questions to help us build exactly what you want:**")
                        
                        # Store answers in session state
                        if "clarification_answers" not in st.session_state:
                            st.session_state.clarification_answers = {}
                        
                        # Display smart questions (these are dict objects from LLM)
                        for i, question in enumerate(questions):
                            # Smart questions are dictionaries with keys: question, options, priority, rationale
                            question_text = question.get("question", "")
                            question_id = question.get("id", f"q_{i}")
                            priority = question.get("priority", "medium")
                            rationale = question.get("rationale", "")
                            
                            # Show priority badge
                            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                            st.markdown(f"{priority_emoji} **Q{i+1}:** {question_text}")
                            
                            # Show rationale if available
                            if rationale:
                                with st.expander("ℹ️ Why are we asking this?"):
                                    st.write(rationale)
                                    if question.get("impact"):
                                        st.write(f"**Impact:** {question['impact']}")
                            
                            # Get question options
                            options = question.get("options", [])
                            default_val = question.get("default")
                            
                            if options and len(options) > 0:
                                # Multiple choice question
                                default_index = 0
                                if default_val and default_val in options:
                                    default_index = options.index(default_val)
                                
                                answer = st.radio(
                                    f"Select your choice:",
                                    options,
                                    index=default_index,
                                    key=f"smart_q_{i}_{question_id}"
                                )
                                st.session_state.clarification_answers[question_id] = answer
                            
                            else:
                                # Text input question
                                placeholder = default_val if default_val else "Enter your answer..."
                                answer = st.text_input(
                                    f"Your answer:",
                                    key=f"smart_q_{i}_{question_id}",
                                    placeholder=placeholder
                                )
                                st.session_state.clarification_answers[question_id] = answer
                            
                            st.markdown("---")
                        
                        # Button to proceed with generation
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.button("✅ Proceed with Generation", type="primary", use_container_width=True):
                                # Update config with answers
                                project_id = f"{project_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                config = {
                                        "project_name": project_name,
                                        "description": project_description,
                                        "category": project_category,
                                        "project_id": project_id, # <-- THIS LINE WAS MISSING
                                        "components": {
                                            "model": enable_model, "fastapi": enable_fastapi, "streamlit": enable_streamlit,
                                            "database": enable_database, "redis": enable_redis, "vector_db": enable_vector_db,
                                            "docker": enable_docker, "kubernetes": enable_kubernetes, "ci_cd": enable_ci_cd,
                                            "monitoring": enable_monitoring, "tests": enable_tests, "docs": enable_docs,
                                            "gpu": enable_gpu, "api_auth": enable_api_auth, "logging": enable_logging,
                                            "data_pipeline": enable_data_pipeline
                                        },
                                    "database_type": database_type, "ml_framework": ml_framework,
                                    "llm_provider": llm_provider, "vector_store": vector_store,
                                    "base_model": base_model, "finetuning_method": finetuning_method,
                                    "cloud_provider": cloud_provider,
                                    "clarifications": st.session_state.clarification_answers
                                }
                                
                                # Proceed to generation
                                st.success("✅ Clarifications received! Generating your project...")
                                st.session_state.clarification_questions = None
                                generate_complete_project(config, loop)
                                st.rerun()
                            
                        with col2:
                            if st.button("🔄 Regenerate Questions", use_container_width=True):
                                st.session_state.clarification_questions = None
                                st.session_state.clarification_answers = {}
                                st.rerun()
                    
                    else:
                        # No clarification needed, proceed directly
                        st.success("✅ No clarification needed! Your description is clear. Generating...")
                        generate_complete_project(config, loop)
                
                finally:
                    loop.close()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    
    # Display questions from session state (persists across reruns)
   
    elif st.session_state.clarification_questions is not None:
        st.markdown("---")
        st.markdown("## 🤔 Clarification & Generation")
        
        questions = st.session_state.clarification_questions
        
        if questions and len(questions) > 0:
            st.success(f"✅ Generated {len(questions)} intelligent questions based on your project!")
            st.info("💡 **Please answer these questions to help us build exactly what you want:**")
            
            # Display smart questions (these are dict objects from LLM)
            for i, question in enumerate(questions):
                # Smart questions are dictionaries with keys: question, options, priority, rationale
                question_text = question.get("question", "")
                question_id = question.get("id", f"q_{i}")
                priority = question.get("priority", "medium")
                rationale = question.get("rationale", "")
                
                # Show priority badge
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "critical": "🔴"}.get(priority, "⚪")
                st.markdown(f"{priority_emoji} **Q{i+1}:** {question_text}")
                
                # Show rationale if available
                if rationale:
                    with st.expander("ℹ️ Why are we asking this?"):
                        st.write(rationale)
                        if question.get("impact"):
                            st.write(f"**Impact:** {question['impact']}")
                
                # Get question options
                options = question.get("options", [])
                default_val = question.get("default")
                
                if options and len(options) > 0:
                    # Multiple choice question

                    default_index = 0
                    if default_val and default_val in options:
                        default_index = options.index(default_val)
                    
                    answer = st.radio(
                        f"Select your choice:",
                        options,
                        index=default_index,
                        key=f"smart_q_{i}_{question_id}"
                    )
                    
                    # If user selects "Custom", show text input
                    if "custom" in answer.lower() and "specify" in answer.lower():
                        custom_prompt = question.get("custom_prompt", "Please specify your custom choice:")

                        custom_answer = st.text_input(
                            custom_prompt,
                            key=f"custom_input_{i}_{question_id}",
                            placeholder="Enter your custom value..."
                        )
                        if custom_answer:
                            st.session_state.clarification_answers[question_id] = f"Custom: {custom_answer}"
                        else:
                            st.warning("⚠️ Please provide your custom specification above")
                            st.session_state.clarification_answers[question_id] = answer
                    else:
                        st.session_state.clarification_answers[question_id] = answer
                
                else:
                    # Text input question
                    placeholder = default_val if default_val else "Enter your answer..."
                    answer = st.text_input(
                        f"Your answer:",
                        key=f"smart_q_{i}_{question_id}",
                        placeholder=placeholder
                    )
                    st.session_state.clarification_answers[question_id] = answer
                
                st.markdown("---")
            
            # Button to proceed with generation
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("✅ Proceed with Generation", type="primary", use_container_width=True, key="proceed_generation"):
                    
                    # Create a unique project ID and add it to the config
                    project_id = f"{project_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    config = {
                        "project_name": project_name,
                        "description": project_description,
                        "category": project_category,
                        "project_id": project_id, 
                        "components": {
                            "model": enable_model, "fastapi": enable_fastapi, "streamlit": enable_streamlit,
                            "database": enable_database, "redis": enable_redis, "vector_db": enable_vector_db,
                            "docker": enable_docker, "kubernetes": enable_kubernetes, "ci_cd": enable_ci_cd,
                            "monitoring": enable_monitoring, "tests": enable_tests, "docs": enable_docs,
                            "gpu": enable_gpu, "api_auth": enable_api_auth, "logging": enable_logging,
                            "data_pipeline": enable_data_pipeline
                        },
                        "database_type": database_type, "ml_framework": ml_framework,
                        "llm_provider": llm_provider, "vector_store": vector_store,
                        "base_model": base_model, "finetuning_method": finetuning_method,
                        "cloud_provider": cloud_provider,
                        "clarifications": st.session_state.clarification_answers
                    }
                    
                    if st.session_state.uploaded_files:
                        temp_dir = Path("temp_data") / project_id
                        if temp_dir.exists(): # Check if temp dir exists from previous run
                            # This is a simplified re-application. A more robust way would be to store summaries in session_state.
                            logger.info("Re-applying document context for generation...")
                            config["description"] = f"Context from {len(st.session_state.uploaded_files)} files was analyzed.\n\n" + project_description


                    # Proceed to generation
                    st.success("✅ Clarifications received! Generating your project...")
                    st.session_state.clarification_questions = None # Clear questions
                    
                    generate_complete_project(config)
                    st.rerun()
            with col2:
                if st.button("🔄 Regenerate Questions", use_container_width=True, key="regen_questions"):
                    st.session_state.clarification_questions = None
                    st.session_state.clarification_answers = {}
                    st.rerun()



def generate_complete_project(config, loop=None):
    """
    Generate complete AI/ML project using the advanced TaskTracker and Iterative Refinement workflow.
    This orchestrates all agents to create a production-ready project.
    
    Workflow:
    1. PromptEngineerAgent -> DynamicArchitectAgent -> TaskTracker
    2. SharedMemory -> Store project context
    3. Loop (Code Generation):
       - CodingAgent (task="generate_from_spec")
    4. Post-Generation Loop (run once all code is generated):
       - SetupAutomationTool (Create venv, install deps, git init)
       - TaskRunner (Run tests)
       - DebuggingAgent (if tests fail)
       - ReviewOrchestrator (run all 6 reviews)
       - AdvancedDocumentationAgent
       - ComplexInterviewAgent
       - DeploymentAgent
    5. Save project, create ZIP, and initialize chat.
    """
    st.markdown("---")
    st.markdown("### 🔄 Generating Your Complete Project")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st_lottie:
            show_lottie_animation(LOTTIE_ANIMATIONS["loading"], height=200, key="loading_gen")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    close_loop = False
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        close_loop = True
    
    # --- Constants for Iterative Loop ---
    MAX_ITERATIONS = 3
    TARGET_SCORE = 75.0
    
    try:
        all_files = {} # This will hold the generated code for review/zipping
        project_name = config["project_name"]
        project_id = config["project_id"]
        output_dir = Path("output") / project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting generation for project: {project_name} ({project_id}) at {output_dir}")

        # --- Initialize Core Components ---
        llm_client = GeminiClient()
        prompt_manager = PromptManager()
        shared_memory = SharedMemory()
        
        if "project_type" not in config:
            config["project_type"] = config.get("category", "custom_project")
        if "requirements" not in config:
            config["requirements"] = config.get("description", "")

        # --- Phase 0: Prompt Engineering & Planning ---
        st.markdown("#### 🧠 Phase 0: Intelligent Analysis & Planning")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st_lottie:
                show_lottie_animation(LOTTIE_ANIMATIONS["ai_brain"], height=150, key="brain_anim")

        status_text.text("🔍 Step 0.1/0.3: Finalizing specification...")
        progress_bar.progress(2)
        with st.spinner("Finalizing project specification..."):
            if st.session_state.project_specification:
                enhanced_spec = st.session_state.project_specification
                enhanced_spec["clarifications"] = config.get("clarifications", {})
                logger.info("Loaded specification from session state.")
            else:
                prompt_engineer = PromptEngineerAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                enhanced_description = config["description"]
                context = { "project_type": config["category"], "requirements": config["description"], **config["components"] }
                enhanced_spec = loop.run_until_complete(
                    prompt_engineer.enhance_user_input(enhanced_description, context)
                )
                logger.warning("No specification in session state, running enhancement again.")
            st.success("✅ Specification finalized!")

        status_text.text("🏗️ Step 0.2/0.3: Designing dynamic system architecture...")
        progress_bar.progress(5)
        with st.spinner("Planning dynamic architecture..."):
            architect_agent = DynamicArchitectAgent(
                llm_client=llm_client, 
                prompt_manager=prompt_manager
            )
            
            architecture_obj = loop.run_until_complete(
                architect_agent.execute_task({
                    "task_type": "design_architecture",
                    "data": {
                        "enhanced_spec": enhanced_spec,
                        "components": config["components"],
                        "clarifications": config.get("clarifications", {})
                    }
                })
            )
            
        
            # Check if the returned object is valid and has files.
            if not isinstance(architecture_obj, ProjectArchitecture) or not architecture_obj.file_specs:
                 st.error(f"Failed to generate a valid architecture plan. Received: {architecture_obj}")
                 raise ValueError(f"Architecture generation failed. Object: {architecture_obj}")
            

            st.success(f"✅ Dynamic architecture designed! ({len(architecture_obj.build_order)} files planned)")
            with st.expander("📋 View Architecture Plan"):
                 st.json(architecture_obj.to_dict())

        status_text.text("📝 Step 0.3/0.3: Creating persistent task list (TODO.json)...")
        progress_bar.progress(8)
        with st.spinner("Creating task breakdown..."):
            tracker = create_tasks_from_architecture(architecture_obj, output_dir)
            tracker.save()
            st.success(f"✅ Tasks identified: {len(tracker.tasks)} components (saved to TODO.json)")
            with st.expander("🔍 View Implementation Tasks"):
                st.json([t.to_dict() for t in tracker.tasks])
        
        progress_bar.progress(10)
        
        # --- Store critical info in shared memory ---
        shared_memory.store("import_map", architecture_obj.import_map, persistent=True)
        shared_memory.store("full_architecture", architecture_obj.to_dict(), persistent=True)
        shared_memory.store("project_context", enhanced_spec, persistent=True)
        
        # --- Initialize all agents once ---
        coding_agent = CodingAgent(llm_client=llm_client, prompt_manager=prompt_manager)
        task_runner = TaskRunner(working_dir=output_dir)
        testing_agent = TestingAgent(llm_client=llm_client, prompt_manager=prompt_manager, task_runner=task_runner)
        advanced_doc_agent = AdvancedDocumentationAgent(llm_client=llm_client, prompt_manager=prompt_manager)
        readability_reviewer = ReadabilityReviewer()
        logic_flow_reviewer = LogicFlowReviewer()
        code_connectivity_reviewer = CodeConnectivityReviewer()
        project_connectivity_reviewer = ProjectConnectivityReviewer()
        performance_reviewer = PerformanceReviewer()
        security_reviewer = SecurityReviewer()
        review_orchestrator = ReviewOrchestrator(
            readability_reviewer, logic_flow_reviewer, code_connectivity_reviewer,
            project_connectivity_reviewer, performance_reviewer, security_reviewer
        )
        interview_agent = ComplexInterviewAgent(llm_client=llm_client, prompt_manager=prompt_manager)
        deployment_agent = DeploymentAgent(llm_client=llm_client, prompt_manager=prompt_manager)
        modification_agent = InteractiveModificationAgent(llm_client=llm_client, prompt_manager=prompt_manager, project_root=str(output_dir))
        debugging_agent = DebuggingAgent(
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            project_root=str(output_dir),
            task_runner=task_runner,
            modification_agent=modification_agent
        )
        
        
        # --- Main Iterative Loop ---
        feedback_reports = {}
        review_report = {} # Initialize review_report
        parsed_test_output = None # Initialize parsed_test_output

        for iteration in range(1, MAX_ITERATIONS + 1):
            st.markdown("---")
            st.markdown(f"#### 🔁 Iteration {iteration} / {MAX_ITERATIONS}")
            
            # --- Phase 1: Code Generation ---
            if iteration == 1:
                st.markdown("##### 🏗️ Phase 1: Initial Code Generation")
                file_log_placeholder = st.empty()
                file_log_content = ""
            else:
                st.markdown("##### 🏗️ Phase 1: Code Refinement")
                file_log_placeholder = st.empty()
                file_log_content = f"**Applying feedback from Iteration {iteration-1}...**\n"
                file_log_placeholder.markdown(file_log_content)
            
            total_files = len(architecture_obj.build_order)
            generated_files_count = 0
            
            tracker.reset_task_status() # Reset all tasks to PENDING
            
            while (task_to_run := tracker.get_next_task()):
                if task_to_run.type != TaskType.CODE_GENERATION:
                    tracker.mark_complete(task_to_run.id, {"status": "skipped"})
                    continue
                    
                generated_files_count += 1
                file_path = task_to_run.target
                status_text.text(f"✍️ [Iter {iteration}] Generating file {generated_files_count}/{total_files}: `{file_path}`...")
                progress = (10 + int(60 * generated_files_count / total_files)) # Code gen is 60% of progress
                progress_bar.progress(progress)
                
                with st.spinner(f"Agent is writing `{file_path}` (Attempt {iteration})..."):
                    tracker.mark_started(task_to_run.id)
                    
                    file_spec = architecture_obj.get_file_spec(file_path)
                    if not file_spec:
                        error_msg = f"Could not find file_spec for {file_path} in architecture plan."
                        logger.error(f"FATAL: {error_msg}")
                        tracker.mark_failed(task_to_run.id, error_msg)
                        continue
                    
                    coding_task_data = {
                        "task_type": "generate_from_spec",
                        "data": {
                            "file_spec": file_spec.to_dict(), 
                            "import_map": shared_memory.retrieve("import_map"), 
                            "project_context": shared_memory.retrieve("project_context")
                        },
                        "feedback": feedback_reports # Pass feedback from previous loop
                    }
                    
                    result = loop.run_until_complete(coding_agent.execute_task(coding_task_data))
                    
                    if result.get("status") == "success":
                        code = result.get("code", "")
                        clean_code = strip_markdown_code_fences(code)
                        all_files[file_path] = clean_code
                        
                        full_file_path = output_dir / file_path
                        full_file_path.parent.mkdir(parents=True, exist_ok=True)
                        full_file_path.write_text(clean_code, encoding='utf-8')
                        
                        if iteration == 1:
                            file_log_content += f"- 📄 Generated: `{file_path}` ({len(clean_code)} chars)\n"
                            file_log_placeholder.markdown(file_log_content)
                        
                        tracker.mark_complete(task_to_run.id, {"file_size": len(clean_code)})
                    else:
                        error_message = result.get("message", "Unknown error")
                        logger.error(f"Failed to generate file `{file_path}`: {error_message}")
                        all_files[file_path] = f"# ERROR: Failed to generate file.\n# Reason: {error_message}"
                        tracker.mark_failed(task_to_run.id, error_message)

            st.success(f"✅ Iteration {iteration}: Code generation/refinement complete!")
            progress_bar.progress(70)

            # --- Phase 2: Setup, Test, & Review ---
            st.markdown("##### 🛠️ Phase 2: Setup, Test, & Review")
            
            # Step 2.1: Setup Project (Venv, Install, Git)
            status_text.text(f"⚙️ [Iter {iteration}] Setting up project environment...")
            progress_bar.progress(75)
            setup_result = None
            with st.spinner(f"Running SetupAutomationTool (conda/venv, install, git)... Iteration {iteration}"):
                dependencies = list(architecture_obj.dependencies)
                dev_dependencies = list(set(["pytest", "pytest-cov", "black", "flake8", "mypy"]))
                
                setup_config = SetupConfig(
                    project_name=project_name,
                    project_path=str(output_dir),
                    project_type=ProjectType.ML,
                    env_type=EnvironmentType.CONDA,
                    venv_name=f"{project_name}_env",
                    dependencies=dependencies,
                    dev_dependencies=dev_dependencies,
                    init_git=(iteration == 1),
                    initial_commit=(iteration == 1),
                    config_files=[ConfigFileType.PYPROJECT_TOML, ConfigFileType.GITIGNORE, ConfigFileType.README],
                    skip_structure=True 
                )
                setup_tool = SetupAutomationTool(config=setup_config)
                
                setup_result = setup_tool.setup_project()
                
                if not setup_result.success:
                    st.warning(f"Project setup warnings: {setup_result.message}")
                    with st.expander("View Setup Errors"):
                        st.json(setup_result.to_dict())
                else:
                    st.success("✅ Environment created & dependencies installed.")
            
            # Step 2.2: Generate Tests (Only on first iteration)
            if iteration == 1 and config["components"]["tests"]:
                status_text.text("🧪 Generating Unit Tests...")
                progress_bar.progress(80)
                with st.spinner("TestingAgent is generating unit tests..."):
                    test_files_result = loop.run_until_complete(testing_agent.execute_task({
                        "task_type": "generate_project_tests",
                        "data": {
                            "project_files": all_files,
                            "architecture": architecture_obj.to_dict()
                        }
                    }))
                    
                    if test_files_result.get("status") == "success" and "test_files" in test_files_result:
                        test_files = test_files_result["test_files"]
                        for filepath, content in test_files.items():
                            clean_content = strip_markdown_code_fences(content)
                            all_files[filepath] = clean_content
                            (output_dir / filepath).parent.mkdir(parents=True, exist_ok=True)
                            (output_dir / filepath).write_text(clean_content, encoding='utf-8')
                            file_log_content += f"- 🧪 Generated: `{filepath}` ({len(clean_content)} chars)\n"
                            file_log_placeholder.markdown(file_log_content)
                        st.success(f"✅ TestingAgent: {len(test_files)} test files generated.")
                    else:
                        st.warning(f"⚠️ TestingAgent failed to generate tests: {test_files_result.get('message')}")
            
            # Step 2.3: Run Live Tests
            parsed_test_output = None
            if config["components"]["tests"]:
                status_text.text(f"🏃 [Iter {iteration}] Running live tests...")
                progress_bar.progress(85)
                with st.spinner(f"TaskRunner is executing pytest (Iteration {iteration})..."):
                    if setup_result.env_result and setup_result.env_result.env_info:
                        python_exe = setup_result.env_result.env_info.python_executable
                        test_command = f'"{python_exe}" -m pytest tests/'
                        
                        test_run_result = task_runner.execute_command(command=test_command, timeout=300)
                        
                        test_failure_handler = TestFailureHandler()
                        parsed_test_output = test_failure_handler.parse_pytest_output(test_run_result.output)
                        
                        if not test_run_result.success or parsed_test_output.has_failures:
                            st.warning(f"⚠️ Iteration {iteration}: Tests failed ({parsed_test_output.failed} failures).")
                            feedback_reports["test_report"] = parsed_test_output.to_dict()
                            
                            status_text.text(f"🪛 [Iter {iteration}] Debugging test failures...")
                            debug_results = loop.run_until_complete(debugging_agent.execute_task({
                                "task_type": "debug_test_failures",
                                "data": {
                                    "test_results_output": test_run_result.output,
                                    "files": all_files
                                }
                            }))
                            st.info(f"Debugging completed: {debug_results.get('message', 'See logs.')}")
                            with st.expander("View Debugging Analysis & Fixes"):
                                st.json(debug_results)
                            
                        else:
                            st.success(f"✅ Iteration {iteration}: All {parsed_test_output.passed} tests passed!")
                            feedback_reports["test_report"] = None
                    else:
                        st.error("Could not find venv python executable. Skipping tests.")
            
            # Step 2.4: Run Review
            status_text.text(f"🔍 [Iter {iteration}] Orchestrating Code Reviews...")
            progress_bar.progress(90)
            with st.spinner(f"ReviewOrchestrator is analyzing files (Iteration {iteration})..."):
                review_report = review_orchestrator.review_all(all_files, project_context=enhanced_spec)
                review_content = json.dumps(review_report, indent=2, default=str)
                (output_dir / "reviews").mkdir(parents=True, exist_ok=True)
                (output_dir / f"review_report_iter_{iteration}.json").write_text(review_content, encoding='utf-8')
                
                overall_score = review_report.get('overall_score', 0)
                st.success(f"✅ Review complete. Overall Score: {overall_score:.1f}")
                feedback_reports["review_report"] = review_report
            
            # --- Phase 3: Decision ---
            st.markdown("##### 🏁 Phase 3: Iteration Decision")
            progress_bar.progress(95)
            
            if overall_score >= TARGET_SCORE and (parsed_test_output is None or not parsed_test_output.has_failures):
                st.success(f"🎉 **Quality target met!** (Score: {overall_score:.1f} >= {TARGET_SCORE})")
                logger.info(f"Iteration {iteration} successful. Score {overall_score} >= {TARGET_SCORE}.")
                break
            elif iteration < MAX_ITERATIONS:
                st.warning(f"⚠️ Quality target not met (Score: {overall_score:.1f} < {TARGET_SCORE}). Starting next iteration...")
                logger.warning(f"Iteration {iteration} failed. Score {overall_score} < {TARGET_SCORE}. Retrying...")
                time.sleep(2)
            else:
                st.error(f"🚫 **Max iterations ({MAX_ITERATIONS}) reached.** Stopping with final score: {overall_score:.1f}")
                logger.error(f"Max iterations reached. Final score: {overall_score}")
                break
        
        # --- Phase 4: Finalization (outside the loop) ---
        st.markdown("---")
        st.markdown("#### ✨ Phase 4: Finalizing Project")
        
        # Run DeploymentAgent, Documentation, Interview agents ONCE
        if config["components"]["docker"] or config["components"]["kubernetes"]:
            status_text.text("🐳 Step 4.1: Generating Deployment Configuration...")
            progress_bar.progress(96)
            with st.spinner("Creating deployment files..."):
                deployment_files = loop.run_until_complete(
                    deployment_agent.execute_task({
                        "task_type": "generate_deployment_config",
                        "data": {
                            "project_name": project_name,
                            "components": config["components"],
                            "config": {
                                "kubernetes": config["components"].get("kubernetes", False),
                                "ci_cd": config["components"].get("ci_cd", False),
                                "ci_platform": "github",
                                "monitoring": config["components"].get("monitoring", False),
                                "gpu_support": config["components"].get("gpu", False),
                                "database_type": config.get("database_type", "postgresql")
                            }
                        }
                    })
                )
                for filename, content in deployment_files.items():
                    clean_content = strip_markdown_code_fences(content)
                    all_files[filename] = clean_content 
                    (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
                    (output_dir / filename).write_text(clean_content, encoding='utf-8')
                    file_log_content += f"- 🐳 Generated: `{filename}` ({len(clean_content)} chars)\n"
                    file_log_placeholder.markdown(file_log_content)
                st.success(f"✅ Generated {len(deployment_files)} deployment files")

        if config["components"]["docs"]:
            status_text.text("📝 Step 4.2: Generating Project Documentation...")
            progress_bar.progress(97)
            with st.spinner("DocumentationAgent is writing docs..."):
                file_structure = list(all_files.keys())
                doc_suite = loop.run_until_complete(
                    advanced_doc_agent.generate_documentation_suite(config, file_structure)
                )
                for filename, content in doc_suite.items():
                    all_files[filename] = content
                    (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
                    (output_dir / filename).write_text(content, encoding='utf-8')
                file_log_content += f"- 📝 Generated {len(doc_suite)} documentation files.\n"
                file_log_placeholder.markdown(file_log_content)
                st.success(f"✅ DocumentationAgent: {len(doc_suite)} files generated.")

        status_text.text("🎤 Step 4.3: Generating Interview Questions...")
        progress_bar.progress(98)
        with st.spinner("InterviewAgent is preparing questions..."):
            interview_doc_list = loop.run_until_complete(
                interview_agent.generate_interview_questions(
                    role=config.get("category", "AI/ML Engineer"),
                    topic=config.get("description", "Project Overview"),
                    difficulty="Medium"
                )
            )
            interview_content = f"# Interview Prep for {project_name}\n\n"
            for q in interview_doc_list:
                interview_content += f"## ❓ {q.get('question', 'N/A')}\n\n"
                interview_content += "**Key Points to Cover:**\n" + "".join(f"- {p}\n" for p in q.get('answer_key', []))
                interview_content += "\n**Follow-up Questions:**\n" + "".join(f"- {f}\n" for f in q.get('follow_ups', []))
                interview_content += "\n---\n"
            
            all_files["docs/interview_questions.md"] = interview_content
            (output_dir / "docs").mkdir(parents=True, exist_ok=True)
            (output_dir / "docs/interview_questions.md").write_text(interview_content, encoding='utf-8')
            file_log_content += f"- 🎤 Generated: `docs/interview_questions.md`\n"
            file_log_placeholder.markdown(file_log_content)
            st.success("✅ InterviewAgent: Questions generated.")

        # Complete!
        progress_bar.progress(100)
        status_text.text("✅ Project generation, setup, and validation complete!")
        
        if st_lottie:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                show_lottie_animation(LOTTIE_ANIMATIONS["party"], height=200, key="success_anim")
        
        st.success(f"🎉 **Successfully generated {project_name}!**")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(all_files))
        with col2:
            total_lines = sum(content.count('\n') for content in all_files.values())
            st.metric("Total Lines", f"{total_lines:,}")
        with col3:
            st.metric("Final Score", f"{review_report.get('overall_score', 0):.1f}")
        with col4:
            if config["components"]["tests"] and parsed_test_output:
                st.metric("Tests Passed", f"{parsed_test_output.passed}/{parsed_test_output.total_tests}")
            else:
                 st.metric("Tests", "N/A")
        
        st.info(f"📁 Project saved to: `{output_dir}`")

        # -Add expander for detailed review report ---
        with st.expander("🔍 View Detailed Code Review Report"):
            st.json(review_report)
        
        
        # Create ZIP
        zip_path = output_dir.parent / f"{project_id}.zip"
        with st.spinner("Creating project ZIP file..."):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in output_dir.rglob("*"):
                    # Exclude venv and other noise
                    if file.is_file() and not any(part in file.parts for part in ['venv', '.venv', '__pycache__', '.pytest_cache']):
                        zipf.write(file, file.relative_to(output_dir))
        
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="📥 Download Complete Project (ZIP)",
                data=f.read(),
                file_name=f"{project_name}.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
        
        with st.expander("🚀 Quick Start Guide"):
            st.markdown(f"""
            ### Deploy Your Project
            
            1. **Extract the ZIP file and navigate into it**
               ```bash
               unzip {project_name}.zip
               cd {output_dir.name}
               ```
            
            2. **Activate the virtual environment**
               ```bash
               # If you used Conda (default)
               conda activate ./{f"{project_name}_env"}
               
               # If you used venv
               # On Linux/macOS
               source venv/bin/activate
               # On Windows
               .\\venv\\Scripts\\activate
               ```

  
            3. **Run your tests**
               ```bash
               pytest tests/
               ```
            """)
            if config["components"]["docker"]:
                st.markdown(f"""
                ---
                ### Or use Docker
                (See `deployment/README.md` for details)
                ```bash
                docker-compose -f deployment/docker-compose.yml up --build
                ```
                """)


        st.session_state.generated_projects[project_id] = {
            "name": project_name,
            "config": config,
            "output_dir": output_dir,
            "zip_path": zip_path,
            "generated_at": datetime.now().isoformat(),
            "files_count": len(all_files),
            "review_score": review_report.get('overall_score', 0)

        }
        st.session_state.current_project_id = project_id
        # --- Post-Generation Chat ---
        st.markdown("### 💬 Chat With Your New Project")
        st.info("Your project is complete. You can now ask questions, request changes, or report bugs below.")

        project_chat_id = project_id
        if project_chat_id not in st.session_state.project_chats:
            chat_save_path = output_dir / "chat_history"
            conv_mgr = ConversationManager(save_path=chat_save_path)
            conv_mgr.start_conversation(
                conversation_id=project_chat_id,
                initial_query=f"Let's discuss the project: {project_name}. {config['description']}",
                context=config
            )
            st.session_state.project_chats[project_chat_id] = conv_mgr
        
        conv_mgr = st.session_state.project_chats[project_chat_id]
        conversation = conv_mgr.active_conversation

        with st.container(height=400, border=True):
            if conversation and conversation.messages:
                for msg in conversation.messages:
                    with st.chat_message(msg.role):
                        st.markdown(msg.content)

        if prompt := st.chat_input("Ask a question, request a change, or report an error...", key=f"chat_input_new_{project_chat_id}"):
            conv_mgr.add_message(conversation, role="user", content=prompt)
            
            with st.spinner("AI is analyzing your request..."):
                interaction_agent = UserInteractionAgent(
                    llm_client=llm_client, 
                    prompt_manager=prompt_manager,
                    project_path=output_dir,
                    conversation=conversation
                )
                
                ai_response = loop.run_until_complete(
                    interaction_agent.handle_user_request(prompt)
                )

            conv_mgr.add_message(conversation, role="assistant", content=ai_response)
            conv_mgr.save_conversation(conversation)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Generation failed: {str(e)}")
        st.code(traceback.format_exc())
    
    finally:
        if close_loop:
            try:
                loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")

if __name__ == "__main__":
    setup_logger(log_level="INFO")
    main()

