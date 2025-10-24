
import streamlit as st
import asyncio
from src.agents.interactive.clarification_agent import ClarificationAgent
from src.ui.utils import show_lottie_animation, LOTTIE_ANIMATIONS
from src.ui.pages.project_generator import generate_complete_project

def render_ai_ml_project_generator():
    """
    Unified AI/ML Project Generator
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
    
    project_description = st.text_area(
        "What do you want to build? (Be as detailed as possible)",
        placeholder="""Examples:
• "Build a CIFAR-10 image classifier using ResNet-18 with data augmentation, trained for 50 epochs"
• "Create a sentiment analysis model for movie reviews using BERT, with FastAPI serving and Streamlit UI"
• "Develop a RAG system for customer support using LangChain, ChromaDB, and OpenAI GPT-4"
        """,
        height=150,
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
        if project_description:
            description_lower = project_description.lower()
            if any(word in description_lower for word in ['rag', 'retrieval', 'langchain']):
                default_category = "GenAI - RAG System"
            elif any(word in description_lower for word in ['fine-tun', 'lora', 'qlora']):
                default_category = "GenAI - LLM Fine-tuning"
            elif any(word in description_lower for word in ['yolo', 'detection']):
                default_category = "Computer Vision - Object Detection"
            elif any(word in description_lower for word in ['image', 'classification', 'vision']):
                default_category = "Computer Vision - Image Classification"
            elif any(word in description_lower for word in ['sentiment', 'text class', 'nlp']):
                default_category = "NLP - Text Classification"
            else:
                default_category = "Custom Project"
        else:
            default_category = "Custom Project"
        
        project_category = st.selectbox(
            "Project Category (auto-detected):",
            [
                "Computer Vision - Image Classification",
                "Computer Vision - Object Detection",
                "NLP - Text Classification",
                "GenAI - RAG System",
                "GenAI - LLM Fine-tuning",
                "Custom Project"
            ],
            index=[
                "Computer Vision - Image Classification",
                "Computer Vision - Object Detection",
                "NLP - Text Classification",
                "GenAI - RAG System",
                "GenAI - LLM Fine-tuning",
                "Custom Project"
            ].index(default_category)
        )
    
    st.markdown("---")
    
    st.markdown("### 2️⃣ Select Components to Include")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Core Components**")
        enable_model = st.checkbox("ML/AI Model", value=True, disabled=True)
        enable_fastapi = st.checkbox("FastAPI Backend", value=True)
        enable_streamlit = st.checkbox("Streamlit UI", value=True)
    
    with col2:
        st.markdown("**💾 Data & Storage**")
        enable_database = st.checkbox("Database", value=False)
        enable_vector_db = st.checkbox("Vector Database", value="GenAI" in project_category)
    
    with col3:
        st.markdown("**🐳 Deployment**")
        enable_docker = st.checkbox("Docker Compose", value=True)
        enable_ci_cd = st.checkbox("CI/CD Pipeline", value=True)

    st.markdown("---")
    
    generate_button = st.button(
        "🚀 Start Interactive Generation",
        type="primary",
        use_container_width=True,
        disabled=not project_description or not project_name
    )
    
    if 'clarification_questions' not in st.session_state:
        st.session_state.clarification_questions = None

    if generate_button and project_description:
        st.markdown("---")
        st.markdown("## 🤔 Clarification & Generation")
        
        config = {
            "project_name": project_name,
            "description": project_description,
            "category": project_category,
            "components": {
                "model": enable_model,
                "fastapi": enable_fastapi,
                "streamlit": enable_streamlit,
                "database": enable_database,
                "vector_db": enable_vector_db,
                "docker": enable_docker,
                "ci_cd": enable_ci_cd,
            }
        }
        
        try:
            st.markdown("### Step 1: Clarification Questions")
            
            with st.spinner("🤔 AI is analyzing your project..."):
                clarification_agent = ClarificationAgent()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    questions = loop.run_until_complete(
                        clarification_agent.generate_smart_questions(config, [])
                    )
                    st.session_state.clarification_questions = questions
                    st.session_state.project_specification = config
                finally:
                    loop.close()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    elif st.session_state.clarification_questions is not None:
        st.markdown("---")
        st.markdown("## 🤔 Clarification & Generation")
        
        questions = st.session_state.clarification_questions
        
        if questions:
            st.success(f"✅ Generated {len(questions)} intelligent questions!")
            
            for i, question in enumerate(questions):
                question_text = question.get("question", "")
                question_id = question.get("id", f"q_{i}")
                options = question.get("options", [])
                
                st.markdown(f"**Q{i+1}:** {question_text}")
                
                if options:
                    answer = st.radio(
                        f"Select your choice:",
                        options,
                        key=f"smart_q_{i}_{question_id}"
                    )
                    st.session_state.clarification_answers[question_id] = answer
                else:
                    answer = st.text_input(
                        f"Your answer:",
                        key=f"smart_q_{i}_{question_id}"
                    )
                    st.session_state.clarification_answers[question_id] = answer
            
            if st.button("✅ Proceed with Generation", type="primary", use_container_width=True):
                config = st.session_state.project_specification
                config["clarifications"] = st.session_state.clarification_answers
                generate_complete_project(config)
