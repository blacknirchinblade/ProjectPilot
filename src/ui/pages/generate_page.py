
import streamlit as st
import time
import asyncio
from pathlib import Path
import traceback
from src.agents.coding.coding_agent import CodingAgent
from src.ui.pages.ai_ml_project_page import render_ai_ml_project_generator

def render_generate_page():
    """Render the code generation page."""
    st.markdown("## ✨ Generate Code")
    
    tab1, tab2 = st.tabs([
        "🤖 AI/ML Project Generator",
        "📄 Simple Code Generation"
    ])
    
    with tab1:
        render_ai_ml_project_generator()
    
    with tab2:
        render_simple_generation()

def render_simple_generation():
    """Render simple code generation interface."""
    st.markdown("### Generate a Single File")
    
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
            include_tests = st.checkbox("Include tests", value=True)
        
        submitted = st.form_submit_button("🚀 Generate", type="primary", use_container_width=True)
    
    if submitted and description:
        with st.spinner("Generating code..."):
            st.session_state.active_agents = [
                {"name": "Coding Agent", "status": "Analyzing requirements..."}
            ]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
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
            
            try:
                status_text.text("🤖 Initializing Coding Agent...")
                agent = CodingAgent()
                
                status_text.text("🔄 Creating event loop...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                status_text.text("⚡ Generating code with AI...")
                try:
                    result = loop.run_until_complete(agent.generate_module({
                        "module_name": Path(file_name).stem,
                        "purpose": description,
                        "requirements": description
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
                else:
                    st.error(f"❌ Generation failed: {result.get('message', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                st.code(traceback.format_exc(), language="python")
                st.session_state.active_agents = []
