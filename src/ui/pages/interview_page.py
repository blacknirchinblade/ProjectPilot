"""
UI Page for the Complex Interview Agent

This page provides a Streamlit interface for the 
ComplexInterviewAgent to generate technical and 
behavioral questions based on user-defined roles and topics.

Author: AutoCoder System
Date: October 21, 2025
"""

import streamlit as st
import asyncio
from loguru import logger
import traceback

# Import the agent and its dependencies
from src.agents.interview.complex_interview_agent import ComplexInterviewAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.utils.logger import setup_logger

def render_interview_page():
    """
    Renders the Streamlit UI for the Complex Interview Agent.
    """
    st.markdown("## 🎤 AI Interview Question Generator")
    st.info("""
    Prepare for your technical interviews! 
    Select a role, provide a topic (like your project's description), 
    and let the `ComplexInterviewAgent` generate a list of challenging questions.
    """)

    with st.form("interview_question_form"):
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input(
                "Job Role", 
                value="Senior AI/ML Engineer",
                help="The job title you are interviewing for."
            )
        with col2:
            difficulty = st.selectbox(
                "Difficulty",
                ("Medium", "Easy", "Hard"),
                index=0,
                help="The difficulty level of the questions."
            )
        
        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=10)
        with col2:
            tech_stack = st.text_input("Tech Stack (optional)", placeholder="e.g., Python, PyTorch")

        topic = st.text_area(
            "Project or Topic Description",
            height=150,
            value="A multi-agent AI system for end-to-end software development using a task-driven, iterative refinement workflow.",
            help="Describe the project, technology, or topic you want to be an expert on."
        )

        submitted = st.form_submit_button("🚀 Generate Questions", use_container_width=True)

    if submitted and topic:
        with st.spinner("🤖 ComplexInterviewAgent is thinking..."):
            try:
                # Initialize components
                llm_client = GeminiClient()
                prompt_manager = PromptManager()
                agent = ComplexInterviewAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                
                # Set up event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the agent
                questions_list = loop.run_until_complete(
                    agent.generate_interview_questions(
                        role=role,
                        topic=topic,
                        difficulty=difficulty,
                        num_questions=num_questions,
                        tech_stack=tech_stack
                    )
                )
                
                loop.close()

                if questions_list:
                    st.success(f"✅ Generated {len(questions_list)} questions!")
                    
                    for i, q_data in enumerate(questions_list, 1):
                        with st.expander(f"**Q{i}: {q_data.get('question', 'No question text')}**"):
                            st.markdown("#### 🔑 Key Points to Cover")
                            for point in q_data.get('answer_key', []):
                                st.markdown(f"- {point}")
                            
                            st.markdown("#### 🔄 Follow-up Questions")
                            for followup in q_data.get('follow_ups', []):
                                st.markdown(f"- *{followup}*")
                            
                            st.markdown(f"**Category:** `{q_data.get('category', 'N/A')}` | **Difficulty:** `{q_data.get('difficulty', 'N/A')}`")
                            
                else:
                    st.error("The agent failed to generate questions.")

            except Exception as e:
                st.error(f"An error occurred while running the agent: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    # This allows you to test the page independently
    # streamlit run src/ui/pages/interview_page.py
    setup_logger(log_level="INFO")
    render_interview_page()