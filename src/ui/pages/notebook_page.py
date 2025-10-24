"""
UI Page for Notebook (EDA) Agent

This page allows users to upload a Jupyter Notebook (.ipynb)
and have the NotebookAgent analyze it, generate EDA,
and suggest visualizations.

Author: AutoCoder System
Date: October 21, 2025
"""

import streamlit as st
import asyncio
from pathlib import Path
import os
import json
import traceback
from loguru import logger

from src.agents.notebook.notebook_agent import NotebookAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager

def render_notebook_page():
    """
    Renders the Streamlit UI for the Notebook EDA Agent.
    """
    st.markdown("### 📊 Notebook EDA & Analysis")
    st.info("""
    Upload your Jupyter Notebook (.ipynb) file. 
    The NotebookAgent will analyze its structure, data, and code to:
    - Provide a high-level summary.
    - Suggest new EDA steps or visualizations.
    - Identify potential improvements or bugs.
    """)

    uploaded_file = st.file_uploader(
        "Upload your .ipynb file",
        type=['ipynb'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Successfully uploaded: `{uploaded_file.name}`")

            if st.button("🚀 Run Notebook Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 NotebookAgent is analyzing your file..."):
                    try:
                        # Initialize components
                        llm_client = GeminiClient()
                        prompt_manager = PromptManager()
                        agent = NotebookAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                        
                        # Set up event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Define the task
                        task = {
                            "task_type": "analyze_notebook",
                            "data": {
                                "notebook_path": str(temp_file_path)
                            }
                        }
                        
                        # Run the agent
                        result = loop.run_until_complete(agent.execute_task(task))
                        
                        loop.close()

                        if result.get("status") == "success":
                            st.success("✅ Analysis Complete!")
                            
                            st.markdown("#### 📝 Analysis Summary")
                            st.markdown(result.get("analysis_summary", "No summary provided."))

                            st.markdown("#### 📈 Suggested Visualizations / EDA")
                            visualizations = result.get("visualizations", [])
                            if visualizations:
                                for vis in visualizations:
                                    st.markdown(f"- **{vis.get('type', 'Plot')}:** {vis.get('description', '')}")
                                    if vis.get('image_path'):
                                        st.image(vis.get('image_path'), caption=vis.get('description', ''))
                            else:
                                st.info("No new visualizations were generated.")
                            
                        else:
                            st.error(f"Analysis failed: {result.get('message', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"An error occurred while running the agent: {e}")
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"An error occurred while handling the file: {e}")
            st.code(traceback.format_exc())
        finally:
            # Clean up the temporary file
            if temp_file_path.exists():
                os.remove(temp_file_path)