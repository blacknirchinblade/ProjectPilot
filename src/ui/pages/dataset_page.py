"""
UI Page for Dataset Agent

This page allows users to generate synthetic datasets
using the DatasetAgent.

Author: AutoCoder System
Date: October 21, 2025
"""

import streamlit as st
import asyncio
import traceback
from loguru import logger

from src.agents.dataset.dataset_agent import DatasetAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager

def render_dataset_generation_page():
    """
    Renders the Streamlit UI for the Dataset Agent.
    """
    st.markdown("### 🧬 Synthetic Dataset Generator")
    st.info("""
    Define a schema and let the DatasetAgent generate a Python script 
    (using libraries like Faker and Scikit-learn) to create synthetic data for your ML models.
    """)

    with st.form("dataset_generation_form"):
        col1, col2 = st.columns(2)
        with col1:
            task_type_ml = st.selectbox(
                "Select Task Type",
                ("classification", "regression", "clustering"),
                index=0,
                help="The type of ML task this dataset is for."
            )
        with col2:
            num_samples = st.number_input(
                "Number of Samples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        features_text = st.text_area(
            "Define Features (one per line)",
            height=150,
            placeholder="Example:\n- user_age (int, 18-65)\n- purchase_amount (float, 5.0-500.0)\n- product_category (categorical, ['electronics', 'clothing', 'home'])\n- sentiment (str, 'positive'/'negative'/'neutral')"
        )
        
        submitted = st.form_submit_button("🚀 Generate Dataset Script", use_container_width=True)

    if submitted and features_text:
        with st.spinner("🤖 DatasetAgent is designing your data..."):
            try:
                # Parse features
                features = [f.strip() for f in features_text.split('\n') if f.strip()]
                
                if not features:
                    st.error("Please define at least one feature.")
                    return

                # Initialize components
                llm_client = GeminiClient()
                prompt_manager = PromptManager()
                agent = DatasetAgent(llm_client=llm_client, prompt_manager=prompt_manager)
                
                # Set up event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Define the task
                task = {
                    "task_type": "create_ml_dataset",
                    "data": {
                        "task_type_ml": task_type_ml,
                        "features": features,
                        "num_samples": num_samples
                    }
                }
                
                # Run the agent
                result = loop.run_until_complete(agent.execute_task(task))
                
                loop.close()

                if result.get("status") == "success":
                    st.success("✅ Dataset generation script created successfully!")
                    
                    st.markdown("#### 🐍 Python Script")
                    st.code(result.get("dataset_code", "# No code generated."), language="python")
                    
                    st.download_button(
                        label="📥 Download Script",
                        data=result.get("dataset_code", ""),
                        file_name="generate_dataset.py",
                        mime="text/x-python"
                    )
                    
                    with st.expander("View Splits Information"):
                        st.json(result.get("splits", {}))
                        
                else:
                    st.error(f"Generation failed: {result.get('message', 'Unknown error')}")

            except Exception as e:
                st.error(f"An error occurred while running the agent: {e}")
                st.code(traceback.format_exc())