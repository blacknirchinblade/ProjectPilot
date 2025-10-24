"""
Streamlit Agent - Generates Interactive Dashboard Applications

This agent creates complete Streamlit applications for ML/DL projects including:
- Interactive dashboards
- File upload components
- Data visualization pages
- Model inference interfaces
- Real-time prediction UIs
- Charts and metrics displays
- API integration
- Multi-page applications

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class StreamlitAgent:
    """
    StreamlitAgent generates complete Streamlit dashboard applications.
    
    Features:
    - Single-page and multi-page dashboards
    - File upload components (images, CSV, text)
    - Visualization pages (charts, metrics, tables)
    - Model inference interfaces
    - API integration (REST API calls)
    - Real-time updates
    - Custom themes and styling
    - Session state management
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize the Streamlit Agent.
        
        Args:
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
        """
        super().__init__(
            name="streamlit_agent",
            role="Expert Streamlit UI/UX Developer",
            agent_type="coding", # Use 'coding' temp
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        # Component templates
        self.templates = {
            "image_classification": self._get_image_classification_template,
            "text_classification": self._get_text_classification_template,
            "object_detection": self._get_object_detection_template,
            "time_series": self._get_time_series_template,
            "genai_chat": self._get_genai_chat_template,
            "genai_rag": self._get_genai_rag_template,
            "data_analysis": self._get_data_analysis_template,
            "model_comparison": self._get_model_comparison_template,
        }
        logger.info(f"{self.name} initialized for Streamlit app generation")
    
    async def generate_dashboard(
        self,
        project_type: str,
        project_name: str,
        api_url: str = "http://api:8000",
        features: Optional[Dict[str, bool]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate a complete Streamlit dashboard application.
        
        Args:
            project_type: Type of ML project (image_classification, text_classification, etc.)
            project_name: Name of the project
            api_url: URL of the FastAPI backend
            features: Dictionary of enabled features
            config: Additional configuration
        
        Returns:
            Dictionary with generated files: {filename: code}
        """

        # Ensure clients are available if not passed during init
        if not self.llm_client or not self.prompt_manager:
            logger.warning("Re-initializing LLM client for StreamlitAgent. Should be passed in __init__.")
            self.llm_client = GeminiClient()
            self.prompt_manager = PromptManager()
        features = features or {}
        config = config or {}
        
        # Generate main app
        main_app = self._generate_main_app(
            project_type, project_name, api_url, features, config
        )
        
        # Generate pages if multi-page enabled
        pages = {}
        if features.get("multi_page", True):
            pages = self._generate_pages(project_type, api_url, features, config)
        
        # Generate components
        components = self._generate_components(project_type, features)
        
        # Generate utilities
        utils = self._generate_utils(api_url)
        
        # Generate config files
        config_files = self._generate_config_files(project_name, features)
        
        # Combine all files
        result = {
            "app.py": main_app,
            **pages,
            **components,
            **utils,
            **config_files
        }
        
        return result
    
    def _generate_main_app(
        self,
        project_type: str,
        project_name: str,
        api_url: str,
        features: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Generate the main Streamlit app file."""
        
        # Get template for project type
        template_func = self.templates.get(
            project_type,
            self._get_generic_template
        )
        
        return template_func(project_name, api_url, features, config)
    
    def _get_image_classification_template(
        self,
        project_name: str,
        api_url: str,
        features: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Generate image classification dashboard."""
        
        has_webcam = features.get("webcam_input", False)
        has_batch = features.get("batch_processing", False)
        has_metrics = features.get("model_metrics", True)
        has_visualization = features.get("visualization", True)
        
        code = f'''"""
{project_name.replace("_", " ").title()} - Image Classification Dashboard

Interactive Streamlit dashboard for image classification using ML model.
"""

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional

# Page configuration
st.set_page_config(
    page_title="{project_name.replace("_", " ").title()}",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }}
    .prediction-box {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "{api_url}"

# Initialize session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False

def check_api_health() -> bool:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{{API_URL}}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_image(image: Image.Image, confidence_threshold: float = 0.5) -> Optional[Dict]:
    """Send image to API for prediction."""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Send to API
        files = {{"file": ("image.png", img_byte_arr, "image/png")}}
        response = requests.post(f"{{API_URL}}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Add to history
            st.session_state.prediction_history.append({{
                "class": result["class"],
                "confidence": result["confidence"],
                "timestamp": pd.Timestamp.now()
            }})
            
            return result
        else:
            st.error(f"API Error: {{response.status_code}}")
            return None
    except Exception as e:
        st.error(f"Prediction failed: {{str(e)}}")
        return None

# Sidebar
with st.sidebar:
    st.title("🖼️ {project_name.replace('_', ' ').title()}")
    st.markdown("---")
    
    # API Status
    st.markdown("### API Status")
    if check_api_health():
        st.success("✅ API Online")
        st.session_state.api_healthy = True
    else:
        st.error("❌ API Offline")
        st.session_state.api_healthy = False
    
    st.markdown("---")
    
    # Settings
    st.markdown("### Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for accepting prediction"
    )
    
    show_probabilities = st.checkbox(
        "Show All Probabilities",
        value=True,
        help="Display probability distribution for all classes"
    )
    
    show_top_k = st.slider(
        "Top K Predictions",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of top predictions to display"
    )
    
    st.markdown("---")
    
    # About
    st.markdown("### About")
    st.info(
        "Upload an image to get real-time predictions from our "
        "trained ML model. The model analyzes the image and returns "
        "the predicted class with confidence scores."
    )
    
    # Model Info
    {"" if not has_metrics else """
    st.markdown("---")
    st.markdown("### Model Performance")
    try:
        metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            col2.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
    except:
        st.info("Metrics unavailable")
    """}

# Main Content
st.markdown('<h1 class="main-header">🖼️ Image Classification</h1>', unsafe_allow_html=True)

# Create tabs for different input methods
tabs = st.tabs(["📁 Upload Image"{"" if not has_webcam else ', "📷 Webcam"'}{"" if not has_batch else ', "📦 Batch Processing"'}])

# Tab 1: File Upload
with tabs[0]:
    st.markdown("### Upload an Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Supported formats: PNG, JPG, JPEG, WEBP"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("#### Image Information")
            st.write(f"**Size:** {{image.size[0]}} x {{image.size[1]}} pixels")
            st.write(f"**Format:** {{image.format}}")
            st.write(f"**Mode:** {{image.mode}}")
        
        with col2:
            # Prediction button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                if not st.session_state.api_healthy:
                    st.error("⚠️ API is offline. Please check your connection.")
                else:
                    with st.spinner("🤖 Analyzing image..."):
                        result = predict_image(image, confidence_threshold)
                        
                        if result:
                            # Display prediction in styled box
                            if result["confidence"] >= confidence_threshold:
                                st.markdown(
                                    f'<div class="prediction-box">'
                                    f'<h2>✅ Prediction</h2>'
                                    f'<h1>{{result["class"]}}</h1>'
                                    f'<h3>Confidence: {{result["confidence"]:.2%}}</h3>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.warning(
                                    f"⚠️ Low confidence prediction: **{{result['class']}}** "
                                    f"({{result['confidence']:.2%}})"
                                )
                            
                            if has_visualization:
                                code_block = f"""
# Probability distribution
if show_probabilities and "probabilities" in result:
    st.markdown("#### 📊 Probability Distribution")
    
    # Prepare data
    probs_df = pd.DataFrame({{{{
        'Class': list(result['probabilities'].keys()),
        'Probability': list(result['probabilities'].values())
    }}}}).sort_values('Probability', ascending=False)
    
    # Bar chart
    fig = px.bar(
        probs_df, 
        x='Probability', 
        y='Class', 
        orientation='h',
        title="Top Predictions",
        labels={{
"Class": "Predicted Class", "Probability": "Confidence Score"}},
        text='Probability'
    )
    fig.update_traces(texttemplate='%{{text:.2%}}', textposition='outside')
    fig.update_layout(yaxis_title="", xaxis_title="Confidence")
    st.plotly_chart(fig, use_container_width=True)
"""
                                st.code(code_block, language='python')

            # Sidebar for configuration
            st.sidebar.header("⚙️ Configuration")
            # Confidence threshold
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.5, 0.01,
                "Minimum confidence for prediction"
            )
            
            # Show probabilities
            show_probabilities = st.sidebar.checkbox(
                "Show Probabilities",
                True,
                "Display class probabilities"
            )
            
            # Top K predictions
            show_top_k = st.sidebar.slider(
                "Top K Predictions",
                1, 10, 3,
                "Number of top predictions to display"
            )

# Prediction History
if st.session_state.prediction_history:
    with st.expander("📜 Prediction History"):
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">Powered by {project_name.replace("_", " ").title()} | '
    'Built with Streamlit 🎈</p>',
    unsafe_allow_html=True
)
'''
        
        return code
    
    def _get_text_classification_template(
        self,
        project_name: str,
        api_url: str,
        features: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Generate text classification dashboard."""
        
        code = f'''"""
{project_name.replace("_", " ").title()} - Text Classification Dashboard
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from typing import Dict, Optional

st.set_page_config(
    page_title="{project_name.replace("_", " ").title()}",
    page_icon="📝",
    layout="wide"
)

API_URL = "{api_url}"

# Sidebar
with st.sidebar:
    st.title("📝 Text Classifier")
    st.markdown("---")
    
    # Settings
    st.markdown("### Settings")
    max_length = st.slider("Max Text Length", 100, 1000, 500)
    show_probabilities = st.checkbox("Show Probabilities", value=True)

# Main content
st.title("📝 Text Classification")

# Input methods
tab1, tab2 = st.tabs(["✍️ Enter Text", "📁 Upload File"])

with tab1:
    st.markdown("### Enter text to classify")
    
    text_input = st.text_area(
        "Text",
        height=200,
        max_chars=max_length,
        placeholder="Enter your text here..."
    )
    
    if st.button("🔍 Classify Text", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{{API_URL}}/predict",
                        json={{"text": text_input}}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        st.success(f"✅ Prediction: **{{result['class']}}**")
                        st.metric("Confidence", f"{{result['confidence']:.2%}}")
                        
                        # Probabilities
                        if show_probabilities and "probabilities" in result:
                            st.markdown("#### Probability Distribution")
                            probs_df = pd.DataFrame({{
                                'Class': list(result['probabilities'].keys()),
                                'Probability': list(result['probabilities'].values())
                            }})
                            
                            fig = px.bar(probs_df, x='Class', y='Probability')
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Prediction failed")
                except Exception as e:
                    st.error(f"Error: {{str(e)}}")
        else:
            st.warning("Please enter some text")

with tab2:
    st.markdown("### Upload a text file")
    
    uploaded_file = st.file_uploader("Choose file", type=['txt', 'csv'])
    
    if uploaded_file:
        content = uploaded_file.read().decode('utf-8')
        st.text_area("File content", content, height=200)
        
        if st.button("🔍 Classify File Content", type="primary"):
            # Similar processing as tab1
            st.info("Processing file...")
'''
        
        return code
    
    def _get_genai_chat_template(
        self,
        project_name: str,
        api_url: str,
        features: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Generate GenAI chat interface."""
        
        code = f'''"""
{project_name.replace("_", " ").title()} - GenAI Chat Interface
"""

import streamlit as st
import requests
from typing import List, Dict

st.set_page_config(
    page_title="{project_name.replace("_", " ").title()}",
    page_icon="🤖",
    layout="wide"
)

API_URL = "{api_url}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🤖 GenAI Chat")
    st.markdown("---")
    
    # Model settings
    st.markdown("### Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 Chat with AI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({{"role": "user", "content": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{{API_URL}}/chat",
                    json={{
                        "message": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result["response"]
                    st.markdown(ai_response)
                    
                    # Add to history
                    st.session_state.messages.append({{
                        "role": "assistant",
                        "content": ai_response
                    }})
                else:
                    st.error("Failed to get response")
            except Exception as e:
                st.error(f"Error: {{str(e)}}")
'''
        
        return code
    
    def _get_genai_rag_template(
        self,
        project_name: str,
        api_url: str,
        features: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Generate GenAI RAG (Retrieval-Augmented Generation) interface."""
        
        code = f'''"""
{project_name.replace("_", " ").title()} - RAG Q&A System
"""

import streamlit as st
import requests
from typing import List

st.set_page_config(
    page_title="{project_name.replace("_", " ").title()}",
    page_icon="📚",
    layout="wide"
)

API_URL = "{api_url}"

# Sidebar
with st.sidebar:
    st.title("📚 RAG Q&A System")
    st.markdown("---")
    
    # Upload documents
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Add to knowledge base",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📥 Index Documents"):
            with st.spinner("Indexing..."):
                # Upload to API for indexing
                files = [{{"file": f}} for f in uploaded_files]
                response = requests.post(f"{{API_URL}}/index", files=files)
                if response.status_code == 200:
                    st.success(f"Indexed {{len(uploaded_files)}} documents")
    
    st.markdown("---")
    st.markdown("### Settings")
    top_k = st.slider("Retrieved Chunks", 1, 10, 3)
    show_sources = st.checkbox("Show Sources", value=True)

# Main content
st.title("📚 Ask Questions About Your Documents")

# Query input
question = st.text_input(
    "Your question",
    placeholder="What would you like to know?"
)

if st.button("🔍 Search", type="primary"):
    if question:
        with st.spinner("Searching knowledge base..."):
            try:
                response = requests.post(
                    f"{{API_URL}}/query",
                    json={{"question": question, "top_k": top_k}}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.success(result["answer"])
                    
                    # Show sources
                    if show_sources and "sources" in result:
                        with st.expander("📖 Sources"):
                            for idx, source in enumerate(result["sources"], 1):
                                st.markdown(f"**Source {{idx}}:**")
                                st.write(source["content"])
                                st.caption(f"Relevance: {{source['score']:.2%}}")
                                st.markdown("---")
            except Exception as e:
                st.error(f"Error: {{str(e)}}")
    else:
        st.warning("Please enter a question")
'''
        
        return code
    
    def _get_object_detection_template(self, project_name: str, api_url: str, features: Dict, config: Dict) -> str:
        """Generate object detection dashboard."""
        return "# Object Detection Dashboard\nimport streamlit as st\n# TODO: Implement"
    
    def _get_time_series_template(self, project_name: str, api_url: str, features: Dict, config: Dict) -> str:
        """Generate time series dashboard."""
        return "# Time Series Dashboard\nimport streamlit as st\n# TODO: Implement"
    
    def _get_data_analysis_template(self, project_name: str, api_url: str, features: Dict, config: Dict) -> str:
        """Generate data analysis dashboard."""
        return "# Data Analysis Dashboard\nimport streamlit as st\n# TODO: Implement"
    
    def _get_model_comparison_template(self, project_name: str, api_url: str, features: Dict, config: Dict) -> str:
        """Generate model comparison dashboard."""
        return "# Model Comparison Dashboard\nimport streamlit as st\n# TODO: Implement"
    
    def _get_generic_template(self, project_name: str, api_url: str, features: Dict, config: Dict) -> str:
        """Generate generic dashboard."""
        return "# Generic ML Dashboard\nimport streamlit as st\n# TODO: Implement"
    
    def _generate_pages(
        self,
        project_type: str,
        api_url: str,
        features: Dict[str, bool],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate multi-page application files."""
        
        pages = {}
        
        # Performance metrics page
        if features.get("model_metrics", True):
            pages["pages/1_📊_Performance_Metrics.py"] = self._create_metrics_page(api_url)
        
        # Settings page
        pages["pages/2_⚙️_Settings.py"] = self._create_settings_page()
        
        # History page
        if features.get("prediction_history", True):
            pages["pages/3_📜_History.py"] = self._create_history_page()
        
        return pages
    
    def _create_metrics_page(self, api_url: str) -> str:
        """Create performance metrics page."""
        return f'''"""Performance Metrics Page"""

import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Performance Metrics", page_icon="📊")

st.title("📊 Model Performance Metrics")

# Fetch metrics from API
try:
    response = requests.get("{api_url}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{{metrics.get('accuracy', 0):.2%}}")
        col2.metric("Precision", f"{{metrics.get('precision', 0):.2%}}")
        col3.metric("Recall", f"{{metrics.get('recall', 0):.2%}}")
        col4.metric("F1-Score", f"{{metrics.get('f1_score', 0):.3f}}")
        
        # Confusion matrix
        if "confusion_matrix" in metrics:
            st.markdown("### Confusion Matrix")
            # TODO: Plot confusion matrix
    else:
        st.error("Failed to fetch metrics")
except Exception as e:
    st.error(f"Error: {{str(e)}}")
'''
    
    def _create_settings_page(self) -> str:
        """Create settings page."""
        return '''"""Settings Page"""

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️")

st.title("⚙️ Application Settings")

# API Configuration
st.markdown("### API Configuration")
api_url = st.text_input("API URL", value="http://api:8000")
api_timeout = st.number_input("Timeout (seconds)", value=30)

# Display Settings
st.markdown("### Display Settings")
theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
show_advanced = st.checkbox("Show Advanced Options")

if st.button("💾 Save Settings"):
    st.success("Settings saved!")
'''
    
    def _create_history_page(self) -> str:
        """Create prediction history page."""
        return '''"""Prediction History Page"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="History", page_icon="📜")

st.title("📜 Prediction History")

if "prediction_history" in st.session_state:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
        
        # Download button
        csv = history_df.to_csv(index=False)
        st.download_button(
            "📥 Download History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )
    else:
        st.info("No predictions yet")
else:
    st.info("No history available")
'''
    
    def _generate_components(
        self,
        project_type: str,
        features: Dict[str, bool]
    ) -> Dict[str, str]:
        """Generate reusable component files."""
        
        components = {}
        
        # File uploader component
        components["components/__init__.py"] = ""
        components["components/file_uploader.py"] = self._create_file_uploader_component()
        
        # Result display component
        components["components/result_display.py"] = self._create_result_display_component()
        
        # Metrics chart component
        if features.get("visualization", True):
            components["components/metrics_chart.py"] = self._create_metrics_chart_component()
        
        return components
    
    def _create_file_uploader_component(self) -> str:
        """Create file uploader component."""
        return '''"""Reusable File Uploader Component"""

import streamlit as st
from typing import List, Optional

def create_file_uploader(
    label: str = "Upload file",
    file_types: Optional[List[str]] = None,
    multiple: bool = False,
    help_text: Optional[str] = None
):
    """
    Create a styled file uploader component.
    
    Args:
        label: Label for the uploader
        file_types: Allowed file extensions
        multiple: Allow multiple files
        help_text: Help text to display
    
    Returns:
        Uploaded file(s) or None
    """
    file_types = file_types or ['png', 'jpg', 'jpeg']
    
    uploaded = st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=multiple,
        help=help_text or f"Supported formats: {', '.join(file_types).upper()}"
    )
    
    return uploaded
'''
    
    def _create_result_display_component(self) -> str:
        """Create result display component."""
        return '''"""Result Display Component"""

import streamlit as st
from typing import Dict, Optional

def display_prediction_result(
    result: Dict,
    confidence_threshold: float = 0.5,
    show_styled: bool = True
):
    """
    Display prediction result in a styled format.
    
    Args:
        result: Prediction result dictionary
        confidence_threshold: Threshold for acceptance
        show_styled: Use styled HTML display
    """
    if show_styled:
        if result["confidence"] >= confidence_threshold:
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
                f'color: white; padding: 2rem; border-radius: 1rem; text-align: center;">'
                f'<h2>✅ Prediction</h2>'
                f'<h1>{result["class"]}</h1>'
                f'<h3>Confidence: {result["confidence"]:.2%}</h3>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning(f"⚠️ Low confidence: {result['class']} ({result['confidence']:.2%})")
    else:
        st.success(f"Prediction: **{result['class']}**")
        st.metric("Confidence", f"{result['confidence']:.2%}")
'''
    
    def _create_metrics_chart_component(self) -> str:
        """Create metrics visualization component."""
        return '''"""Metrics Visualization Component"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_probability_chart(probabilities: dict, top_k: int = 5):
    """Create probability distribution bar chart."""
    df = pd.DataFrame({
        'Class': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    }).sort_values('Probability', ascending=False).head(top_k)
    
    fig = px.bar(
        df,
        x='Class',
        y='Probability',
        color='Probability',
        color_continuous_scale='viridis',
        title=f'Top {top_k} Predictions'
    )
    
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_confusion_matrix(cm_data):
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    
    return fig
'''
    
    def _generate_utils(self, api_url: str) -> Dict[str, str]:
        """Generate utility files."""
        
        utils = {}
        
        # API client
        utils["utils/__init__.py"] = ""
        utils["utils/api_client.py"] = self._create_api_client(api_url)
        
        # Image processing utilities
        utils["utils/image_processing.py"] = self._create_image_processing_utils()
        
        return utils
    
    def _create_api_client(self, api_url: str) -> str:
        """Create API client utility."""
        return f'''"""API Client for Backend Communication"""

import requests
from typing import Dict, Optional, Any
import io
from PIL import Image

class APIClient:
    """Client for communicating with FastAPI backend."""
    
    def __init__(self, base_url: str = "{api_url}"):
        self.base_url = base_url
        self.timeout = 30
    
    def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{{self.base_url}}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict_image(self, image: Image.Image) -> Optional[Dict]:
        """Send image for prediction."""
        try:
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Send request
            files = {{"file": ("image.png", img_byte_arr, "image/png")}}
            response = requests.post(
                f"{{self.base_url}}/predict",
                files=files,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Prediction error: {{e}}")
            return None
    
    def predict_text(self, text: str) -> Optional[Dict]:
        """Send text for prediction."""
        try:
            response = requests.post(
                f"{{self.base_url}}/predict",
                json={{"text": text}},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Prediction error: {{e}}")
            return None
    
    def get_metrics(self) -> Optional[Dict]:
        """Get model performance metrics."""
        try:
            response = requests.get(f"{{self.base_url}}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
'''
    
    def _create_image_processing_utils(self) -> str:
        """Create image processing utilities."""
        return '''"""Image Processing Utilities"""

from PIL import Image
import numpy as np
from typing import Tuple

def resize_image(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize image to target size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def normalize_image(image: Image.Image) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img_array = np.array(image).astype(np.float32) / 255.0
    return img_array

def preprocess_image(
    image: Image.Image,
    size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """Complete image preprocessing pipeline."""
    # Resize
    image = resize_image(image, size)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Normalize
    if normalize:
        return normalize_image(image)
    
    return np.array(image)
'''
    
    def _generate_config_files(
        self,
        project_name: str,
        features: Dict[str, bool]
    ) -> Dict[str, str]:
        """Generate configuration files."""
        
        config_files = {}
        
        # Requirements.txt
        config_files["requirements.txt"] = self._create_requirements(features)
        
        # Streamlit config
        config_files[".streamlit/config.toml"] = self._create_streamlit_config()
        
        # README
        config_files["README.md"] = self._create_readme(project_name)
        
        # Dockerfile
        config_files["Dockerfile"] = self._create_dockerfile()
        
        return config_files
    
    def _create_requirements(self, features: Dict[str, bool]) -> str:
        """Create requirements.txt."""
        
        requirements = [
            "streamlit>=1.28.0",
            "requests>=2.31.0",
            "pillow>=10.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
        ]
        
        if features.get("visualization", True):
            requirements.extend([
                "plotly>=5.17.0",
                "matplotlib>=3.7.0",
                "seaborn>=0.12.0"
            ])
        
        return "\n".join(requirements)
    
    def _create_streamlit_config(self) -> str:
        """Create Streamlit configuration."""
        return '''[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
'''
    
    def _create_readme(self, project_name: str) -> str:
        """Create README file."""
        return f'''# {project_name.replace("_", " ").title()} - Streamlit Dashboard

Interactive dashboard for ML model predictions.

## Features

- 🖼️ Image/Text upload and classification
- 📊 Real-time predictions
- 📈 Probability visualization
- 📜 Prediction history
- ⚙️ Configurable settings

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Navigate to http://localhost:8501 in your browser.

## Configuration

Edit `.streamlit/config.toml` to customize theme and settings.

## API Integration

The dashboard connects to the FastAPI backend at `http://api:8000`.
Make sure the API is running before starting the dashboard.

## Docker

```bash
docker build -t {project_name}-frontend .
docker run -p 8501:8501 {project_name}-frontend
```
'''
    
    def _create_dockerfile(self) -> str:
        """Create Dockerfile for Streamlit app."""
        return '''FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''


# Example usage
async def main():
    """Example usage of StreamlitAgent."""
    agent = StreamlitAgent()
    
    # Generate image classification dashboard
    result = await agent.generate_dashboard(
        project_type="image_classification",
        project_name="cifar10_classifier",
        api_url="http://api:8000",
        features={
            "multi_page": True,
            "webcam_input": True,
            "batch_processing": True,
            "model_metrics": True,
            "visualization": True,
            "prediction_history": True
        }
    )
    
    print(f"Generated {len(result)} files:")
    for filename in result.keys():
        print(f"  - {filename}")


if __name__ == "__main__":
    asyncio.run(main())
