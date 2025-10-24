"""UI Styles for Streamlit Application
Defines custom CSS styles for various UI components in the AutoCoder Streamlit app.
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""
import streamlit as st

CUSTOM_CSS = """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86DE;
        --secondary-color: #54A0FF;
        --success-color: #10AC84;
        --warning-color: #F79F1F;
        --danger-color: #EE5A6F;
        --dark-bg: #1E1E1E;
        --light-bg: #2D2D2D;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        animation: fadeIn 1s ease-in;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Agent card animations */
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    .agent-card.active {
        animation: pulse 2s infinite;
    }
    
    /* Progress bar styling */
    .progress-container {
        background: var(--light-bg);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background: var(--primary-color);
        border-radius: 4px;
        transition: width 0.3s ease;
        animation: progressPulse 1.5s ease-in-out infinite;
    }
    
    /* Project card styling */
    .project-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .project-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Code preview styling */
    .code-preview {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes progressPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background: var(--success-color);
        color: white;
    }
    
    .status-warning {
        background: var(--warning-color);
        color: white;
    }
    
    .status-error {
        background: var(--danger-color);
        color: white;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
"""

def load_styles():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
