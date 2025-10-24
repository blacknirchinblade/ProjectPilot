"""UI Utilities for Streamlit Application
Utility functions for the AutoCoder Streamlit interface.
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""
import streamlit as st
import requests
import re
from streamlit_lottie import st_lottie

LOTTIE_ANIMATIONS = {
    # Analysis & Thinking
    "thinking": "https://assets2.lottiefiles.com/packages/lf20_touohxv0.json",
    "analyzing": "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json",

    # Coding & Generation
    "coding": "https://assets4.lottiefiles.com/packages/lf20_4kx2q32n.json",
    "building": "https://assets5.lottiefiles.com/packages/lf20_iorpbol0.json",

    # Success & Celebration
    "success": "https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json",
    "party": "https://assets8.lottiefiles.com/packages/lf20_rovf9gzu.json",

    # Working & Progress
    "loading": "https://assets4.lottiefiles.com/packages/lf20_a2chheio.json",
    "robot_working": "https://assets10.lottiefiles.com/packages/lf20_abqysclq.json",
    "process_image": "https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json",

    # Questions & Clarifications
    "questions": "https://assets7.lottiefiles.com/packages/lf20_dews3j6m.json",
    "conversation": "https://assets1.lottiefiles.com/packages/lf20_zjjlx2cs.json",

    # Project Types
    "ai_brain": "https://lottie.host/2Lb6PykY6B.json",
    "database": "https://assets5.lottiefiles.com/packages/lf20_jtbfg2nb.json",
    "rocket": "https://assets3.lottiefiles.com/packages/lf20_migux0tx.json",
}

def load_lottie_url(url: str) -> dict:
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.warning(f"Could not load animation: {e}")
    return None

def show_lottie_animation(animation_url: str, height: int = 200, key: str = None):
    """Display Lottie animation"""
    lottie_json = load_lottie_url(animation_url)
    if lottie_json:
        st_lottie(lottie_json, height=height, key=key)

def strip_markdown_code_fences(content: str) -> str:
    """
    Remove markdown code fences from generated code.
    """
    pattern = r'^```[\w]*\n(.*?)\n?```$'
    match = re.match(pattern, content.strip(), re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    lines = content.split('\n')
    
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    return '\n'.join(lines)
