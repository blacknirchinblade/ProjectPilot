
import streamlit as st

def render_settings_page():
    """Render settings page."""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["🎨 Appearance", "🤖 Agent Settings", "💾 Storage"])
    
    with tab1:
        st.markdown("### Appearance Settings")
        st.selectbox("Theme:", ["Dark", "Light", "Auto"])
        st.slider("Animation Speed:", 0.5, 2.0, 1.0, 0.1)
        st.checkbox("Show Agent Animations", value=True)
    
    with tab2:
        st.markdown("### Agent Configuration")
        st.slider("Default Temperature:", 0.0, 1.0, 0.4, 0.1)
        st.number_input("Max Retries:", 1, 5, 3)
        st.checkbox("Enable Code Validation", value=True)
    
    with tab3:
        st.markdown("### Storage Settings")
        st.checkbox("Auto-save Projects", value=True)
        st.number_input("Max History Items:", 10, 100, 50)
        
        if st.button("🗑️ Clear All History", type="secondary"):
            if st.confirm("Are you sure you want to clear all history?"):
                st.success("History cleared!")
