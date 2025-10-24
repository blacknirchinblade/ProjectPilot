
import streamlit as st
from pathlib import Path
from datetime import datetime

def render_upload_page():
    """Render project upload page."""
    st.markdown("## 📤 Upload Project")
    
    st.markdown("""
    Upload an existing project to work with AutoCoder. Supported formats:
    - 📦 ZIP archives
    - 📁 Individual Python files
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["zip", "py"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        project_name = st.text_input("Project Name:", value=Path(uploaded_file.name).stem)
        project_description = st.text_area("Description:", height=100)
        
        if st.button("📤 Upload and Process", type="primary", use_container_width=True):
            with st.spinner("Processing upload..."):
                try:
                    # --- THIS IS THE FIX ---
                    # 1. Define output directory
                    output_dir = Path("output") / project_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # 2. Save and extract if it's a zip file
                    if uploaded_file.type == "application/zip":
                        zip_path = output_dir / uploaded_file.name
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        import zipfile
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(output_dir)
                        
                        zip_path.unlink() # Remove the zip file after extraction
                        
                        message = f"Unzipped to: `{output_dir}`"
                    else: # Handle single files
                        file_path = output_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        message = f"Saved file to: `{file_path}`"

                    # 3. Add to Project History
                    from src.ui.project import Project
                    new_project = Project(
                        id=project_name,
                        name=project_name,
                        description=project_description or "Uploaded project.",
                        path=str(output_dir),
                        project_type="Uploaded",
                        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    st.session_state.project_manager.add_project(new_project)
                    # --- END FIX ---
                    
                    st.success(f"✅ Successfully processed '{uploaded_file.name}'. {message}")
                    st.info("Project added to 'Recent Projects'. You can now chat with it on the History page.")

                except Exception as e:
                    st.error(f"❌ Error processing upload: {e}")
