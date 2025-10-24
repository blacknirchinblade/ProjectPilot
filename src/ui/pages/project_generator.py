
import streamlit as st
import asyncio
import json
from pathlib import Path
from datetime import datetime
import zipfile
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
from src.agents.prompt_engineering.prompt_engineer_agent import PromptEngineerAgent
from src.agents.architecture.dynamic_architect_agent import DynamicArchitectAgent
from src.agents.planning.planning_agent import PlanningAgent
from src.agents.dataset.dataset_agent import DatasetAgent
from src.agents.specialized.database_agent import DatabaseAgent
from src.agents.specialized.api_agent import APIAgent
from src.agents.specialized.streamlit_agent import StreamlitAgent
from src.agents.specialized.genai_agent import GenAIAgent
from src.agents.specialized.deployment_agent import DeploymentAgent
from src.agents.testing.testing_agent import TestingAgent
from src.agents.documentation.advanced_doc_agent import AdvancedDocumentationAgent
from src.agents.review.review_orchestrator import ReviewOrchestrator
from src.agents.refactoring.refactoring_agent import RefactoringAgent
from src.ui.utils import show_lottie_animation, LOTTIE_ANIMATIONS, strip_markdown_code_fences
import traceback
def generate_complete_project(config, loop=None):
    """
    Generate complete AI/ML project with all selected components.
    """
    st.markdown("---")
    st.markdown("### 🔄 Generating Your Complete Project")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    close_loop = False
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        close_loop = True
    
    try:
        all_files = {}
        project_name = config["project_name"]

        if "project_type" not in config:
            config["project_type"] = config.get("category", "custom_project")
        if "requirements" not in config:
            config["requirements"] = config.get("description", "")

        st.markdown("#### 🧠 Phase 0: Intelligent Analysis & Planning")

        status_text.text("🔍 Step 0.1/0.3: Enhancing requirements...")
        progress_bar.progress(2)

        with st.spinner("Analyzing requirements..."):
            llm_client = GeminiClient()
            prompt_manager = PromptManager()
            prompt_engineer = PromptEngineerAgent(llm_client=llm_client, prompt_manager=prompt_manager)
            
            enhanced_description = config["description"]
            if "clarifications" in config and config["clarifications"]:
                clarifications_text = "\n\nUser Clarifications:\n"
                for q_id, answer in config["clarifications"].items():
                    clarifications_text += f"- {q_id}: {answer}\n"
                enhanced_description += clarifications_text

            context = {
                "project_type": config.get("project_type", "custom_project"),
                "requirements": config.get("description", ""),
                "components": config["components"],
            }

            enhanced_spec = loop.run_until_complete(
                prompt_engineer.enhance_user_input(user_input=enhanced_description, context=context)
            )
            st.success("✅ Requirements analyzed and enhanced!")
        
        status_text.text("🏗️ Step 0.2/0.3: Designing system architecture...")
        progress_bar.progress(5)
        
        with st.spinner("Planning dynamic architecture..."):
            architect_agent = DynamicArchitectAgent(llm_client=llm_client, prompt_manager=prompt_manager)
            architecture = loop.run_until_complete(
                architect_agent.design_architecture({"enhanced_spec": enhanced_spec, "components": config["components"]})
            )
            st.success("✅ Dynamic architecture designed!")
            
        status_text.text("📝 Step 0.3/0.3: Breaking down into tasks...")
        progress_bar.progress(8)
        
        with st.spinner("Creating task breakdown..."):
            planning_agent = PlanningAgent(llm_client=llm_client, prompt_manager=prompt_manager)
            task_breakdown = loop.run_until_complete(
                planning_agent.breakdown_tasks({"architecture": architecture, "enhanced_spec": enhanced_spec})
            )
            st.success(f"✅ Tasks identified: {len(task_breakdown.get('tasks', []))} components")
        
        progress_bar.progress(10)
        st.markdown("---")
        st.markdown("#### 🏗️ Phase 1: Code Generation")
        
        config["enhanced_specification"] = enhanced_spec
        config["architecture"] = architecture
        config["task_breakdown"] = task_breakdown

        data_intensive_categories = ["Image Classification", "Object Detection", "Text Classification"]
        if any(cat in config["category"] for cat in data_intensive_categories):
            status_text.text("📊 Step 1.0: Generating Data Preparation Scripts...")
            progress_bar.progress(11)
            with st.spinner("Invoking DatasetAgent..."):
                dataset_agent = DatasetAgent()
                data_scripts = loop.run_until_complete(dataset_agent.generate_scripts(config))
                for filename, content in data_scripts.items():
                    all_files[f"src/data/{filename}"] = strip_markdown_code_fences(content)
                st.success(f"✅ DatasetAgent: {len(data_scripts)} scripts generated.")
                progress_bar.progress(14)
        
        if config["components"].get("database"):
            status_text.text("🗄️ Step 1.1: Generating Database Models...")
            progress_bar.progress(12)
            with st.spinner("Creating database models..."):
                database_agent = DatabaseAgent()
                db_files = loop.run_until_complete(database_agent.generate_database_models(project_name=project_name, config=config))
                for filename, content in db_files.items():
                    all_files[f"backend/{filename}"] = strip_markdown_code_fences(content)
                st.success(f"✅ Database: {len(db_files)} files generated")
                progress_bar.progress(18)

        if config["components"].get("fastapi"):
            status_text.text("🌐 Step 1.2: Generating API Endpoints...")
            progress_bar.progress(20)
            with st.spinner("Creating API endpoints..."):
                api_agent = APIAgent()
                api_files = loop.run_until_complete(api_agent.generate_api(project_name=project_name, config=config))
                for filename, content in api_files.items():
                    all_files[f"backend/{filename}"] = strip_markdown_code_fences(content)
                st.success(f"✅ API: {len(api_files)} files generated")
                progress_bar.progress(25)

        if config["components"]["streamlit"]:
            status_text.text("📊 Step 1.3: Generating Interactive Dashboard...")
            progress_bar.progress(30)
            with st.spinner("Creating Streamlit UI..."):
                streamlit_agent = StreamlitAgent()
                dashboard_files = loop.run_until_complete(
                    streamlit_agent.generate_dashboard(project_type="image_classification", project_name=project_name)
                )
                for filename, content in dashboard_files.items():
                    all_files[f"frontend/{filename}"] = strip_markdown_code_fences(content)
                st.success(f"✅ Dashboard: {len(dashboard_files)} files generated")
                progress_bar.progress(35)

        if "GenAI" in config["category"]:
            status_text.text("🧠 Step 1.4: Generating GenAI System...")
            progress_bar.progress(45)
            with st.spinner("Creating GenAI architecture..."):
                genai_agent = GenAIAgent()
                genai_files = loop.run_until_complete(
                    genai_agent.generate_genai_project(project_type="rag_system", project_name=project_name, config=config)
                )
                for filename, content in genai_files.items():
                    all_files[f"backend/{filename}"] = strip_markdown_code_fences(content)
                st.success(f"✅ GenAI: {len(genai_files)} files generated")
                progress_bar.progress(65)

        st.markdown("---")
        st.markdown("#### 🚀 Phase 2: Deployment & Documentation")

        if config["components"]["docker"]:
            status_text.text("🐳 Step 2.1: Generating Deployment Configuration...")
            progress_bar.progress(75)
            with st.spinner("Creating deployment files..."):
                deployment_agent = DeploymentAgent()
                deployment_files = loop.run_until_complete(
                    deployment_agent.generate_deployment_config(project_name=project_name, components=config["components"], config=config)
                )
                for filename, content in deployment_files.items():
                    all_files[f"deployment/{filename}"] = strip_markdown_code_fences(content)
                st.success(f"✅ Deployment: {len(deployment_files)} files generated")
                progress_bar.progress(95)

        status_text.text("🧪 Step 2.2: Generating Tests, Docs, and Reviews...")
        progress_bar.progress(97)
        with st.spinner("Running quality assurance agents..."):
            testing_agent = TestingAgent()
            advanced_doc_agent = AdvancedDocumentationAgent(llm_client, prompt_manager)
            review_orchestrator = ReviewOrchestrator()
            refactoring_agent = RefactoringAgent()

            test_files = loop.run_until_complete(
                loop.run_in_executor(None, testing_agent.generate_tests, project_name, all_files)
            )
            doc_suite = loop.run_until_complete(
                advanced_doc_agent.generate_documentation_suite(config, list(all_files.keys()))
            )
            review_report = loop.run_until_complete(
                review_orchestrator.orchestrate_reviews(all_files)
            )
            refactored_files = loop.run_until_complete(
                loop.run_in_executor(None, refactoring_agent.refactor_code, project_name, all_files)
            )

            for filename, content in test_files.items():
                all_files[f"tests/{filename}"] = strip_markdown_code_fences(content)
            for filename, content in doc_suite.items():
                all_files[filename] = content
            all_files["reviews/consolidated_review_report.json"] = json.dumps(review_report, indent=2)
            
            st.success("✅ Tests, Docs, and Reviews complete!")
            progress_bar.progress(99)
        
        progress_bar.progress(100)
        status_text.text("✅ Project generation complete!")
        
        st.success(f"🎉 **Successfully generated {project_name}!**")
        
        output_dir = Path("output") / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for filepath, content in all_files.items():
            file_path = output_dir / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
        
        zip_path = output_dir.parent / f"{project_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath, content in all_files.items():
                zipf.writestr(filepath, content)
        
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="📥 Download Complete Project (ZIP)",
                data=f.read(),
                file_name=f"{project_name}.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
    
    except Exception as e:
        st.error(f"❌ Generation failed: {str(e)}")
        st.code(traceback.format_exc())
    
    finally:
        if close_loop:
            loop.close()
