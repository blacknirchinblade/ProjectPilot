# Projectify: Your Autonomous AI/ML Project Generation Copilot (In Development)

**Projectify is an advanced multi-agent system that automates the entire software development lifecycle for AI/ML projects.** Describe your idea in natural language, and watch as a team of specialized AI agents collaborates to deliver a complete, production-ready codebase.

**Note:** This project is currently under active development. While many core features are functional, some agents are still being built and refined.

### Key Features
- **✨ Natural Language to Code:** Go from a simple prompt to a full-stack application.
- **🤖 Multi-Agent Collaboration:** Specialized agents for architecture, coding, testing, and review work together to ensure high-quality output.
- **🚀 Full-Stack Generation:** Creates ML models, FastAPI backends, Streamlit UIs, and more.
- **✅ Production-Ready Components:** Generates Docker files, CI/CD pipelines, unit tests, and documentation.
- **🧠 Interactive & Iterative:** The system asks clarifying questions and iteratively refines code based on automated reviews.

---

## 🤖 Agent Status

This table shows the current development status of our specialized AI agents.

| Agent | Status | Description |
| :--- | :--- | :--- |
| **Core Agents** | | |
| Clarification Agent | ✅ **Functional** | Asks intelligent questions to refine project requirements. |
| Dynamic Architect | ✅ **Functional** | Designs the project structure and file specifications. |
| Coding Agent | ✅ **Functional** | Generates code based on architectural specifications. |
| Review Orchestrator | ✅ **Functional** | Manages a team of reviewer agents to score code quality. |
| Advanced Doc Agent | ✅ **Functional** | Generates comprehensive project documentation (`README`, `WORKFLOW`, etc.). |
| Complex Interview Agent| ✅ **Functional** | Creates interview questions based on the generated project. |
| **Specialized Agents** | | |
| Database Agent | ✅ **Functional** | Generates database schemas and SQLAlchemy models. |
| API Agent | ✅ **Functional** | Creates REST API endpoints for FastAPI and Flask. |
| Streamlit Agent | 🛠️ **Under Development** | Focuses on generating complex, multi-page Streamlit UIs. |
| Deployment Agent | 🛠️ **Under Development** | Generates Dockerfiles, `docker-compose.yml`, and deployment scripts. |
| **Reviewer Agents** | | |
| Readability Reviewer | ✅ **Functional** | Checks code for style, formatting, and readability. |
| Logic Flow Reviewer | ✅ **Functional** | Analyzes the logical consistency of the code. |
| Security Reviewer | 🛠️ **Under Development** | Scans for common security vulnerabilities (e.g., SQL injection, hardcoded secrets). |
| Performance Reviewer | 💡 **Planned** | Analyzes code for performance bottlenecks and suggests optimizations. |
| **Interactive Agents** | | |
| Conversation Manager | ✅ **Functional** | Manages the state for the post-generation project chat. |
| Interactive Mod Agent | 💡 **Planned** | Applies user-requested changes to the codebase via chat. |

---

### 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Projectify.git
    cd Projectify
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    -   Create a `.env` file and add your `GOOGLE_API_KEY`.
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```

### 🛣️ Roadmap

Our immediate focus is on completing the "Under Development" agents to provide a more robust end-to-end experience. Key upcoming features include:
-   **Full Deployment Automation:** Finalizing the `DeploymentAgent` for one-click deployments.
-   **Enhanced Security Scanning:** Completing the `SecurityReviewer` to catch critical vulnerabilities.
-   **Interactive Code Modifications:** Implementing the `InteractiveModificationAgent` to allow users to refactor and enhance their projects via chat.

### 🙌 Contributing

We welcome contributions! If you're interested in helping build the future of automated code generation, please check out our contributing guidelines (coming soon) or open an issue to discuss your ideas.