# ProjectPilot Agent Workflow

## End-to-End Workflow

1. **ClarificationAgent**: Asks dynamic, LLM-driven questions to clarify user requirements and resolve ambiguities.
2. **PromptEngineerAgent**: Transforms clarified requirements into detailed, actionable specifications for all downstream agents.
3. **PlanningAgent**: Analyzes requirements, breaks down the project into tasks, and designs the system architecture.
4. **CodingAgent / GenAIAgent**: Generates code for core logic, ML pipelines, and GenAI features.
5. **DatabaseAgent / APIAgent / StreamlitAgent**: Builds backend, APIs, and dashboards as specified.
6. **DeploymentAgent**: Creates deployment configurations (Docker, Kubernetes, CI/CD, cloud, monitoring).
7. **TestingAgent / DocumentationAgent / ReviewAgent / RefactoringAgent**: Ensures code quality, generates documentation, reviews, and refactors as needed.

## Agent Directory
- All agents are in `src/agents/`.
- Each agent is modular and can be extended or replaced.

## Customizing the Workflow
- Add new agents by creating a new Python file in `src/agents/` and integrating it into the main workflow.
- Update the Streamlit UI (`streamlit_app.py`) to include new agent steps if needed.
- Modify prompts and templates in `config/` for new behaviors.

---
*See also: `README.md` for project overview and `DATA_GUIDE.md` for dataset usage.*
