# Per-Project Agent Workflow

This document explains how ProjectPilot dynamically orchestrates agents for every new project you generate. Each project request triggers a fresh, context-aware pipeline:

## 1. ClarificationAgent
- Analyzes your project description and selected options.
- Asks dynamic, LLM-driven questions to resolve ambiguities and gather missing details.
- Collects your answers for precise requirements.

## 2. PromptEngineerAgent
- Transforms clarified requirements into a detailed, actionable specification.
- Ensures all downstream agents have the context they need.

## 3. PlanningAgent
- Breaks down the specification into tasks and components.
- Designs the system architecture and workflow.

## 4. CodingAgent / GenAIAgent
- Generates core code, ML pipelines, and GenAI features as required.
- Handles logic, data processing, and model integration.

## 5. DatabaseAgent / APIAgent / StreamlitAgent
- Builds database models, API endpoints, and Streamlit dashboards based on your selections.

## 6. DeploymentAgent
- Creates deployment configurations: Docker, Kubernetes, CI/CD, cloud, and monitoring setups.

## 7. TestingAgent / DocumentationAgent / ReviewAgent / RefactoringAgent
- Generates unit tests, documentation, and reviews/refactors code for quality and maintainability.

---

**Every project you generate runs through this pipeline, ensuring a complete, production-ready result tailored to your inputs.**

- Agents are instantiated fresh for each project, using your specific requirements.
- The pipeline is modular: you can extend, skip, or customize steps as needed.

For more details, see `WORKFLOW.md` or the main `README.md`.