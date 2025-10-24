# System Architecture

AutoCoder is designed with a modular, multi-agent architecture. Each component is responsible for a specific part of the software development lifecycle, allowing for a clear separation of concerns and making the system extensible.

## Core Components

1.  **Streamlit Web UI (`streamlit_app.py`)**
    -   The main entry point for users.
    -   Provides a user-friendly interface for describing projects, answering questions, and viewing results.
    -   Manages the overall state of the application.

2.  **Orchestrator (`orchestrator.py`)**
    -   The "brain" of the system.
    -   Takes the user's request and breaks it down into a series of tasks.
    -   Delegates tasks to the appropriate agents in the correct sequence.

3.  **Agent Framework (`src/agents/`)**
    -   A collection of specialized AI agents, each inheriting from a `BaseAgent`.
    -   Each agent is an expert in a specific domain (e.g., architecture, coding, testing).
    -   Agents communicate and collaborate through the orchestrator.

4.  **LLM Integration (`src/llm/`)**
    -   Provides a consistent interface for communicating with the underlying Large Language Model (e.g., Google's Gemini).
    -   Manages API calls, prompt formatting, and response parsing.

5.  **Automation Tools (`src/tools/`)**
    -   A suite of non-AI helper utilities for performing concrete tasks like creating files, installing dependencies, and running tests.
    -   Agents use these tools to interact with the file system and the user's environment.

## Data Flow

1.  The user provides a project description in the **Streamlit UI**.
2.  The UI sends the request to the **Orchestrator**.
3.  The Orchestrator invokes the **ClarificationAgent** to generate clarifying questions.
4.  The user's answers are sent back to the Orchestrator.
5.  The Orchestrator tasks the **DynamicArchitectAgent** to design the project structure.
6.  The Orchestrator then iterates through the designed structure, tasking the **CodingAgent** to write the code for each file.
7.  As code is generated, the **TestingAgent** and **ReviewOrchestrator** are invoked to ensure quality.
8.  This process continues iteratively until the project meets the required quality standards.
