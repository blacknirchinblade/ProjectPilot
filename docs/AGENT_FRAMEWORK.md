# Agent Framework

The power of AutoCoder lies in its specialized agents. Each agent is an expert in a specific domain, allowing for a high degree of accuracy and quality in the generated code.

## BaseAgent

All agents inherit from a `BaseAgent` class, which provides common functionality:
-   A unique name and role.
-   Integration with the LLM client and prompt manager.
-   An `execute_task` method, which is the standard entry point for all agents.

## Core Agents

1.  **ClarificationAgent**
    -   **Role**: Expert Analyst
    -   **Responsibility**: Analyzes the user's initial project description and generates intelligent, clarifying questions to ensure the final project meets the user's exact needs.

2.  **DynamicArchitectAgent**
    -   **Role**: Expert Software Architect
    -   **Responsibility**: Designs the entire project structure, including directories, files, dependencies, and the relationships between them. It does this dynamically based on the user's request, without relying on templates.

3.  **CodingAgent**
    -   **Role**: Expert Programmer
    -   **Responsibility**: Writes the code for individual files based on the specifications provided by the ArchitectAgent. It can write code in multiple languages and is an expert in applying best practices.

4.  **TestingAgent**
    -   **Role**: QA Engineer
    -   **Responsibility**: Generates unit tests, integration tests, and other quality assurance checks for the code written by the CodingAgent.

5.  **ReviewOrchestrator**
    -   **Role**: Lead Code Reviewer
    -   **Responsibility**: Manages a team of specialized sub-reviewers (e.g., for readability, performance, security) to perform a comprehensive review of the generated code and provide a quality score.

## Agent Communication

Agents do not communicate directly with each other. Instead, they are coordinated by the central **Orchestrator**. This ensures a clear and predictable workflow:
1.  The Orchestrator assigns a task to an agent.
2.  The agent executes the task and returns the result to the Orchestrator.
3.  The Orchestrator then uses this result to inform the next task for the next agent.

This centralized communication model prevents complex and unpredictable agent interactions, making the system easier to debug and maintain.
