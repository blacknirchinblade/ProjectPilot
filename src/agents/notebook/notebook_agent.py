"""
Notebook Agent

Performs Exploratory Data Analysis (EDA) on a given dataset and generates
a Jupyter Notebook (.ipynb) with the analysis, visualizations, and insights.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import pandas as pd
from typing import Dict, Any
from loguru import logger
import json

from src.agents.base_agent import BaseAgent

class NotebookAgent(BaseAgent):
    """
    Generates a Jupyter Notebook for Exploratory Data Analysis.
    """

    def __init__(self, llm_client=None, prompt_manager=None):
        super().__init__(
            name="notebook_agent",
            role="Expert Data Analyst",
            agent_type="notebook",  # Added missing agent_type
            llm_client=llm_client,
            prompt_manager=prompt_manager,
        )

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a task for the NotebookAgent.
        """
        logger.info(f"Executing task for NotebookAgent with data: {task_data}")
        # This is a placeholder implementation.
        # In a real scenario, you would have different task types
        # like 'generate_notebook', 'refine_notebook', etc.
        if task_data.get("task_type") == "generate_notebook":
            notebook_str = await self.generate_notebook(
                task_data.get("data_path"),
                task_data.get("analysis_goals")
            )
            return {"status": "success", "notebook": notebook_str}
        else:
            return {"status": "error", "message": "Unknown task type"}

    async def generate_notebook(self, data_path: str, analysis_goals: str) -> str:
        """
        Generates an EDA notebook.

        Args:
            data_path: The path to the dataset (e.g., a CSV file).
            analysis_goals: The user's goals for the analysis.

        Returns:
            The content of the generated .ipynb file as a string.
        """
        logger.info(f"Generating EDA notebook for dataset at {data_path} with goals: {analysis_goals}")

        # In a real implementation, this would be a sophisticated, multi-step
        # process involving LLM calls to generate code, markdown, and visualizations.
        # For now, we use a template.

        # Read the dataset to get column names
        try:
            df = pd.read_csv(data_path)
            columns = df.columns.tolist()
        except Exception as e:
            logger.error(f"Could not read the dataset: {e}")
            columns = ["column_1", "column_2"]

        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [
                        "# Exploratory Data Analysis\n\n",
                        f"This notebook performs an EDA based on the following goals: {analysis_goals}"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "import pandas as pd\n",
                        "import seaborn as sns\n",
                        "import matplotlib.pyplot as plt\n\n",
                        f"df = pd.read_csv('{data_path}')\n",
                        "df.head()"
                    ],
                    "outputs": []
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## Data Overview\n",
                        "Let's start by getting a basic overview of the data."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "print('Data Shape:', df.shape)\n",
                        "print('\\nInfo:')\n",
                        "df.info()\n",
                        "print('\\nMissing Values:')\n",
                        "print(df.isnull().sum())"
                    ],
                    "outputs": []
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## Visualizations\n",
                        "Now, let's create some visualizations to understand the data distribution."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "sns.pairplot(df)\n",
                        "plt.show()"
                    ],
                    "outputs": []
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 2
        }

        logger.success("Successfully generated EDA notebook content.")
        return json.dumps(notebook_content, indent=2)
