"""
Specialized Agents Module

This module contains specialized agents for specific development tasks:
- DatabaseAgent: SQLAlchemy models, migrations, seed data
- APIAgent: REST API endpoints, schemas, authentication
- FrontendAgent: React/Vue components (coming soon)
- ConfigAgent: Configuration management (coming soon)
- MigrationAgent: Database migrations (coming soon)

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from .database_agent import DatabaseAgent
from .api_agent import APIAgent

__all__ = [
    "DatabaseAgent",
    "APIAgent",
]
