"""
Database Agent - SQLAlchemy Model and Migration Generation (Corrected)

This agent specializes in:
- Generating SQLAlchemy ORM models
- Creating database relationships (one-to-many, many-to-many)
- Generating Alembic migration scripts
- Creating seed data scripts
- Database configuration and initialization

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
from typing import Dict, Any, List, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager
import json


class DatabaseAgent(BaseAgent):
    """
    Agent specialized in database-related code generation.
    
    Capabilities:
    - SQLAlchemy model generation with relationships
    - Alembic migration script generation
    - Database seeding scripts
    - Database configuration files
    - Query helper functions
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize the database agent.
        
        Args:
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
        """
        super().__init__(
            agent_type="database",
            name="database_agent",
            role="Database Schema Expert and ORM Specialist",
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        logger.info(f"{self.name} initialized for database tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a database-related task.
        
        Args:
            task: Task dictionary with:
                - task_type: Type of task (generate_model, generate_config, etc.)
                - data: Task-specific data
        
        Returns:
            Result dictionary
        """
        task_type = task.get("task_type", "")
        data = task.get("data", {})
        
        task_map = {
            "generate_model": self.generate_sqlalchemy_model,
            "generate_migration": self.generate_alembic_migration,
            "generate_seed": self.generate_seed_data,
            "generate_config": self.generate_database_config,
            "generate_queries": self.generate_query_helpers,
            "generate_repository": self.generate_repository_pattern
        }
        
        if task_type in task_map:
            return await task_map[task_type](data)
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    async def generate_sqlalchemy_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a SQLAlchemy ORM model.
        
        Args:
            data: Dictionary with:
                - model_name: Name of the model (e.g., "User")
                - table_name: Database table name (e.g., "users")
                - fields: List of field definitions
                - relationships: List of relationships to other models
                - indexes: Optional index definitions
                - constraints: Optional constraint definitions
        
        Returns:
            Dictionary with generated model code
        """
        try:
            model_name = data.get("model_name", "")
            table_name = data.get("table_name", model_name.lower() + "s")
            fields = data.get("fields", [])
            relationships = data.get("relationships", [])
            indexes = data.get("indexes", [])
            constraints = data.get("constraints", [])
            
            if not model_name or not fields:
                return {
                    "status": "error",
                    "message": "Missing model_name or fields"
                }
            
            # Build prompt
            prompt = self.get_prompt(
                category="database_prompts", # Assuming a category name
                prompt_name="generate_sqlalchemy_model",
                variables={
                    "model_name": model_name,
                    "table_name": table_name,
                    "fields": json.dumps(fields, indent=2),
                    "relationships": json.dumps(relationships, indent=2),
                    "indexes": json.dumps(indexes, indent=2),
                    "constraints": json.dumps(constraints, indent=2),
                    "requirements": self._build_model_requirements(
                        model_name, table_name, fields, relationships, indexes, constraints
                    )
                }
            )
            
            logger.info(f"{self.name} generating SQLAlchemy model: {model_name}")
            
            code = await self.generate_response(
                prompt=prompt,
                agent_type="coding" # Use precise temperature
            )
            
            clean_code = self._clean_code(code)
            
            if not self._validate_sqlalchemy_model(clean_code, model_name):
                logger.warning(f"Generated model for {model_name} may have issues")
            
            return {
                "status": "success",
                "task": "generate_sqlalchemy_model",
                "model_name": model_name,
                "table_name": table_name,
                "code": clean_code,
                "attempts": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating SQLAlchemy model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _build_model_requirements(
        self,
        model_name: str,
        table_name: str,
        fields: List[Dict],
        relationships: List[Dict],
        indexes: List[Dict],
        constraints: List[Dict]
    ) -> str:
        """Build detailed requirements for model generation."""
        req = f"""
Generate a production-ready SQLAlchemy ORM model (using DeclarativeBase) with the following specifications:

Model Name: {model_name}
Table Name: {table_name}

FIELDS:
"""
        for field in fields:
            field_name = field.get("name", "")
            field_type = field.get("type", "String")
            nullable = field.get("nullable", True)
            unique = field.get("unique", False)
            default = field.get("default", None)
            
            req += f"\n- {field_name}: {field_type}"
            if not nullable:
                req += " (NOT NULL)"
            if unique:
                req += " (UNIQUE)"
            if default is not None:
                req += f" (DEFAULT: {default})"
        
        if relationships:
            req += "\n\nRELATIONSHIPS:"
            for rel in relationships:
                rel_name = rel.get("name", "")
                rel_type = rel.get("type", "many-to-one")
                target_model = rel.get("target_model", "")
                back_populates = rel.get("back_populates", "")
                
                req += f"\n- {rel_name}: {rel_type} with {target_model}"
                if back_populates:
                    req += f" (back_populates: {back_populates})"
        
        if indexes:
            req += "\n\nINDEXES:"
            for idx in indexes:
                idx_name = idx.get("name", "")
                idx_columns = idx.get("columns", [])
                req += f"\n- {idx_name}: columns {', '.join(idx_columns)}"
        
        if constraints:
            req += "\n\nCONSTRAINTS:"
            for const in constraints:
                const_type = const.get("type", "")
                const_details = const.get("details", "")
                req += f"\n- {const_type}: {const_details}"
        
        req += """

REQUIREMENTS:
- Use SQLAlchemy 2.0 declarative syntax (e.g., from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column).
- Import all types from `sqlalchemy` (e.g., String, Integer, DateTime, ForeignKey).
- Include proper type hints for all columns (e.g., `name: Mapped[str] = mapped_column(String(100))`).
- Add comprehensive docstrings for the class and fields.
- Include a `__repr__` method.
- Include `created_at` and `updated_at` timestamp fields with server defaults.
- Follow PEP 8 style guidelines.
"""
        return req
    
    def _validate_sqlalchemy_model(self, code: str, model_name: str) -> bool:
        """Validate that generated code is a proper SQLAlchemy model."""
        required_patterns = [
            r"class\s+" + model_name,
            r"__tablename__\s*=",
            r"mapped_column\(", # Check for SQLAlchemy 2.0 syntax
            r"Mapped\[", # Check for SQLAlchemy 2.0 type hints
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, code):
                logger.warning(f"Model validation failed: missing {pattern}")
                return False
        
        return True
    
    async def generate_alembic_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an Alembic migration script."""
        try:
            migration_name = data.get("migration_name", "")
            description = data.get("description", "")
            upgrade_ops = data.get("upgrade_operations", [])
            downgrade_ops = data.get("downgrade_operations", [])
            
            if not migration_name:
                return {"status": "error", "message": "Missing migration_name"}
            
            prompt = self.get_prompt(
                category="database_prompts",
                prompt_name="generate_alembic_migration",
                variables={
                    "migration_name": migration_name,
                    "description": description,
                    "requirements": self._build_migration_requirements(
                        migration_name, description, upgrade_ops, downgrade_ops
                    )
                }
            )
            
            logger.info(f"{self.name} generating migration: {migration_name}")
            
            code = await self.generate_response(prompt=prompt, agent_type="coding")
            
            return {
                "status": "success",
                "task": "generate_alembic_migration",
                "migration_name": migration_name,
                "code": self._clean_code(code),
                "attempts": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating migration: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_migration_requirements(
        self,
        migration_name: str,
        description: str,
        upgrade_ops: List[str],
        downgrade_ops: List[str]
    ) -> str:
        """Build requirements for migration generation."""
        req = f"""
Generate an Alembic migration script with the following specifications:

Migration Name: {migration_name}
Description: {description}

UPGRADE OPERATIONS:
"""
        for op in upgrade_ops:
            req += f"\n- {op}"
        
        req += "\n\nDOWNGRADE OPERATIONS:"
        for op in downgrade_ops:
            req += f"\n- {op}"
        
        req += """

REQUIREMENTS:
- Use Alembic operations (op.create_table, op.add_column, etc.)
- Include revision ID and down_revision (use placeholders like 'abc123def456' and '123abc456def')
- Add comprehensive comments.
- Ensure operations are reversible.
- Import alembic as `op` and sqlalchemy as `sa`.
"""
        return req
    
    async def generate_seed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate database seed data script."""
        try:
            model_name = data.get("model_name", "")
            num_records = data.get("num_records", 10)
            field_generators = data.get("field_generators", {})
            
            if not model_name:
                return {"status": "error", "message": "Missing model_name"}
            
            prompt = self.get_prompt(
                category="database_prompts",
                prompt_name="generate_seed_data",
                variables={
                    "module_name": f"seed_{model_name.lower()}",
                    "model_name": model_name,
                    "num_records": num_records,
                    "requirements": self._build_seed_requirements(
                        model_name, num_records, field_generators
                    )
                }
            )
            
            logger.info(f"{self.name} generating seed data for: {model_name}")
            
            code = await self.generate_response(prompt=prompt, agent_type="coding")
            
            return {
                "status": "success",
                "task": "generate_seed_data",
                "model_name": model_name,
                "num_records": num_records,
                "code": self._clean_code(code),
                "attempts": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating seed data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_seed_requirements(
        self,
        model_name: str,
        num_records: int,
        field_generators: Dict[str, str]
    ) -> str:
        """Build requirements for seed data generation."""
        req = f"""
Generate a database seeding script with the following specifications:

Model: {model_name}
Number of Records: {num_records}

FIELD GENERATION STRATEGIES (use Faker):
"""
        for field_name, strategy in field_generators.items():
            req += f"\n- {field_name}: {strategy}"
        
        req += """

REQUIREMENTS:
- Use Faker library for realistic test data.
- Handle database session properly (assume a `db_session` fixture or function).
- Include error handling and rollback.
- Generate diverse and realistic data.
- Check for existing data before seeding to make it idempotent.
- Add progress logging.
- Include a main function `seed_data(session)` that performs the seeding.
"""
        return req
    
    async def generate_database_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate database configuration and initialization code."""
        try:
            db_type = data.get("database_type", "postgresql")
            config_type = data.get("config_type", "development")
            features = data.get("features", [])
            
            prompt = self.get_prompt(
                category="database_prompts",
                prompt_name="generate_database_config",
                variables={
                    "module_name": "database_config",
                    "requirements": self._build_config_requirements(
                        db_type, config_type, features
                    )
                }
            )
            
            logger.info(f"{self.name} generating database config: {db_type}")
            
            code = await self.generate_response(prompt=prompt, agent_type="coding")
            
            return {
                "status": "success",
                "task": "generate_database_config",
                "database_type": db_type,
                "config_type": config_type,
                "code": self._clean_code(code),
                "attempts": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating database config: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_config_requirements(
        self,
        db_type: str,
        config_type: str,
        features: List[str]
    ) -> str:
        """Build requirements for database configuration."""
        req = f"""
Generate database configuration code with the following specifications:

Database Type: {db_type}
Configuration Type: {config_type}
FEATURES TO INCLUDE: {', '.join(features)}

REQUIREMENTS:
- Create SQLAlchemy 2.0 async engine and session factory (`create_async_engine`, `async_sessionmaker`).
- Use environment variables for configuration (e.g., `DATABASE_URL`).
- Include a `DeclarativeBase` for models to inherit from.
- Add a connection pooling configuration.
- Include an async function `init_db()` to create all tables.
- Add a health check function `check_db_connection()`.
- Include a session context manager `get_db_session()` (async generator).
"""
        return req
    
    async def generate_query_helpers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate query helper functions for a model."""
        try:
            model_name = data.get("model_name", "")
            query_types = data.get("query_types", ["CRUD"])
            
            if not model_name:
                return {"status": "error", "message": "Missing model_name"}
            
            prompt = self.get_prompt(
                category="database_prompts",
                prompt_name="generate_query_helpers",
                variables={
                    "module_name": f"{model_name.lower()}_queries",
                    "model_name": model_name,
                    "requirements": self._build_query_requirements(
                        model_name, query_types
                    )
                }
            )
            
            logger.info(f"{self.name} generating query helpers for: {model_name}")
            
            code = await self.generate_response(prompt=prompt, agent_type="coding")
            
            return {
                "status": "success",
                "task": "generate_query_helpers",
                "model_name": model_name,
                "code": self._clean_code(code),
                "attempts": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating query helpers: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_query_requirements(
        self,
        model_name: str,
        query_types: List[str]
    ) -> str:
        """Build requirements for query helper generation."""
        req = f"""
Generate query helper functions for the {model_name} model.

QUERY TYPES TO IMPLEMENT: {', '.join(query_types)}

REQUIREMENTS:
- Implement async CRUD operations (Create, Read, Update, Delete) using SQLAlchemy 2.0.
- All functions should accept an `AsyncSession` as the first argument.
- Use `select()` statements and `session.execute()`.
- Include pagination support for list operations (limit, offset).
- Include filtering and search functions.
- Add sorting capabilities.
- Handle edge cases (not found, duplicates, etc.).
- Use Pydantic schemas for create/update data (e.g., {model_name}CreateSchema).
"""
        return req
    
    async def generate_repository_pattern(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a repository pattern implementation for a model."""
        try:
            model_name = data.get("model_name", "")
            include_async = data.get("include_async", True)
            
            if not model_name:
                return {"status": "error", "message": "Missing model_name"}
            
            prompt = self.get_prompt(
                category="database_prompts",
                prompt_name="generate_repository_pattern",
                variables={
                    "module_name": f"{model_name.lower()}_repository",
                    "model_name": model_name,
                    "requirements": f"""
Generate a repository pattern implementation for {model_name} with:

- BaseRepository class with common operations.
- {model_name}Repository extending BaseRepository.
- Async CRUD operations (create, read, update, delete) using SQLAlchemy 2.0.
- All methods must accept an `AsyncSession`.
- Type hints for all functions and arguments.
- Comprehensive docstrings.
"""
                }
            )
            
            logger.info(f"{self.name} generating repository for: {model_name}")
            
            code = await self.generate_response(prompt=prompt, agent_type="coding")
            
            return {
                "status": "success",
                "task": "generate_repository_pattern",
                "model_name": model_name,
                "code": self._clean_code(code),
                "attempts": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating repository: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _clean_code(self, code: str) -> str:
        """Clean generated code."""
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()