"""
API Agent - Specialized agent for generating REST API code.

This agent generates FastAPI and Flask REST APIs with:
- RESTful route handlers
- Pydantic schemas for request/response validation
- Authentication and authorization middleware
- API documentation (OpenAPI/Swagger)
- Error handling and logging
- CORS configuration
- Database integration
- API testing code

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from src.agents.base_agent import BaseAgent


class APIAgent(BaseAgent):
    """
    Specialized agent for generating REST API code.
    
    Capabilities:
    - Generate FastAPI applications
    - Generate Flask applications
    - Create Pydantic schemas
    - Generate route handlers (CRUD operations)
    - Create authentication middleware
    - Generate OpenAPI documentation
    - Create API tests
    """
    
    def __init__(self):
        """Initialize the API Agent."""
        super().__init__(
            name="api_agent",
            role="Expert Backend API Developer",
            agent_type="coding"  # Use coding type for code generation
        )
        self.system_context = (
            "You are an expert in building RESTful APIs with FastAPI and Flask. "
            "You specialize in creating production-ready API endpoints with proper "
            "validation, error handling, authentication, and documentation. "
            "You follow REST best practices and API design principles."
        )
        logger.info("api_agent initialized and ready for API generation tasks")
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API generation task.
        
        Args:
            task_data: Task configuration with 'task_type' and parameters
            
        Returns:
            Dictionary with generation results
        """
        task_type = task_data.get("task_type")
        
        logger.info(f"api_agent executing task: {task_type}")
        
        if task_type == "generate_fastapi_app":
            return await self.generate_fastapi_app(task_data)
        elif task_type == "generate_flask_app":
            return await self.generate_flask_app(task_data)
        elif task_type == "generate_routes":
            return await self.generate_routes(task_data)
        elif task_type == "generate_schemas":
            return await self.generate_schemas(task_data)
        elif task_type == "generate_auth":
            return await self.generate_auth(task_data)
        elif task_type == "generate_middleware":
            return await self.generate_middleware(task_data)
        elif task_type == "generate_api_tests":
            return await self.generate_api_tests(task_data)
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    async def generate_fastapi_app(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete FastAPI application.
        
        Args:
            data: Configuration with app_name, description, endpoints, auth_type, database
            
        Returns:
            Dictionary with generated code and metadata
        """
        app_name = data.get("app_name", "MyAPI")
        description = data.get("description", "FastAPI application")
        endpoints = data.get("endpoints", [])
        auth_type = data.get("auth_type", "none")  # none, jwt, oauth2, api_key
        database = data.get("database", "sqlite")
        include_cors = data.get("include_cors", True)
        
        logger.info(f"Generating FastAPI app: {app_name}")
        
        # Build requirements
        requirements = self._build_fastapi_requirements(auth_type, database, include_cors)
        
        prompt = f"""Generate a complete FastAPI application with the following specifications:

**Application Details:**
- Name: {app_name}
- Description: {description}
- Authentication: {auth_type}
- Database: {database}
- CORS Enabled: {include_cors}

**Endpoints to Generate:**
{self._format_endpoints(endpoints)}

**Requirements:**
{requirements}

**Generate a production-ready FastAPI application that includes:**

1. **Main Application Setup:**
   - FastAPI app initialization with metadata
   - CORS middleware configuration (if enabled)
   - Database connection setup
   - Logging configuration with loguru
   - Error handlers for common HTTP exceptions

2. **Router Structure:**
   - Separate routers for different resource types
   - Proper route definitions with HTTP methods
   - Request/response models using Pydantic
   - Path and query parameter validation

3. **Authentication (if specified):**
   - JWT token generation and validation
   - OAuth2 password flow (if oauth2)
   - API key validation (if api_key)
   - Protected route decorators

4. **Database Integration:**
   - SQLAlchemy models
   - Async database session management
   - CRUD operations
   - Connection pooling

5. **Documentation:**
   - OpenAPI schema with detailed descriptions
   - Example requests and responses
   - API versioning support

6. **Error Handling:**
   - Custom exception handlers
   - Proper HTTP status codes
   - Detailed error messages
   - Validation error formatting

7. **Code Quality:**
   - Type hints for all functions
   - Docstrings following Google style
   - Input validation
   - Proper logging

**Generate the main.py file with all components integrated.**

IMPORTANT: Return ONLY the Python code, no explanations or markdown."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.4
            )
            
            code = response.strip()
            
            # Clean code
            code = self._clean_code(code)
            
            # Validate FastAPI code
            if not self._validate_fastapi_code(code):
                return {
                    "status": "error",
                    "message": "Generated code failed validation"
                }
            
            logger.success(f"✓ Generated FastAPI app: {app_name}")
            
            return {
                "status": "success",
                "code": code,
                "file_name": "main.py",
                "requirements": requirements,
                "app_name": app_name,
                "framework": "FastAPI",
                "lines": len(code.split('\n')),
                "characters": len(code)
            }
            
        except Exception as e:
            logger.error(f"Error generating FastAPI app: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_flask_app(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete Flask application.
        
        Args:
            data: Configuration with app_name, description, endpoints, auth_type
            
        Returns:
            Dictionary with generated code and metadata
        """
        app_name = data.get("app_name", "MyFlaskAPI")
        description = data.get("description", "Flask API application")
        endpoints = data.get("endpoints", [])
        auth_type = data.get("auth_type", "none")
        
        logger.info(f"Generating Flask app: {app_name}")
        
        requirements = self._build_flask_requirements(auth_type)
        
        prompt = f"""Generate a complete Flask REST API application with:

**Application Details:**
- Name: {app_name}
- Description: {description}
- Authentication: {auth_type}

**Endpoints:**
{self._format_endpoints(endpoints)}

**Requirements:**
{requirements}

**Generate a production-ready Flask application that includes:**

1. Application factory pattern
2. Blueprint organization
3. Request validation with marshmallow
4. Authentication (if specified)
5. Error handling
6. CORS support
7. Database integration with Flask-SQLAlchemy
8. Logging configuration
9. Configuration management

IMPORTANT: Return ONLY the Python code for app.py, no explanations."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.4
            )
            
            code = response.strip()
            code = self._clean_code(code)
            
            logger.success(f"✓ Generated Flask app: {app_name}")
            
            return {
                "status": "success",
                "code": code,
                "file_name": "app.py",
                "requirements": requirements,
                "app_name": app_name,
                "framework": "Flask",
                "lines": len(code.split('\n')),
                "characters": len(code)
            }
            
        except Exception as e:
            logger.error(f"Error generating Flask app: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_routes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate route handlers for specific resources.
        
        Args:
            data: Configuration with resource_name, operations, framework
            
        Returns:
            Dictionary with generated route code
        """
        resource_name = data.get("resource_name", "Item")
        operations = data.get("operations", ["create", "read", "update", "delete"])
        framework = data.get("framework", "fastapi")
        
        logger.info(f"Generating {framework} routes for {resource_name}")
        
        prompt = f"""Generate RESTful route handlers for a {resource_name} resource using {framework.upper()}.

**Resource:** {resource_name}
**Operations:** {', '.join(operations)}
**Framework:** {framework}

**Generate routes for these operations:**
- CREATE: POST /{resource_name.lower()}s
- READ: GET /{resource_name.lower()}s/{{id}}
- LIST: GET /{resource_name.lower()}s
- UPDATE: PUT /{resource_name.lower()}s/{{id}}
- DELETE: DELETE /{resource_name.lower()}s/{{id}}

**Include:**
1. Proper HTTP methods and status codes
2. Request/response validation
3. Error handling
4. Query parameters for filtering/pagination (LIST)
5. Path parameters validation
6. Docstrings with OpenAPI descriptions

IMPORTANT: Return ONLY the Python code for routes, no explanations."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.4
            )
            
            code = response.strip()
            code = self._clean_code(code)
            
            logger.success(f"✓ Generated routes for {resource_name}")
            
            return {
                "status": "success",
                "code": code,
                "file_name": f"{resource_name.lower()}_routes.py",
                "resource": resource_name,
                "operations": operations,
                "framework": framework
            }
            
        except Exception as e:
            logger.error(f"Error generating routes: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_schemas(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Pydantic schemas for API request/response validation.
        
        Args:
            data: Configuration with model_name, fields, framework
            
        Returns:
            Dictionary with generated schema code
        """
        model_name = data.get("model_name", "Item")
        fields = data.get("fields", [])
        include_examples = data.get("include_examples", True)
        
        logger.info(f"Generating Pydantic schemas for {model_name}")
        
        prompt = f"""Generate Pydantic schemas for a {model_name} API resource.

**Model:** {model_name}
**Fields:** {self._format_fields(fields)}
**Include Examples:** {include_examples}

**Generate these Pydantic schemas:**

1. **Base Schema ({model_name}Base):**
   - Common fields shared across all schemas
   - Field validation rules
   - Example values (if enabled)

2. **Create Schema ({model_name}Create):**
   - Fields required for creation
   - No ID field
   - Additional validation

3. **Update Schema ({model_name}Update):**
   - All fields optional
   - Allows partial updates

4. **Response Schema ({model_name}Response):**
   - Includes ID and timestamps
   - Inherits from base
   - ORM mode enabled

5. **List Response Schema ({model_name}List):**
   - Pagination metadata
   - List of items

**Include:**
- Field validators
- Custom validation logic
- JSON schema examples
- Type hints
- Docstrings

IMPORTANT: Return ONLY the Python code, no explanations."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.3
            )
            
            code = response.strip()
            code = self._clean_code(code)
            
            logger.success(f"✓ Generated schemas for {model_name}")
            
            return {
                "status": "success",
                "code": code,
                "file_name": f"{model_name.lower()}_schemas.py",
                "model": model_name,
                "fields": fields
            }
            
        except Exception as e:
            logger.error(f"Error generating schemas: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_auth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate authentication and authorization code.
        
        Args:
            data: Configuration with auth_type, framework
            
        Returns:
            Dictionary with generated auth code
        """
        auth_type = data.get("auth_type", "jwt")
        framework = data.get("framework", "fastapi")
        
        logger.info(f"Generating {auth_type} authentication for {framework}")
        
        prompt = f"""Generate {auth_type.upper()} authentication implementation for {framework.upper()}.

**Authentication Type:** {auth_type}
**Framework:** {framework}

**Generate complete authentication system with:**

1. **Token Generation:**
   - Create access and refresh tokens
   - Token payload with user info
   - Expiration handling

2. **Token Validation:**
   - Verify token signature
   - Check expiration
   - Extract user data

3. **Password Hashing:**
   - Bcrypt/Argon2 hashing
   - Password verification

4. **Authentication Dependencies:**
   - Dependency injection for protected routes
   - Current user retrieval
   - Permission checking

5. **Login/Logout Endpoints:**
   - Login route with credentials
   - Token refresh route
   - Logout/revoke token

6. **Error Handling:**
   - Invalid credentials
   - Expired tokens
   - Unauthorized access

IMPORTANT: Return ONLY the Python code for auth.py, no explanations."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.3
            )
            
            code = response.strip()
            code = self._clean_code(code)
            
            logger.success(f"✓ Generated {auth_type} authentication")
            
            return {
                "status": "success",
                "code": code,
                "file_name": "auth.py",
                "auth_type": auth_type,
                "framework": framework
            }
            
        except Exception as e:
            logger.error(f"Error generating auth: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_middleware(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate custom middleware for API.
        
        Args:
            data: Configuration with middleware_type, framework
            
        Returns:
            Dictionary with generated middleware code
        """
        middleware_type = data.get("middleware_type", "logging")
        framework = data.get("framework", "fastapi")
        
        logger.info(f"Generating {middleware_type} middleware")
        
        prompt = f"""Generate {middleware_type} middleware for {framework.upper()}.

**Middleware Type:** {middleware_type}
**Framework:** {framework}

**Generate middleware that:**
- Intercepts requests and responses
- Performs the required processing
- Maintains performance
- Includes proper error handling

IMPORTANT: Return ONLY the Python code, no explanations."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.4
            )
            
            code = response.strip()
            code = self._clean_code(code)
            
            return {
                "status": "success",
                "code": code,
                "file_name": f"{middleware_type}_middleware.py",
                "middleware_type": middleware_type
            }
            
        except Exception as e:
            logger.error(f"Error generating middleware: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_api_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate API tests using pytest.
        
        Args:
            data: Configuration with endpoints, framework
            
        Returns:
            Dictionary with generated test code
        """
        endpoints = data.get("endpoints", [])
        framework = data.get("framework", "fastapi")
        test_framework = data.get("test_framework", "pytest")
        
        logger.info(f"Generating API tests for {len(endpoints)} endpoints")
        
        prompt = f"""Generate comprehensive API tests using {test_framework} for a {framework} application.

**Endpoints to Test:**
{self._format_endpoints(endpoints)}

**Generate test suite that includes:**

1. **Test Client Setup:**
   - Fixture for test client
   - Database fixture (if needed)
   - Authentication fixtures

2. **Test Cases for Each Endpoint:**
   - Success cases
   - Validation errors
   - Authentication errors
   - Not found errors
   - Edge cases

3. **Test Organization:**
   - Test classes grouped by resource
   - Descriptive test names
   - Clear assertions

4. **Mock Data:**
   - Fixtures for test data
   - Factory functions

5. **Coverage:**
   - All HTTP methods
   - All status codes
   - Request/response validation

IMPORTANT: Return ONLY the Python code for test_api.py, no explanations."""

        try:
            response = await self.generate_response(
                prompt=prompt,
                temperature=0.4
            )
            
            code = response.strip()
            code = self._clean_code(code)
            
            logger.success("✓ Generated API tests")
            
            return {
                "status": "success",
                "code": code,
                "file_name": "test_api.py",
                "endpoints": endpoints,
                "test_framework": test_framework
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    # Helper methods
    
    def _build_fastapi_requirements(
        self, 
        auth_type: str, 
        database: str,
        include_cors: bool
    ) -> str:
        """Build requirements list for FastAPI."""
        reqs = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.4.0",
            "loguru>=0.7.0"
        ]
        
        if auth_type == "jwt":
            reqs.extend([
                "python-jose[cryptography]>=3.3.0",
                "passlib[bcrypt]>=1.7.4",
                "python-multipart>=0.0.6"
            ])
        elif auth_type == "oauth2":
            reqs.extend([
                "authlib>=1.2.0",
                "httpx>=0.25.0"
            ])
        
        if database == "postgresql":
            reqs.extend([
                "sqlalchemy>=2.0.0",
                "asyncpg>=0.29.0",
                "alembic>=1.12.0"
            ])
        elif database == "mysql":
            reqs.extend([
                "sqlalchemy>=2.0.0",
                "aiomysql>=0.2.0",
                "alembic>=1.12.0"
            ])
        else:  # sqlite
            reqs.extend([
                "sqlalchemy>=2.0.0",
                "aiosqlite>=0.19.0"
            ])
        
        if include_cors:
            reqs.append("python-cors>=1.0.0")
        
        return "\n".join(f"- {req}" for req in reqs)
    
    def _build_flask_requirements(self, auth_type: str) -> str:
        """Build requirements list for Flask."""
        reqs = [
            "flask>=3.0.0",
            "flask-restful>=0.3.10",
            "flask-sqlalchemy>=3.1.0",
            "flask-cors>=4.0.0",
            "marshmallow>=3.20.0",
            "loguru>=0.7.0"
        ]
        
        if auth_type == "jwt":
            reqs.extend([
                "flask-jwt-extended>=4.5.0",
                "bcrypt>=4.1.0"
            ])
        
        return "\n".join(f"- {req}" for req in reqs)
    
    def _format_endpoints(self, endpoints: List[Dict]) -> str:
        """Format endpoints for prompt."""
        if not endpoints:
            return "- Standard CRUD endpoints (Create, Read, Update, Delete)"
        
        lines = []
        for ep in endpoints:
            method = ep.get("method", "GET")
            path = ep.get("path", "/")
            desc = ep.get("description", "")
            lines.append(f"- {method} {path}: {desc}")
        
        return "\n".join(lines)
    
    def _format_fields(self, fields: List[Dict]) -> str:
        """Format fields for prompt."""
        if not fields:
            return "Standard fields (id, name, description, created_at, updated_at)"
        
        lines = []
        for field in fields:
            name = field.get("name", "")
            type_ = field.get("type", "str")
            required = field.get("required", True)
            req_str = "required" if required else "optional"
            lines.append(f"- {name}: {type_} ({req_str})")
        
        return "\n".join(lines)
    
    def _validate_fastapi_code(self, code: str) -> bool:
        """Validate generated FastAPI code."""
        # Check for required imports
        required = ["fastapi", "FastAPI"]
        return all(req in code for req in required)
    
    def _clean_code(self, code: str) -> str:
        """Clean generated code."""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
