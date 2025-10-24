"""
Deployment Agent - Generates Deployment Configurations

This agent creates deployment configurations for ML/DL projects including:
- Docker multi-container setups (docker-compose)
- Kubernetes manifests (deployments, services, ingress)
- CI/CD pipelines (GitHub Actions, GitLab CI)
- Cloud deployment configs (AWS, GCP, Azure)
- Model serving configurations (TorchServe, TensorFlow Serving)
- Monitoring setup (Prometheus, Grafana)
- Environment management

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
import traceback
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class DeploymentAgent(BaseAgent):
    """
    DeploymentAgent generates deployment configurations for ML/DL applications.
    
    Features:
    - Docker and Docker Compose configurations
    - Kubernetes manifests (deployment, service, ingress, configmap)
    - CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
    - Cloud deployment (AWS ECS, GCP Cloud Run, Azure Container Apps)
    - Model serving (TorchServe, TensorFlow Serving, TorchServe)
    - Monitoring and logging (Prometheus, Grafana, ELK)
    - Nginx reverse proxy configurations
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize the Deployment Agent.
        
        Args:
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        super().__init__(
            name="deployment_agent",
            role="Expert DevOps and MLOps Engineer",
            agent_type="coding", # Use 'coding' temp for precision
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        logger.info(f"{self.name} initialized for deployment tasks")
    

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a deployment task.
        
        Args:
            task: Task dictionary with type and data
                - task_type: "generate_deployment_config"
                - data: Task-specific parameters
        
        Returns:
            Dictionary with deployment files
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        # Ensure clients are available
        if not self.llm_client or not self.prompt_manager:
            logger.warning("Re-initializing LLM client for DeploymentAgent.")
            self.llm_client = GeminiClient()
            self.prompt_manager = PromptManager()

        try:
            if task_type == "generate_deployment_config":
                return await self.generate_deployment_config(
                    project_name=data.get("project_name"),
                    components=data.get("components", {}),
                    config=data.get("config", {})
                )
            else:
                logger.error(f"Unknown task type: {task_type}")
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
        except Exception as e:
            logger.error(f"Error executing deployment task '{task_type}': {e}\n{traceback.format_exc()}")
            return {"status": "error", "task": task_type, "message": str(e)}

    async def generate_deployment_config(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate complete deployment configuration.
        
        Args:
            project_name: Name of the project
            components: Dictionary of components to deploy
            config: Additional configuration
        
        Returns:
            Dictionary with generated files: {filename: code}
        """
        config = config or {}
        
        files = {}
        
        # Docker configurations
        files.update(self._generate_docker_files(project_name, components, config))
        
        # Kubernetes manifests (if enabled)
        if config.get("kubernetes", False):
            files.update(self._generate_kubernetes_manifests(project_name, components, config))
        
        # CI/CD pipelines (if enabled)
        if config.get("ci_cd", False):
            ci_platform = config.get("ci_platform", "github")
            files.update(self._generate_ci_cd_pipeline(project_name, components, ci_platform))
        
        # Cloud deployment configs (if enabled)
        cloud_provider = config.get("cloud_provider")
        if cloud_provider:
            files.update(self._generate_cloud_config(project_name, components, cloud_provider))
        
        # Monitoring setup (if enabled)
        if config.get("monitoring", False):
            files.update(self._generate_monitoring_config(project_name, components))
        
        # Nginx reverse proxy (if enabled)
        if config.get("nginx", False):
            files.update(self._generate_nginx_config(project_name, components))
        
        # Documentation
        files["deployment/README.md"] = self._generate_deployment_docs(
            project_name, components, config
        )
        
        return files
    

    def _generate_docker_files(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Docker and Docker Compose files."""
        
        files = {}
        
        # Individual Dockerfiles for each component
        if components.get("ml_model", False):
            files["ml_model/Dockerfile"] = self._create_ml_model_dockerfile(config)
        
        if components.get("api", False):
            files["api/Dockerfile"] = self._create_api_dockerfile(config)
        
        if components.get("frontend", False):
            files["frontend/Dockerfile"] = self._create_frontend_dockerfile(config)
        
        # Docker Compose - main orchestration file
        files["docker-compose.yml"] = self._create_docker_compose(
            project_name, components, config
        )
        
        # Docker Compose for production
        files["docker-compose.prod.yml"] = self._create_docker_compose_prod(
            project_name, components, config
        )
        
        # Environment files
        files[".env.example"] = self._create_env_example(components, config)
        files[".dockerignore"] = self._create_dockerignore()
        
        # Helper scripts
        files["scripts/build.sh"] = self._create_build_script()
        files["scripts/deploy.sh"] = self._create_deploy_script()
        
        return files
    
    def _generate_docker_files(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Docker and Docker Compose files."""
        
        files = {}
        
        # Individual Dockerfiles for each component
        if components.get("ml_model", False):
            files["ml_model/Dockerfile"] = self._create_ml_model_dockerfile(config)
        
        if components.get("api", False):
            files["api/Dockerfile"] = self._create_api_dockerfile(config)
        
        if components.get("frontend", False):
            files["frontend/Dockerfile"] = self._create_frontend_dockerfile(config)
        
        # Docker Compose - main orchestration file
        files["docker-compose.yml"] = self._create_docker_compose(
            project_name, components, config
        )
        
        # Docker Compose for production
        files["docker-compose.prod.yml"] = self._create_docker_compose_prod(
            project_name, components, config
        )
        
        # Environment files
        files[".env.example"] = self._create_env_example(components, config)
        files[".dockerignore"] = self._create_dockerignore()
        
        # Helper scripts
        files["scripts/build.sh"] = self._create_build_script()
        files["scripts/deploy.sh"] = self._create_deploy_script()
        
        return files
    
    def _create_ml_model_dockerfile(self, config: Dict) -> str:
        """Create Dockerfile for ML model service."""
        use_gpu = config.get("gpu", False)
        framework = config.get("framework", "pytorch")
        if use_gpu:
            base_image = "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
        else:
            base_image = "python:3.10-slim"
        dockerfile = (
            f'# Dockerfile for ML Model Service\n'
            f'FROM {base_image}\n\n'
            'WORKDIR /app\n\n'
            '# Install system dependencies\n'
            'RUN apt-get update && apt-get install -y \\\n'
            '    build-essential \\\n'
            '    curl \\\n'
            '    && rm -rf /var/lib/apt/lists/*\n\n'
            '# Install Python dependencies\n'
            'COPY requirements.txt .\n'
            'RUN pip install --no-cache-dir -r requirements.txt\n\n'
            '# Copy model code\n'
            'COPY models/ ./models/\n'
            'COPY inference/ ./inference/\n'
            'COPY training/ ./training/\n'
            'COPY data/ ./data/\n\n'
            '# Copy trained model (if available)\n'
            'COPY checkpoints/ ./checkpoints/ 2>/dev/null || true\n\n'
            '# Set environment variables\n'
            'ENV PYTHONUNBUFFERED=1\n'
            'ENV MODEL_PATH=/app/checkpoints/best_model.pth\n'
        )
        if use_gpu:
            dockerfile += 'ENV CUDA_VISIBLE_DEVICES=0\n'
        dockerfile += (
            '\n# Health check\n'
            'HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\\n'
            '    CMD curl -f http://localhost:5000/health || exit 1\n\n'
            '# Expose port\n'
            'EXPOSE 5000\n\n'
            '# Run model server\n'
            'CMD ["python", "inference/server.py"]\n'
        )
        return dockerfile
    
    def _create_api_dockerfile(self, config: Dict) -> str:
        """Create Dockerfile for API service."""
        
        return '''# Dockerfile for FastAPI Backend
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/ 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with Uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers $WORKERS
'''
    
    def _create_frontend_dockerfile(self, config: Dict) -> str:
        """Create Dockerfile for frontend service."""
        
        frontend_type = config.get("frontend_type", "streamlit")
        
        if frontend_type == "streamlit":
            return '''# Dockerfile for Streamlit Frontend
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY pages/ ./pages/ 2>/dev/null || true
COPY components/ ./components/ 2>/dev/null || true
COPY utils/ ./utils/ 2>/dev/null || true
COPY .streamlit/ ./.streamlit/ 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''
        else:
            return "# React/Vue Dockerfile\n# TODO: Implement"
    
    def _create_docker_compose(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Create docker-compose.yml for development."""
        
        services = []
        networks = ["ml_network"]
        volumes = []
        
        # ML Model Service
        if components.get("ml_model", False):
            gpu_config = '''
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]''' if config.get("gpu", False) else ""
            
            services.append(f'''  ml_model:
    build:
      context: ./ml_model
      dockerfile: Dockerfile
    container_name: {project_name}_ml_model
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/checkpoints/best_model.pth
      - LOG_LEVEL=INFO{gpu_config}
    networks:
      - ml_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
''')
        
        # Database Service
        if components.get("database", False):
            db_type = config.get("database_type", "postgresql")
            
            if db_type == "postgresql":
                services.append(f'''  database:
    image: postgres:15-alpine
    container_name: {project_name}_db
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=mluser
      - POSTGRES_PASSWORD=mlpassword
      - POSTGRES_DB={project_name}_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ml_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser"]
      interval: 10s
      timeout: 5s
      retries: 5
''')
                volumes.append("  postgres_data:")
        
        # Redis Cache
        if components.get("redis", False):
            services.append(f'''  redis:
    image: redis:7-alpine
    container_name: {project_name}_redis
    ports:
      - "6379:6379"
    networks:
      - ml_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
''')
        
        # Vector Database (for RAG)
        if components.get("vector_db", False):
            services.append(f'''  chromadb:
    image: chromadb/chroma:latest
    container_name: {project_name}_chromadb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
    networks:
      - ml_network
    restart: unless-stopped
''')
            volumes.append("  chroma_data:")
        
        # API Service
        if components.get("api", False):
            depends_on = []
            if components.get("ml_model"):
                depends_on.append("ml_model")
            if components.get("database"):
                depends_on.append("database")
            if components.get("redis"):
                depends_on.append("redis")
            
            depends_on_str = ""
            if depends_on:
                depends_on_str = "depends_on:\n" + "\n".join([f"      - {s}" for s in depends_on])

            services.append(f'''  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: {project_name}_api
    ports:
      - "8000:8000"
    volumes:
      - ./api/app:/app/app
    environment:
      - DATABASE_URL=postgresql://mluser:mlpassword@database:5432/{project_name}_db
      - REDIS_URL=redis://redis:6379
      - ML_SERVICE_URL=http://ml_model:5000
      - CORS_ORIGINS=http://localhost:8501
    {depends_on_str}
    networks:
      - ml_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
''')
        
        # Frontend Service
        if components.get("frontend", False):
            depends_on = []
            if components.get("api"):
                depends_on.append("api")
            
            depends_on_str = ""
            if depends_on:
                depends_on_str = "depends_on:\n" + "\n".join([f"      - {s}" for s in depends_on])

            services.append(f'''  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: {project_name}_frontend
    ports:
      - "8501:8501"
    volumes:
      - ./frontend:/app
    environment:
      - API_URL=http://api:8000
    {depends_on_str}
    networks:
      - ml_network
    restart: unless-stopped
''')
        
        volumes_str = ""
        if volumes:
            volumes_str = "volumes:\n" + "\n".join(volumes)

        compose_content = f"""# Docker Compose Configuration for {project_name}
# Development Environment

version: '3.8'

services:
{chr(10).join(services)}

networks:
  ml_network:
    driver: bridge

{volumes_str}
"""
        
        return compose_content
    
    def _create_docker_compose_prod(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Create docker-compose.prod.yml for production."""
        
        return f'''# Docker Compose Configuration for {project_name}
# Production Environment

version: '3.8'

services:
  # Production services with optimized settings
  # - No volume mounts for code
  # - Resource limits
  # - Proper logging
  # - Health checks
  # - Restart policies

  # TODO: Add production-specific configurations
  # - Use pre-built images
  # - Add resource limits
  # - Configure logging drivers
  # - Add secrets management

# Use this file with:
# docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
'''
    
    def _create_env_example(self, components: Dict, config: Dict) -> str:
        """Create .env.example file."""
        
        env_vars = [
            "# Environment Configuration",
            "",
            "# General",
            "ENVIRONMENT=development",
            "LOG_LEVEL=INFO",
            ""
        ]
        
        if components.get("database"):
            env_vars.extend([
                "# Database",
                "POSTGRES_USER=mluser",
                "POSTGRES_PASSWORD=mlpassword",
                "DATABASE_URL=postgresql://mluser:mlpassword@database:5432/ml_db",
                ""
            ])
        
        if components.get("redis"):
            env_vars.extend([
                "# Redis",
                "REDIS_URL=redis://redis:6379",
                ""
            ])
        
        if components.get("ml_model"):
            env_vars.extend([
                "# ML Model",
                "MODEL_PATH=/app/checkpoints/best_model.pth",
                "ML_SERVICE_URL=http://ml_model:5000",
                ""
            ])
        
        if components.get("api"):
            env_vars.extend([
                "# API",
                "API_URL=http://api:8000",
                "CORS_ORIGINS=http://localhost:8501",
                ""
            ])
        
        if config.get("llm_provider") == "OpenAI":
            env_vars.extend([
                "# OpenAI",
                "OPENAI_API_KEY=your_openai_key_here",
                ""
            ])
        
        return "\n".join(env_vars)
    
    def _create_dockerignore(self) -> str:
        """Create .dockerignore file."""
        
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Docker
Dockerfile*
docker-compose*.yml

# Data (large files)
data/
datasets/
*.csv
*.parquet

# Models (large files - use volumes instead)
models/
checkpoints/
*.pth
*.pt
*.h5
*.pb

# Documentation
docs/
*.md

# Tests
tests/
'''
    
    def _create_build_script(self) -> str:
        """Create build.sh script."""
        
        return '''#!/bin/bash
# Build Script for Docker Images

set -e

echo "🔨 Building Docker images..."

# Build ML model image
if [ -d "ml_model" ]; then
    echo "Building ML model service..."
    docker build -t ml-model:latest ./ml_model
fi

# Build API image
if [ -d "api" ]; then
    echo "Building API service..."
    docker build -t api:latest ./api
fi

# Build frontend image
if [ -d "frontend" ]; then
    echo "Building frontend service..."
    docker build -t frontend:latest ./frontend
fi

echo "✅ All images built successfully!"
echo "Run 'docker-compose up' to start services"
'''
    
    def _create_deploy_script(self) -> str:
        """Create deploy.sh script."""
        
        return '''#!/bin/bash
# Deployment Script

set -e

echo "🚀 Deploying application..."

# Pull latest changes
git pull origin main

# Build images
./scripts/build.sh

# Stop existing containers
docker-compose down

# Start new containers
docker-compose up -d

# Show logs
docker-compose logs -f
'''
    
    def _generate_kubernetes_manifests(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Kubernetes manifests."""
        
        files = {}
        
        # Namespace
        files["k8s/namespace.yaml"] = f'''apiVersion: v1
kind: Namespace
metadata:
  name: {project_name}
'''
        
        # ConfigMap
        files["k8s/configmap.yaml"] = self._create_k8s_configmap(project_name, components)
        
        # Secrets
        files["k8s/secrets.yaml"] = self._create_k8s_secrets(project_name)
        
        # Deployments for each service
        if components.get("ml_model"):
            files["k8s/ml-model-deployment.yaml"] = self._create_k8s_deployment(
                project_name, "ml-model", "5000", config
            )
        
        if components.get("api"):
            files["k8s/api-deployment.yaml"] = self._create_k8s_deployment(
                project_name, "api", "8000", config
            )
        
        if components.get("frontend"):
            files["k8s/frontend-deployment.yaml"] = self._create_k8s_deployment(
                project_name, "frontend", "8501", config
            )
        
        # Services
        files["k8s/services.yaml"] = self._create_k8s_services(project_name, components)
        
        # Ingress
        files["k8s/ingress.yaml"] = self._create_k8s_ingress(project_name, components)
        
        # HPA (Horizontal Pod Autoscaler)
        if config.get("autoscaling", False):
            files["k8s/hpa.yaml"] = self._create_k8s_hpa(project_name, components)
        
        return files
    
    def _create_k8s_configmap(self, project_name: str, components: Dict) -> str:
        """Create Kubernetes ConfigMap."""
        
        return f'''apiVersion: v1
kind: ConfigMap
metadata:
  name: {project_name}-config
  namespace: {project_name}
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  API_URL: "http://api-service:8000"
  ML_SERVICE_URL: "http://ml-model-service:5000"
'''
    
    def _create_k8s_secrets(self, project_name: str) -> str:
        """Create Kubernetes Secrets template."""
        
        return f'''apiVersion: v1
kind: Secret
metadata:
  name: {project_name}-secrets
  namespace: {project_name}
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:password@postgres:5432/db"
  OPENAI_API_KEY: "your-api-key-here"
  # Add other secrets here
'''
    
    def _create_k8s_deployment(
        self,
        project_name: str,
        service_name: str,
        port: str,
        config: Dict
    ) -> str:
        """Create Kubernetes Deployment."""
        
        return f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}-deployment
  namespace: {project_name}
  labels:
    app: {service_name}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
    spec:
      containers:
      - name: {service_name}
        image: {service_name}:latest
        ports:
        - containerPort: {port}
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: {project_name}-config
              key: LOG_LEVEL
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
'''
    
    def _create_k8s_services(self, project_name: str, components: Dict) -> str:
        """Create Kubernetes Services."""
        
        services = []
        
        if components.get("ml_model"):
            services.append(f'''apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
  namespace: {project_name}
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 5000
    targetPort: 5000
  type: ClusterIP
''')
        
        if components.get("api"):
            services.append(f'''---
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: {project_name}
spec:
  selector:
    app: api
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
''')
        
        if components.get("frontend"):
            services.append(f'''---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: {project_name}
spec:
  selector:
    app: frontend
  ports:
  - protocol: TCP
    port: 8501
    targetPort: 8501
  type: LoadBalancer
''')
        
        return "\n".join(services)
    
    def _create_k8s_ingress(self, project_name: str, components: Dict) -> str:
        """Create Kubernetes Ingress."""
        
        return f'''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {project_name}-ingress
  namespace: {project_name}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: {project_name}.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 8501
'''
    
    def _create_k8s_hpa(self, project_name: str, components: Dict) -> str:
        """Create Horizontal Pod Autoscaler."""
        
        return f'''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: {project_name}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
    
    def _generate_ci_cd_pipeline(
        self,
        project_name: str,
        components: Dict[str, bool],
        platform: str
    ) -> Dict[str, str]:
        """Generate CI/CD pipeline configuration."""
        
        files = {}
        
        if platform == "github":
            files[".github/workflows/ci-cd.yml"] = self._create_github_actions(
                project_name, components
            )
        elif platform == "gitlab":
            files[".gitlab-ci.yml"] = self._create_gitlab_ci(project_name, components)
        
        return files
    
    def _create_github_actions(self, project_name: str, components: Dict) -> str:
        """Create GitHub Actions workflow."""
        
        # Dynamically create build steps based on components
        build_steps = []
        if components.get("ml_model", False):
            build_steps.append('''
    - name: Build and push ML Model image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/ml-model:latest ./ml_model
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/ml-model:latest
''')
        if components.get("api", False):
            build_steps.append('''
    - name: Build and push API image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/api:latest ./api
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/api:latest
''')
        if components.get("frontend", False):
            build_steps.append('''
    - name: Build and push Frontend image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:latest ./frontend
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:latest
''')
        
        build_steps_str = "\n".join(build_steps)

        return f'''name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
{build_steps_str}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      run: |
        # Add kubectl deployment commands here
        echo "Deploying to Kubernetes..."
        # kubectl apply -f k8s/
'''


    
    def _create_gitlab_ci(self, project_name: str, components: Dict) -> str:
        """Create GitLab CI configuration."""
        
        return '''# GitLab CI/CD Pipeline
# TODO: Implement GitLab CI configuration
'''
    
    def _generate_cloud_config(
        self,
        project_name: str,
        components: Dict[str, bool],
        cloud_provider: str
    ) -> Dict[str, str]:
        """Generate cloud-specific deployment configs."""
        
        files = {}
        
        if cloud_provider == "aws":
            files["cloud/aws-ecs-task-definition.json"] = "# AWS ECS\n# TODO"
        elif cloud_provider == "gcp":
            files["cloud/gcp-cloud-run.yaml"] = "# GCP Cloud Run\n# TODO"
        elif cloud_provider == "azure":
            files["cloud/azure-container-apps.yaml"] = "# Azure\n# TODO"
        
        return files
    
    def _generate_monitoring_config(
        self,
        project_name: str,
        components: Dict[str, bool]
    ) -> Dict[str, str]:
        """Generate monitoring configuration."""
        
        files = {}
        
        # Prometheus config
        files["monitoring/prometheus.yml"] = f'''# Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-model'
    static_configs:
      - targets: ['ml-model:5000']
  
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
'''
        
        # Grafana dashboard
        files["monitoring/grafana-dashboard.json"] = '{"dashboard": "TODO"}'
        
        return files
    
    def _generate_nginx_config(
        self,
        project_name: str,
        components: Dict[str, bool]
    ) -> Dict[str, str]:
        """Generate Nginx reverse proxy configuration."""
        
        nginx_conf = f'''# Nginx Reverse Proxy Configuration
upstream api_backend {{
    server api:8000;
}}

upstream frontend_backend {{
    server frontend:8501;
}}

server {{
    listen 80;
    server_name {project_name}.local;

    # API endpoints
    location /api/ {{
        proxy_pass http://api_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    # Frontend
    location / {{
        proxy_pass http://frontend_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }}
}}
'''
        
        return {"nginx/nginx.conf": nginx_conf}
    
    def _generate_deployment_docs(
        self,
        project_name: str,
        components: Dict[str, bool],
        config: Dict[str, Any]
    ) -> str:
        """Generate deployment documentation."""
        
        return f'''# Deployment Guide for {project_name}

## Architecture

This project consists of the following components:
{"- ML Model Service (port 5000)" if components.get('ml_model') else ""}
{"- FastAPI Backend (port 8000)" if components.get('api') else ""}
{"- Streamlit Frontend (port 8501)" if components.get('frontend') else ""}
{"- PostgreSQL Database (port 5432)" if components.get('database') else ""}
{"- Redis Cache (port 6379)" if components.get('redis') else ""}

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- (Optional) Kubernetes cluster for production deployment

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd {project_name}
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Build and run services**
```bash
docker-compose up --build
```

4. **Access services**
{"- ML Model API: http://localhost:5000" if components.get('ml_model') else ""}
{"- Backend API: http://localhost:8000" if components.get('api') else ""}
{"- Frontend: http://localhost:8501" if components.get('frontend') else ""}

## Production Deployment

### Using Docker Compose

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

{"### Using Kubernetes" if config.get('kubernetes') else ""}
{"```bash" if config.get('kubernetes') else ""}
{"kubectl apply -f k8s/" if config.get('kubernetes') else ""}
{"```" if config.get('kubernetes') else ""}

## Monitoring

{"- Prometheus: http://localhost:9090" if config.get('monitoring') else ""}
{"- Grafana: http://localhost:3000" if config.get('monitoring') else ""}

## Troubleshooting

### Check service health
```bash
docker-compose ps
docker-compose logs -f <service-name>
```

### Restart services
```bash
docker-compose restart <service-name>
```

### Clean rebuild
```bash
docker-compose down -v
docker-compose up --build
```

## Scaling

Adjust replicas in docker-compose.yml or k8s/deployments

## Security Checklist

- [ ] Change default passwords in .env
- [ ] Use secrets management (Kubernetes Secrets, AWS Secrets Manager)
- [ ] Enable HTTPS/TLS
- [ ] Set up firewall rules
- [ ] Enable authentication on all services
- [ ] Regular security updates

## CI/CD

{"GitHub Actions workflow configured in `.github/workflows/ci-cd.yml`" if config.get('ci_cd') and config.get('ci_platform') == 'github' else ""}

## License

MIT
'''


# Example usage
async def main():
    """Example usage of DeploymentAgent."""
    agent = DeploymentAgent()
    
    # Generate deployment config for full-stack ML project
    result = await agent.generate_deployment_config(
        project_name="ml_classifier",
        components={
            "ml_model": True,
            "api": True,
            "frontend": True,
            "database": True,
            "redis": True,
            "vector_db": False
        },
        config={
            "gpu": True,
            "kubernetes": True,
            "ci_cd": True,
            "ci_platform": "github",
            "monitoring": True,
            "nginx": True
        }
    )
    
    print(f"Generated {len(result)} deployment files:")
    for filename in result.keys():
        print(f"  - {filename}")


if __name__ == "__main__":
    asyncio.run(main())
