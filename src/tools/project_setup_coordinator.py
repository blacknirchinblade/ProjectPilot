"""
Project Setup Coordinator - Bridge Planning to Project Setup

This module coordinates the project setup phase, integrating the SetupAutomationTool
with the planning workflow. It:
- Maps planning analysis to SetupConfig
- Extracts project metadata (name, type, dependencies)
- Executes automated project setup
- Reports setup results back to the workflow

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from .setup_automation_tool import (
    SetupAutomationTool,
    SetupConfig,
    ProjectType,
    EnvironmentType,
    SetupPhase
)


class ProjectSetupCoordinator:
    """
    Coordinates project setup after planning phase.
    
    This coordinator bridges the planning agent output and the setup automation tool,
    extracting necessary information from planning analysis and executing automated setup.
    
    Attributes:
        default_project_path: Base path for project creation
        auto_setup: Whether to automatically trigger setup
    """
    
    def __init__(
        self,
        default_project_path: Optional[Path] = None,
        auto_setup: bool = True
    ):
        """
        Initialize project setup coordinator.
        
        Args:
            default_project_path: Base directory for projects (default: ./output)
            auto_setup: Whether to automatically setup projects after planning
        """
        self.default_project_path = default_project_path or Path("./output")
        self.auto_setup = auto_setup
        
        logger.info(f"ProjectSetupCoordinator initialized (auto_setup={auto_setup})")
    
    def extract_project_metadata(
        self,
        project_name: str,
        project_description: str,
        planning_analysis: str
    ) -> Dict[str, Any]:
        """
        Extract project metadata from planning analysis.
        
        Args:
            project_name: Project name
            project_description: Project description
            planning_analysis: Output from planning agent
        
        Returns:
            Dictionary containing extracted metadata:
            - project_type: Detected project type
            - dependencies: List of dependencies
            - python_version: Detected Python version
            - requires_gpu: Whether GPU is needed
            - frameworks: Detected frameworks
        """
        metadata = {
            "project_type": ProjectType.LIBRARY,  # Default
            "dependencies": [],
            "python_version": "3.10",  # Default
            "requires_gpu": False,
            "frameworks": []
        }
        
        # Combine all text for analysis
        all_text = f"{project_name}\n{project_description}\n{planning_analysis}".lower()
        
        # Detect project type
        if any(keyword in all_text for keyword in ["ml", "machine learning", "deep learning", "neural network", "model", "training", "dataset", "cifar", "mnist", "pytorch", "tensorflow", "keras"]):
            metadata["project_type"] = ProjectType.ML
            metadata["requires_gpu"] = "gpu" in all_text or "cuda" in all_text or "training" in all_text
        elif any(keyword in all_text for keyword in ["web", "api", "flask", "fastapi", "django", "server", "endpoint", "rest", "http"]):
            metadata["project_type"] = ProjectType.WEB
        elif any(keyword in all_text for keyword in ["cli", "command", "terminal", "script"]):
            metadata["project_type"] = ProjectType.CLI
        
        # Detect Python version
        python_version_match = re.search(r"python\s*(\d+\.\d+)", all_text)
        if python_version_match:
            metadata["python_version"] = python_version_match.group(1)
        
        # Extract dependencies from common patterns
        dependencies = set()
        
        # Framework detection
        frameworks = {
            "pytorch": ["torch", "torchvision"],
            "tensorflow": ["tensorflow"],
            "keras": ["keras", "tensorflow"],
            "scikit-learn": ["scikit-learn"],
            "pandas": ["pandas"],
            "numpy": ["numpy"],
            "matplotlib": ["matplotlib"],
            "seaborn": ["seaborn"],
            "flask": ["flask"],
            "fastapi": ["fastapi", "uvicorn"],
            "django": ["django"],
            "requests": ["requests"],
            "pytest": ["pytest"],
            "jupyterlab": ["jupyterlab"],
            "jupyter": ["jupyter"]
        }
        
        for framework, packages in frameworks.items():
            if framework in all_text:
                dependencies.update(packages)
                metadata["frameworks"].append(framework)
        
        # Common ML dependencies
        if metadata["project_type"] == ProjectType.ML:
            dependencies.update(["numpy", "matplotlib"])
            if "pytorch" in all_text or "torch" in all_text:
                dependencies.update(["torch", "torchvision"])
            elif "tensorflow" in all_text or "keras" in all_text:
                dependencies.update(["tensorflow"])
            
            # Optional ML tools
            if "notebook" in all_text or "jupyter" in all_text:
                dependencies.add("jupyter")
            if "plot" in all_text or "visualization" in all_text:
                dependencies.update(["matplotlib", "seaborn"])
        
        # Common web dependencies
        elif metadata["project_type"] == ProjectType.WEB:
            if "flask" in all_text:
                dependencies.add("flask")
            elif "fastapi" in all_text:
                dependencies.update(["fastapi", "uvicorn"])
            elif "django" in all_text:
                dependencies.add("django")
        
        # Always add testing
        dependencies.add("pytest")
        
        metadata["dependencies"] = sorted(list(dependencies))
        
        logger.info(f"Extracted metadata: type={metadata['project_type'].value}, "
                   f"deps={len(metadata['dependencies'])}, "
                   f"gpu={metadata['requires_gpu']}")
        
        return metadata
    
    def create_setup_config(
        self,
        project_name: str,
        project_description: str,
        planning_analysis: str,
        output_path: Optional[Path] = None,
        environment_type: EnvironmentType = EnvironmentType.VENV,
        force_mode: bool = False
    ) -> SetupConfig:
        """
        Create SetupConfig from planning analysis.
        
        Args:
            project_name: Project name
            project_description: Project description  
            planning_analysis: Output from planning agent
            output_path: Custom output path (optional)
            environment_type: Type of environment to create
            force_mode: Whether to force setup even if errors occur
        
        Returns:
            SetupConfig object ready for SetupAutomationTool
        """
        # Extract metadata
        metadata = self.extract_project_metadata(
            project_name,
            project_description,
            planning_analysis
        )
        
        # Clean project name for directory
        clean_name = re.sub(r'[^\w\s-]', '', project_name)
        clean_name = re.sub(r'[-\s]+', '_', clean_name).strip('_').lower()
        
        # Determine project path
        if output_path:
            project_path = output_path / clean_name
        else:
            project_path = self.default_project_path / clean_name
        
        # Create config
        config = SetupConfig(
            project_name=clean_name,
            project_type=metadata["project_type"],
            python_version=metadata["python_version"],
            dependencies=metadata["dependencies"],
            project_path=str(project_path),
            description=project_description[:200] if project_description else "",
            env_type=environment_type,
            init_git=True,
            skip_venv=False,
            skip_dependencies=False,
            force=force_mode
        )
        
        logger.info(f"Created SetupConfig for '{clean_name}' at {project_path}")
        return config
    
    def setup_project(
        self,
        project_name: str,
        project_description: str,
        planning_analysis: str,
        output_path: Optional[Path] = None,
        environment_type: EnvironmentType = EnvironmentType.VENV,
        force_mode: bool = False,
        skip_phases: Optional[List[SetupPhase]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete project setup.
        
        This is the main entry point for setting up a project after planning.
        
        Args:
            project_name: Project name
            project_description: Project description
            planning_analysis: Output from planning agent
            output_path: Custom output path (optional)
            environment_type: Type of environment to create
            force_mode: Whether to force setup even if errors occur
            skip_phases: Phases to skip (optional)
        
        Returns:
            Dictionary containing:
            - status: "success" or "error"
            - project_path: Path to created project
            - config: SetupConfig used
            - results: Phase-by-phase results
            - errors: List of errors if any
        """
        try:
            # Create setup config
            config = self.create_setup_config(
                project_name,
                project_description,
                planning_analysis,
                output_path=output_path,
                environment_type=environment_type,
                force_mode=force_mode
            )
            
            project_path = Path(config.project_path)
            project_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting project setup at: {project_path}")
            
            # Create setup tool
            setup_tool = SetupAutomationTool(config=config)
            
            # Execute setup
            setup_result = setup_tool.setup_project()
            
            # Process results
            if setup_result.success:
                logger.info(f"✅ Project setup completed successfully: {project_path}")
                return {
                    "status": "success",
                    "project_path": str(project_path),
                    "config": config.to_dict(),
                    "results": {
                        "structure": setup_result.structure_result,
                        "environment": setup_result.env_result,
                        "dependencies": setup_result.deps_result,
                        "configuration": setup_result.config_result,
                        "validation": setup_result.validation_result
                    },
                    "summary": {
                        "phases_completed": setup_result.phases_completed,
                        "phases_failed": setup_result.phases_failed,
                        "total_time": setup_result.total_time
                    },
                    "errors": []
                }
            else:
                logger.warning(f"⚠️  Project setup completed with errors: {project_path}")
                return {
                    "status": "partial_success",
                    "project_path": str(project_path),
                    "config": config.to_dict(),
                    "results": {
                        "structure": setup_result.structure_result,
                        "environment": setup_result.env_result,
                        "dependencies": setup_result.deps_result,
                        "configuration": setup_result.config_result,
                        "validation": setup_result.validation_result
                    },
                    "summary": {
                        "phases_completed": setup_result.phases_completed,
                        "phases_failed": setup_result.phases_failed,
                        "total_time": setup_result.total_time
                    },
                    "errors": setup_result.phases_failed
                }
        
        except Exception as e:
            logger.error(f"❌ Project setup failed: {str(e)}")
            return {
                "status": "error",
                "project_path": None,
                "config": None,
                "results": {},
                "errors": [str(e)]
            }
    
    def validate_setup(self, project_path: Path) -> Dict[str, Any]:
        """
        Validate that project setup was successful.
        
        Args:
            project_path: Path to project directory
        
        Returns:
            Validation results dictionary
        """
        validation = {
            "project_exists": project_path.exists(),
            "has_src": (project_path / "src").exists(),
            "has_tests": (project_path / "tests").exists(),
            "has_venv": (project_path / "venv").exists() or (project_path / ".venv").exists(),
            "has_git": (project_path / ".git").exists(),
            "has_readme": (project_path / "README.md").exists(),
            "has_requirements": (project_path / "requirements.txt").exists()
        }
        
        validation["all_valid"] = all([
            validation["project_exists"],
            validation["has_src"],
            validation["has_tests"]
        ])
        
        logger.info(f"Validation: {sum(validation.values())} / {len(validation)} checks passed")
        
        return validation
    
    async def coordinate_setup_phase(
        self,
        project_name: str,
        project_description: str,
        planning_analysis: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async wrapper for setup_project (for integration with async workflows).
        
        Args:
            project_name: Project name
            project_description: Project description
            planning_analysis: Planning agent output
            **kwargs: Additional arguments for setup_project
        
        Returns:
            Setup results dictionary
        """
        # Execute synchronous setup
        result = self.setup_project(
            project_name,
            project_description,
            planning_analysis,
            **kwargs
        )
        
        # Add validation if successful
        if result["status"] in ("success", "partial_success") and result["project_path"]:
            project_path = Path(result["project_path"])
            result["validation"] = self.validate_setup(project_path)
        
        return result


# ==================== Convenience Functions ====================

def setup_project_from_planning(
    project_name: str,
    project_description: str,
    planning_analysis: str,
    output_path: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to setup project from planning results.
    
    Args:
        project_name: Project name
        project_description: Project description
        planning_analysis: Planning agent output
        output_path: Custom output path (optional)
        **kwargs: Additional configuration options
    
    Returns:
        Setup results dictionary
    """
    coordinator = ProjectSetupCoordinator(default_project_path=output_path)
    return coordinator.setup_project(
        project_name,
        project_description,
        planning_analysis,
        output_path=output_path,
        **kwargs
    )


async def async_setup_project_from_planning(
    project_name: str,
    project_description: str,
    planning_analysis: str,
    output_path: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Async convenience function to setup project from planning results.
    
    Args:
        project_name: Project name
        project_description: Project description
        planning_analysis: Planning agent output
        output_path: Custom output path (optional)
        **kwargs: Additional configuration options
    
    Returns:
        Setup results dictionary
    """
    coordinator = ProjectSetupCoordinator(default_project_path=output_path)
    return await coordinator.coordinate_setup_phase(
        project_name,
        project_description,
        planning_analysis,
        output_path=output_path,
        **kwargs
    )
