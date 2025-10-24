"""
Project Manager for Streamlit Application

Manages project history, saving, loading, and organization.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from .project import Project


class ProjectManager:
    """
    Manages project history and operations.
    
    Features:
    - Save and load projects
    - Project history tracking
    - Export projects as ZIP
    - Import existing projects
    - Resume project work
    """
    
    def __init__(self, projects_dir: str = "data/projects"):
        """
        Initialize the project manager.
        
        Args:
            projects_dir: Directory to store project data
        """
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.projects_dir / "projects.json"
        self.projects: Dict[str, Project] = {}
        
        self._load_metadata()
        logger.info(f"ProjectManager initialized with {len(self.projects)} projects")
    
    def _load_metadata(self):
        """Load project metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    projects_data = json.load(f)
                    self.projects = {pid: Project.from_dict(pdata) for pid, pdata in projects_data.items()}
                logger.info(f"Loaded {len(self.projects)} projects from metadata")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.projects = {}
        else:
            self.projects = {}
    
    def _save_metadata(self):
        """Save project metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({pid: p.to_dict() for pid, p in self.projects.items()}, f, indent=2)
            logger.info("Project metadata saved")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add_project(self, project_data: Dict[str, Any]) -> str:
        """
        Add a new project to history.
        
        Args:
            project_data: Project information dictionary
        
        Returns:
            Project ID
        """
        project_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        project = Project(
            id=project_id,
            name=project_data.get("name", f"Project {project_id}"),
            description=project_data.get("description", ""),
            project_type=project_data.get("type", "unknown"),
            date=datetime.now().isoformat(),
            path=str(self.projects_dir / project_id),
            file_count=project_data.get("file_count", 0),
            metadata=project_data.get("metadata", {})
        )
        
        # Create project directory
        project_dir = Path(project.path)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save project files if provided
        if "files" in project_data:
            for file_data in project_data["files"]:
                file_path = project_dir / file_data["name"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file_data["content"])
        
        self.projects[project_id] = project
        self._save_metadata()
        
        logger.info(f"Added project: {project.name} (ID: {project_id})")
        return project_id
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project identifier
        
        Returns:
            Project object or None if not found
        """
        return self.projects.get(project_id)

    def get_all_projects(self) -> List[Project]:
        """
        Get all projects sorted by date (newest first).
        
        Returns:
            List of project objects
        """
        projects = list(self.projects.values())
        projects.sort(key=lambda p: p.date, reverse=True)
        return projects
    
    def get_recent_projects(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent projects.
        
        Args:
            limit: Maximum number of projects to return
        
        Returns:
            List of recent project dictionaries
        """
        all_projects = self.get_all_projects()
        return all_projects[:limit]
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update project metadata.
        
        Args:
            project_id: Project identifier
            updates: Dictionary of fields to update
        
        Returns:
            True if successful, False otherwise
        """
        if project_id not in self.projects:
            logger.warning(f"Project not found: {project_id}")
            return False
        
        self.projects[project_id].update(updates)
        self.projects[project_id]["last_modified"] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info(f"Updated project: {project_id}")
        return True
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and its files.
        
        Args:
            project_id: Project identifier
        
        Returns:
            True if successful, False otherwise
        """
        if project_id not in self.projects:
            logger.warning(f"Project not found: {project_id}")
            return False
        
        project = self.projects[project_id]
        project_dir = Path(project["path"])
        
        try:
            # Delete project directory
            if project_dir.exists():
                shutil.rmtree(project_dir)
            
            # Remove from metadata
            del self.projects[project_id]
            self._save_metadata()
            
            logger.info(f"Deleted project: {project['name']} (ID: {project_id})")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            return False
    
    def export_project(self, project_id: str, output_path: Optional[str] = None) -> Optional[Path]:
        """
        Export a project as a ZIP file.
        
        Args:
            project_id: Project identifier
            output_path: Optional output path for ZIP file
        
        Returns:
            Path to the ZIP file or None if failed
        """
        if project_id not in self.projects:
            logger.warning(f"Project not found: {project_id}")
            return None
        
        project = self.projects[project_id]
        project_dir = Path(project["path"])
        
        if not project_dir.exists():
            logger.error(f"Project directory not found: {project_dir}")
            return None
        
        # Determine output path
        if output_path is None:
            output_path = self.projects_dir / f"{project['name']}_{project_id}.zip"
        else:
            output_path = Path(output_path)
        
        try:
            # Create ZIP file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in project_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(project_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Exported project to: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting project {project_id}: {e}")
            return None
    
    def import_project(
        self,
        source_path: str,
        project_name: Optional[str] = None,
        project_description: str = ""
    ) -> Optional[str]:
        """
        Import a project from a ZIP file or directory.
        
        Args:
            source_path: Path to ZIP file or directory
            project_name: Optional project name
            project_description: Project description
        
        Returns:
            Project ID if successful, None otherwise
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            logger.error(f"Source path not found: {source_path}")
            return None
        
        # Generate project ID
        project_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import based on source type
            if source_path.is_file() and source_path.suffix == '.zip':
                # Extract ZIP
                with zipfile.ZipFile(source_path, 'r') as zipf:
                    zipf.extractall(project_dir)
                
                if project_name is None:
                    project_name = source_path.stem
            
            elif source_path.is_dir():
                # Copy directory
                shutil.copytree(source_path, project_dir, dirs_exist_ok=True)
                
                if project_name is None:
                    project_name = source_path.name
            
            else:
                logger.error(f"Unsupported source type: {source_path}")
                return None
            
            # Count files
            file_count = len(list(project_dir.rglob("*.py")))
            
            # Add to projects
            project_data = {
                "name": project_name,
                "description": project_description,
                "type": "imported",
                "file_count": file_count,
                "metadata": {
                    "source": str(source_path),
                    "import_date": datetime.now().isoformat()
                }
            }
            
            project = {
                "id": project_id,
                "name": project_data["name"],
                "description": project_data["description"],
                "type": project_data["type"],
                "date": datetime.now().isoformat(),
                "path": str(project_dir),
                "file_count": project_data["file_count"],
                "metadata": project_data["metadata"]
            }
            
            self.projects[project_id] = project
            self._save_metadata()
            
            logger.info(f"Imported project: {project_name} (ID: {project_id})")
            return project_id
        
        except Exception as e:
            logger.error(f"Error importing project: {e}")
            # Cleanup on failure
            if project_dir.exists():
                shutil.rmtree(project_dir)
            return None
    
    def search_projects(
        self,
        query: str,
        search_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search projects by query string.
        
        Args:
            query: Search query
            search_fields: Fields to search in (default: name, description)
        
        Returns:
            List of matching project dictionaries
        """
        if search_fields is None:
            search_fields = ["name", "description"]
        
        query = query.lower()
        results = []
        
        for project in self.projects.values():
            for field in search_fields:
                field_value = str(project.get(field, "")).lower()
                if query in field_value:
                    results.append(project)
                    break
        
        return results
    
    def get_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get all files in a project.
        
        Args:
            project_id: Project identifier
        
        Returns:
            List of file dictionaries with path, size, modified date
        """
        if project_id not in self.projects:
            logger.warning(f"Project not found: {project_id}")
            return []
        
        project = self.projects[project_id]
        project_dir = Path(project["path"])
        
        if not project_dir.exists():
            logger.error(f"Project directory not found: {project_dir}")
            return []
        
        files = []
        for file_path in project_dir.rglob("*"):
            if file_path.is_file():
                files.append({
                    "path": str(file_path.relative_to(project_dir)),
                    "absolute_path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "extension": file_path.suffix
                })
        
        return files
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get project statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_projects = len(self.projects)
        total_files = sum(p.get("file_count", 0) for p in self.projects.values())
        
        project_types = {}
        for project in self.projects.values():
            project_type = project.get("type", "unknown")
            project_types[project_type] = project_types.get(project_type, 0) + 1
        
        return {
            "total_projects": total_projects,
            "total_files": total_files,
            "project_types": project_types,
            "recent_activity": self.get_recent_projects(5)
        }
