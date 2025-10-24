"""
Project Data Class

Defines the structure for a single project.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime

@dataclass
class Project:
    """Represents a single generated project."""
    id: str
    name: str
    description: str
    project_type: str
    date: str
    path: str
    file_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert project object to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.project_type,
            "date": self.date,
            "path": self.path,
            "file_count": self.file_count,
            "metadata": self.metadata,
            "last_modified": self.last_modified
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Create a Project object from a dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", "Untitled Project"),
            description=data.get("description", ""),
            project_type=data.get("type", "unknown"),
            date=data.get("date", datetime.now().isoformat()),
            path=data.get("path", ""),
            file_count=data.get("file_count", 0),
            metadata=data.get("metadata", {}),
            last_modified=data.get("last_modified", "")
        )
