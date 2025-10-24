"""
Configuration File Generator for Setup Automation Tool.

Generates various project configuration files using templates.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .data_structures import (
    ConfigResult,
    ProjectInfo,
    ConfigFileType,
)
from .config_templates import ConfigTemplates


class ConfigGenerator:
    """Generate project configuration files."""
    
    def __init__(self, project_info: ProjectInfo, output_dir: Path):
        """
        Initialize config generator.
        
        Args:
            project_info: Project metadata
            output_dir: Directory to generate files in
        """
        self.project_info = project_info
        self.output_dir = Path(output_dir)
        self.templates = ConfigTemplates()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_setup_py(
        self,
        install_requires: List[str],
        extras_require: Optional[Dict[str, List[str]]] = None
    ) -> Path:
        """
        Generate setup.py file.
        
        Args:
            install_requires: List of package dependencies
            extras_require: Optional extras dependencies
            
        Returns:
            Path to generated file
        """
        logger.info(f"Generating setup.py for {self.project_info.name}")
        
        project_dict = {
            "name": self.project_info.name,
            "version": self.project_info.version,
            "description": self.project_info.description,
            "author": self.project_info.author,
            "author_email": self.project_info.author_email,
            "license": self.project_info.license,
            "python_version": self.project_info.python_version,
            "homepage": self.project_info.homepage,
            "install_requires": install_requires,
            "extras_require": extras_require or {},
            "classifiers": self.project_info.get_classifiers(),
            "keywords": self.project_info.keywords,
        }
        
        content = self.templates.get_setup_py(project_dict)
        
        output_path = self.output_dir / "setup.py"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated setup.py at {output_path}")
        return output_path
    
    def generate_pyproject_toml(
        self,
        dependencies: List[str],
        dev_dependencies: Optional[List[str]] = None
    ) -> Path:
        """
        Generate pyproject.toml file (PEP 621).
        
        Args:
            dependencies: List of package dependencies
            dev_dependencies: Optional development dependencies
            
        Returns:
            Path to generated file
        """
        logger.info(f"Generating pyproject.toml for {self.project_info.name}")
        
        project_dict = {
            "name": self.project_info.name,
            "version": self.project_info.version,
            "description": self.project_info.description,
            "author": self.project_info.author,
            "author_email": self.project_info.author_email,
            "license": self.project_info.license,
            "python_version": self.project_info.python_version,
            "homepage": self.project_info.homepage,
            "repository": self.project_info.repository,
            "install_requires": dependencies,
            "dev_requires": dev_dependencies or [],
            "classifiers": self.project_info.get_classifiers(),
            "keywords": self.project_info.keywords,
        }
        
        content = self.templates.get_pyproject_toml(project_dict)
        
        output_path = self.output_dir / "pyproject.toml"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated pyproject.toml at {output_path}")
        return output_path
    
    def generate_setup_cfg(
        self,
        install_requires: List[str],
        dev_requires: Optional[List[str]] = None
    ) -> Path:
        """
        Generate setup.cfg file.
        
        Args:
            install_requires: List of package dependencies
            dev_requires: Optional development dependencies
            
        Returns:
            Path to generated file
        """
        logger.info(f"Generating setup.cfg for {self.project_info.name}")
        
        project_dict = {
            "name": self.project_info.name,
            "version": self.project_info.version,
            "description": self.project_info.description,
            "author": self.project_info.author,
            "author_email": self.project_info.author_email,
            "license": self.project_info.license,
            "python_version": self.project_info.python_version,
            "install_requires": install_requires,
            "dev_requires": dev_requires or [],
        }
        
        content = self.templates.get_setup_cfg(project_dict)
        
        output_path = self.output_dir / "setup.cfg"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated setup.cfg at {output_path}")
        return output_path
    
    def generate_gitignore(self, template: str = "python") -> Path:
        """
        Generate .gitignore file.
        
        Args:
            template: Template type (default: "python")
            
        Returns:
            Path to generated file
        """
        logger.info("Generating .gitignore")
        
        content = self.templates.get_gitignore(template)
        
        output_path = self.output_dir / ".gitignore"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated .gitignore at {output_path}")
        return output_path
    
    def generate_manifest_in(self) -> Path:
        """
        Generate MANIFEST.in file.
        
        Returns:
            Path to generated file
        """
        logger.info("Generating MANIFEST.in")
        
        content = self.templates.get_manifest_in()
        
        output_path = self.output_dir / "MANIFEST.in"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated MANIFEST.in at {output_path}")
        return output_path
    
    def generate_editorconfig(self) -> Path:
        """
        Generate .editorconfig file.
        
        Returns:
            Path to generated file
        """
        logger.info("Generating .editorconfig")
        
        content = self.templates.get_editorconfig()
        
        output_path = self.output_dir / ".editorconfig"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated .editorconfig at {output_path}")
        return output_path
    
    def generate_pytest_ini(self) -> Path:
        """
        Generate pytest.ini file.
        
        Returns:
            Path to generated file
        """
        logger.info("Generating pytest.ini")
        
        content = self.templates.get_pytest_ini(self.project_info.name)
        
        output_path = self.output_dir / "pytest.ini"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated pytest.ini at {output_path}")
        return output_path
    
    def generate_tox_ini(self) -> Path:
        """
        Generate tox.ini file.
        
        Returns:
            Path to generated file
        """
        logger.info("Generating tox.ini")
        
        content = self.templates.get_tox_ini(self.project_info.python_version)
        
        output_path = self.output_dir / "tox.ini"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated tox.ini at {output_path}")
        return output_path
    
    def generate_requirements_txt(self, dependencies: List[str]) -> Path:
        """
        Generate requirements.txt file.
        
        Args:
            dependencies: List of package dependencies
            
        Returns:
            Path to generated file
        """
        logger.info("Generating requirements.txt")
        
        content = self.templates.get_requirements_txt(dependencies)
        
        output_path = self.output_dir / "requirements.txt"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated requirements.txt at {output_path}")
        return output_path
    
    def generate_readme(self) -> Path:
        """
        Generate README.md file.
        
        Returns:
            Path to generated file
        """
        logger.info("Generating README.md")
        
        project_dict = {
            "name": self.project_info.name,
            "description": self.project_info.description,
            "author": self.project_info.author,
            "author_email": self.project_info.author_email,
            "license": self.project_info.license,
            "repository": self.project_info.repository,
        }
        
        content = self.templates.get_readme_template(project_dict)
        
        output_path = self.output_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated README.md at {output_path}")
        return output_path
    
    def generate_license(self) -> Path:
        """
        Generate LICENSE file.
        
        Returns:
            Path to generated file
        """
        logger.info(f"Generating LICENSE ({self.project_info.license})")
        
        content = self.templates.get_license_template(
            license_type=self.project_info.license,
            author=self.project_info.author,
            year=datetime.now().year
        )
        
        output_path = self.output_dir / "LICENSE"
        output_path.write_text(content, encoding="utf-8")
        
        logger.success(f"Generated LICENSE at {output_path}")
        return output_path
    
    def generate_all(
        self,
        config_types: List[str],
        dependencies: Optional[List[str]] = None,
        dev_dependencies: Optional[List[str]] = None
    ) -> ConfigResult:
        """
        Generate multiple configuration files.
        
        Args:
            config_types: List of config file types to generate
            dependencies: Optional list of dependencies
            dev_dependencies: Optional list of dev dependencies
            
        Returns:
            ConfigResult with generation status
        """
        start_time = datetime.now()
        generated_files = []
        failed_files = []
        
        deps = dependencies or []
        dev_deps = dev_dependencies or []
        
        logger.info(f"Generating {len(config_types)} configuration files")
        
        for config_type in config_types:
            try:
                # Convert enum to value if needed
                if isinstance(config_type, ConfigFileType):
                    config_value = config_type.value
                else:
                    config_value = str(config_type)
                
                if config_value in ["setup.py", ConfigFileType.SETUP_PY.value] or config_type == ConfigFileType.SETUP_PY:
                    path = self.generate_setup_py(deps, {"dev": dev_deps})
                    generated_files.append(path)
                    
                elif config_value in ["pyproject.toml", ConfigFileType.PYPROJECT_TOML.value] or config_type == ConfigFileType.PYPROJECT_TOML:
                    path = self.generate_pyproject_toml(deps, dev_deps)
                    generated_files.append(path)
                    
                elif config_value in ["setup.cfg", ConfigFileType.SETUP_CFG.value] or config_type == ConfigFileType.SETUP_CFG:
                    path = self.generate_setup_cfg(deps, dev_deps)
                    generated_files.append(path)
                    
                elif config_value in [".gitignore", "gitignore", ConfigFileType.GITIGNORE.value] or config_type == ConfigFileType.GITIGNORE:
                    path = self.generate_gitignore()
                    generated_files.append(path)
                    
                elif config_value in ["MANIFEST.in", ConfigFileType.MANIFEST_IN.value] or config_type == ConfigFileType.MANIFEST_IN:
                    path = self.generate_manifest_in()
                    generated_files.append(path)
                    
                elif config_value in [".editorconfig", ConfigFileType.EDITORCONFIG.value] or config_type == ConfigFileType.EDITORCONFIG:
                    path = self.generate_editorconfig()
                    generated_files.append(path)
                    
                elif config_value in ["pytest.ini", ConfigFileType.PYTEST_INI.value] or config_type == ConfigFileType.PYTEST_INI:
                    path = self.generate_pytest_ini()
                    generated_files.append(path)
                    
                elif config_value in ["tox.ini", ConfigFileType.TOX_INI.value] or config_type == ConfigFileType.TOX_INI:
                    path = self.generate_tox_ini()
                    generated_files.append(path)
                    
                elif config_value in ["requirements.txt", ConfigFileType.REQUIREMENTS_TXT.value] or config_type == ConfigFileType.REQUIREMENTS_TXT:
                    path = self.generate_requirements_txt(deps)
                    generated_files.append(path)
                    
                elif config_value in ["README.md", "README", ConfigFileType.README.value] or config_type == ConfigFileType.README:
                    path = self.generate_readme()
                    generated_files.append(path)
                    
                elif config_value in ["LICENSE", ConfigFileType.LICENSE.value] or config_type == ConfigFileType.LICENSE:
                    path = self.generate_license()
                    generated_files.append(path)
                    
                else:
                    logger.warning(f"Unknown config type: {config_type}")
                    failed_files.append(config_type)
                    
            except Exception as e:
                logger.error(f"Failed to generate {config_type}: {e}")
                failed_files.append(config_type)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        success = len(failed_files) == 0
        message = f"Generated {len(generated_files)} configuration files"
        if failed_files:
            message += f", {len(failed_files)} failed"
        
        logger.info(f"Configuration generation complete: {message} in {duration:.2f}s")
        
        return ConfigResult(
            success=success,
            generated_files=generated_files,
            failed_files=failed_files,
            message=message,
            duration=duration
        )
    
    def validate_generated_files(self) -> Dict[str, bool]:
        """
        Validate that generated files exist and are valid.
        
        Returns:
            Dictionary mapping file names to validation status
        """
        validation = {}
        
        for file_name in [
            "setup.py", "pyproject.toml", "setup.cfg",
            ".gitignore", "README.md", "LICENSE",
            "pytest.ini", "requirements.txt"
        ]:
            file_path = self.output_dir / file_name
            validation[file_name] = file_path.exists() and file_path.stat().st_size > 0
        
        return validation
