"""
Configuration file templates for Setup Automation Tool.

Contains templates for generating various project configuration files.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, List, Any


class ConfigTemplates:
    """Templates for generating configuration files."""
    
    @staticmethod
    def get_setup_py(project_info: Dict[str, Any]) -> str:
        """Generate setup.py content."""
        install_requires = project_info.get("install_requires", [])
        extras_require = project_info.get("extras_require", {})
        
        return f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script for {project_info["name"]}."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{project_info["name"]}",
    version="{project_info["version"]}",
    description="{project_info["description"]}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="{project_info["author"]}",
    author_email="{project_info["author_email"]}",
    license="{project_info["license"]}",
    url="{project_info.get("homepage", "")}",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    python_requires=">={project_info["python_version"]}",
    install_requires={install_requires},
    extras_require={extras_require},
    classifiers={project_info.get("classifiers", [])},
    keywords={project_info.get("keywords", [])},
)
'''
    
    @staticmethod
    def get_pyproject_toml(project_info: Dict[str, Any]) -> str:
        """Generate pyproject.toml content (PEP 621)."""
        dependencies = project_info.get("install_requires", [])
        dev_dependencies = project_info.get("dev_requires", [])
        
        # Format dependencies as TOML list
        deps_str = ",\n    ".join([f'"{dep}"' for dep in dependencies])
        dev_deps_str = ",\n    ".join([f'"{dep}"' for dep in dev_dependencies])
        
        return f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_info["name"]}"
version = "{project_info["version"]}"
description = "{project_info["description"]}"
readme = "README.md"
authors = [
    {{name = "{project_info["author"]}", email = "{project_info["author_email"]}"}}
]
license = {{text = "{project_info["license"]}"}}
requires-python = ">={project_info["python_version"]}"
keywords = {project_info.get("keywords", [])}
classifiers = {project_info.get("classifiers", [])}

dependencies = [
    {deps_str}
]

[project.optional-dependencies]
dev = [
    {dev_deps_str}
]

[project.urls]
Homepage = "{project_info.get("homepage", "")}"
Repository = "{project_info.get("repository", "")}"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.black]
line-length = 100
target-version = ['py{project_info["python_version"].replace(".", "")}']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "{project_info["python_version"]}"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
'''
    
    @staticmethod
    def get_setup_cfg(project_info: Dict[str, Any]) -> str:
        """Generate setup.cfg content."""
        return f'''[metadata]
name = {project_info["name"]}
version = {project_info["version"]}
description = {project_info["description"]}
author = {project_info["author"]}
author_email = {project_info["author_email"]}
license = {project_info["license"]}
long_description = file: README.md
long_description_content_type = text/markdown

[options]
packages = find:
package_dir = 
    = src
python_requires = >={project_info["python_version"]}
install_requires =
    {chr(10).join("    " + dep for dep in project_info.get("install_requires", []))}

[options.packages.find]
where = src

[options.extras_require]
dev =
    {chr(10).join("    " + dep for dep in project_info.get("dev_requires", []))}

[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,.venv,venv

[pylint]
max-line-length = 100
'''
    
    @staticmethod
    def get_gitignore(template: str = "python") -> str:
        """Generate .gitignore content."""
        if template == "python":
            return '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project specific
data/
logs/
output/
*.log
.env.local
'''
        return ""
    
    @staticmethod
    def get_manifest_in() -> str:
        """Generate MANIFEST.in content."""
        return '''include README.md
include LICENSE
include requirements.txt
recursive-include src *.py
recursive-include tests *.py
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
'''
    
    @staticmethod
    def get_editorconfig() -> str:
        """Generate .editorconfig content."""
        return '''# EditorConfig is awesome: https://EditorConfig.org

root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.py]
indent_size = 4
max_line_length = 100

[*.{yml,yaml}]
indent_size = 2

[*.{json,toml}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false
'''
    
    @staticmethod
    def get_pytest_ini(project_name: str = "") -> str:
        """Generate pytest.ini content."""
        return f'''[pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --cov={project_name}
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
'''
    
    @staticmethod
    def get_tox_ini(python_version: str = "3.10") -> str:
        """Generate tox.ini content."""
        return f'''[tox]
envlist = py{python_version.replace(".", "")},lint
isolated_build = True

[testenv]
deps =
    pytest
    pytest-cov
commands =
    pytest {{posargs}}

[testenv:lint]
deps =
    black
    flake8
    mypy
    isort
commands =
    black --check src tests
    flake8 src tests
    mypy src
    isort --check-only src tests
'''
    
    @staticmethod
    def get_requirements_txt(dependencies: List[str]) -> str:
        """Generate requirements.txt content."""
        return "\n".join(dependencies) + "\n"
    
    @staticmethod
    def get_readme_template(project_info: Dict[str, Any]) -> str:
        """Generate README.md template."""
        return f'''# {project_info["name"]}

{project_info["description"]}

## Installation

```bash
pip install {project_info["name"]}
```

## Usage

```python
import {project_info["name"]}

# Your code here
```

## Development

### Setup

```bash
# Clone the repository
git clone {project_info.get("repository", "")}
cd {project_info["name"]}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src tests

# Lint
flake8 src tests

# Type check
mypy src
```

## License

{project_info["license"]}

## Author

{project_info["author"]} ({project_info["author_email"]})
'''
    
    @staticmethod
    def get_license_template(license_type: str = "MIT", author: str = "", year: int = 2025) -> str:
        """Generate LICENSE file content."""
        if license_type == "MIT":
            return f'''MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
        return f"Copyright (c) {year} {author}. All rights reserved."
