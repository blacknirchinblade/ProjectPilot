
"""
Dynamic Architecture Agent - Designs project structure from requirements. (Corrected)

No hardcoded templates - fully adaptive to project needs.


Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import re
import traceback
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


@dataclass
class FileSpec:
    """Specification for a single file to be generated."""
    
    path: str                       # "src/models/backbone.py"
    name: str = ""                  # "backbone.py"
    purpose: str = ""               # "CSPDarknet backbone implementation"
    estimated_lines: int = 200      # 200
    dependencies: List[str] = field(default_factory=list)  # ["torch", "src.utils.logger"]
    exports: List[str] = field(default_factory=list)       # ["CSPDarknet", "conv_block"]
    components: List[Dict[str, str]] = field(default_factory=list)  # [{name, description, type}, ...]
    priority: int = 1               # Build priority (lower = earlier)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the FileSpec object to a dictionary."""
        return asdict(self)


@dataclass
class ProjectArchitecture:
    """Complete project architecture with dependencies."""
    
    structure: Dict[str, List[FileSpec]]  # "src/models/": [FileSpec(...), ...]
    dependencies: Set[str] = field(default_factory=set) # All unique dependencies
    build_order: List[str] = field(default_factory=list)  # Order to generate files
    import_map: Dict[str, Dict[str, str]] = field(default_factory=dict)  # "CSPDarknet" → {"module": "src.models.backbone", "symbol": "CSPDarknet"}
    
    # Add a property to check if the structure is empty
    @property
    def file_specs(self) -> List[FileSpec]:
        """Returns a flat list of all file specs."""
        all_specs = []
        for specs in self.structure.values():
            all_specs.extend(specs)
        return all_specs

    def get_file_spec(self, filepath: str) -> Optional[FileSpec]:
        """Get FileSpec for a specific path."""
        for spec in self.file_specs:
            if spec.path == filepath:
                return spec
        return None
    
    def get_total_files(self) -> int:
        """Total number of files to generate."""
        return len(self.file_specs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the ProjectArchitecture object to a dictionary."""
        return {
            "structure": {
                dir_name: [spec.to_dict() for spec in files]
                for dir_name, files in self.structure.items()
            },
            "dependencies": list(self.dependencies),
            "build_order": self.build_order,
            "import_map": self.import_map,
        }


class DynamicArchitectAgent(BaseAgent):
    """
    Analyzes requirements and designs optimal project architecture.
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize DynamicArchitectAgent
        
        Args:
            llm_client: LLM client for generation
            prompt_manager: Prompt manager for templates
        """
        super().__init__(
            name="dynamic_architect_agent",
            role="Expert Software Architect",
            agent_type="architecture",
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        logger.info(f"{self.name} initialized")
    
    async def execute_task(self, task: Dict) -> Any:
        """
        Execute architecture design task (implements BaseAgent abstract method).
        """
        task_type = task.get("task_type", "design_architecture")
        
        if task_type == "design_architecture":
            # --- : Unpack data correctly ---
            data = task.get("data", {})
            enhanced_spec = data.get("enhanced_spec", "")
            components = data.get("components", {})
            clarifications = data.get("clarifications", {})
            
            architecture = await self.design_architecture(
                enhanced_spec, components, clarifications
            )
            
            # Return the object itself, as expected by streamlit_app.py
            return architecture
        else:
            logger.error(f"Unknown task type for DAA: {task_type}")
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def design_architecture(
        self,
        enhanced_spec: Dict[str, Any],
        components: Dict[str, bool],
        clarifications: Dict[str, str]
    ) -> ProjectArchitecture:
        """
        Design complete project architecture from requirements.
        
        Args:
            enhanced_spec: User's project description
            components: Dictionary of selected components (e.g., {"fastapi": True})
            clarifications: Dictionary of user's answers to questions
        
        Returns:
            Complete ProjectArchitecture with all file specs
        """
        logger.info(f"Designing architecture for: {enhanced_spec.get('description', '')[:100]}...")
        
        constraints = {
            'max_lines_per_file': 250,
            'min_files': 5,
            'prefer_modular': True
        }
        
        # Build design prompt
        prompt = self._build_design_prompt(enhanced_spec, components, clarifications, constraints)
        
        # Get architecture from LLM
        response = await self.generate_response(prompt)
        
        # Parse response into ProjectArchitecture
        architecture = self._parse_architecture_response(response)
        
        # This was the line that failed (architecture was None)
        # Now it will fail if the *parsed* architecture is empty
        if not architecture or not architecture.structure:
             logger.error("LLM returned an empty or invalid architecture. Response: {response[:500]}")
             # Return an empty but valid object to prevent NoneType error
             return ProjectArchitecture(structure={})

        # Calculate dependencies, build order, and import map
        architecture.dependencies = self._extract_all_dependencies(architecture)
        architecture.import_map = self._create_import_map(architecture)
        architecture.build_order = self._calculate_build_order(architecture)
        
        logger.success(f"Architecture designed: {architecture.get_total_files()} files across {len(architecture.structure)} directories")
        
        return architecture
    
    def _build_design_prompt(
        self, 
        enhanced_spec: Dict, 
        components: Dict, 
        clarifications: Dict, 
        constraints: Dict
    ) -> str:
        """Create comprehensive prompt for architecture design."""
        
        component_list = [f"- {name}" for name, enabled in components.items() if enabled]
        clarification_list = [f"- {q_id}: {answer}" for q_id, answer in clarifications.items()]

        # Try to load the prompt from the manager
        try:
             # --- FIX: We will load the prompt from the YAML file ---
            # We created this file in a previous step.
            prompt_template = self.get_prompt(
                "architecture_prompts", 
                "design_project_architecture"
            )
            
            return prompt_template.format(
                enhanced_spec=json.dumps(enhanced_spec, indent=2),
                component_list="\n".join(component_list) or "N/A",
                clarification_list="\n".join(clarification_list) or "N/A",
                max_lines_per_file=constraints['max_lines_per_file'],
                min_files=constraints['min_files']
            )
            
        except ValueError:
            logger.error("Could not load 'design_project_architecture' prompt. Using fallback.")
            # Fallback hardcoded prompt (the one from your file)
            return f"""You are an expert software architect. Design a clean, modular project structure.

REQUIREMENTS:
{json.dumps(enhanced_spec, indent=2)}

ENABLED COMPONENTS:
{chr(10).join(component_list)}

USER CLARIFICATIONS:
{chr(10).join(clarification_list) or "N/A"}

DESIGN CONSTRAINTS:
- Max lines per file: {constraints['max_lines_per_file']}
- Minimum files: {constraints['min_files']}
- Prefer modular design: True
- ALL file paths must start with `src/`, `tests/`, `deployment/`, or be a root file (e.g., `requirements.txt`).
- ALL internal dependencies MUST start with `src.` (e.g., `src.utils.logging`).
- **--- FIX: ADDED THIS RULE ---**
- **DO NOT** generate tasks for documentation files like `README.md` or files in the `docs/` folder. The DocumentationAgent will handle this separately.

DESIGN PRINCIPLES:
1. **Small, Focused Files**: Each file has ONE clear purpose
2. **Logical Grouping**: Related files in same directory
3. **Clear Interfaces**: `exports` lists main classes/functions
4. **Dependency-Aware**: `dependencies` lists all *internal* (`src.module.file`) and *external* (`package_name`) imports.

RESPOND WITH JSON (no markdown, just raw JSON):
{{
  "structure": {{
    "src/data/": [
      {{
        "name": "dataset.py",
        "purpose": "PyTorch dataset and dataloader for CIFAR-10.",
        "estimated_lines": 120,
        "dependencies": ["torch", "torchvision", "torchvision.transforms"],
        "exports": ["get_data_loaders"],
        "components": [{{"name": "get_data_loaders", "type": "function", "description": "Returns train, val, and test dataloaders."}}],
        "priority": 1
      }}
    ],
    "src/models/": [
      {{
        "name": "cnn_model.py",
        "purpose": "Defines the simple custom CNN architecture.",
        "estimated_lines": 90,
        "dependencies": ["torch", "torch.nn"],
        "exports": ["SimpleCNN"],
        "components": [{{"name": "SimpleCNN", "type": "class", "description": "A 3-layer CNN model."}}],
        "priority": 1
      }}
    ],
    "src/training/": [
       {{
        "name": "train.py",
        "purpose": "Main training script. Orchestrates data loading, model training, and evaluation.",
        "estimated_lines": 200,
        "dependencies": ["torch", "src.data.dataset", "src.models.cnn_model", "src.core.config"],
        "exports": ["train_model"],
        "components": [{{"name": "train_model", "type": "function", "description": "Main training and validation loop."}}],
        "priority": 2
      }}
    ],
    "src/core/": [
       {{
        "name": "config.py",
        "purpose": "Loads configuration (hyperparameters, paths) from a YAML file or env variables.",
        "estimated_lines": 80,
        "dependencies": ["pydantic"],
        "exports": ["settings"],
        "components": [{{"name": "Settings", "type": "class", "description": "Pydantic class for settings management."}}],
        "priority": 1
      }}
    ],
    "root": [
      {{
        "name": "requirements.txt",
        "purpose": "Lists all external Python dependencies.",
        "estimated_lines": 20,
        "dependencies": [],
        "exports": [],
        "components": [],
        "priority": 1
      }}
    ]
  }}
}}
"""
    
    def _parse_architecture_response(self, response: str) -> Optional[ProjectArchitecture]:
        """
        Parse LLM response into ProjectArchitecture.
        This is a robust parser to handle malformed JSON from the LLM.
        """
        try:
            # --- ROBUST JSON PARSING ---
            # 1. Find the first '{' and the last '}'
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start == -1 or json_end == -1:
                logger.error(f"No JSON object found in response. Response start: {response[:200]}")
                raise json.JSONDecodeError("No JSON object found in response", response, 0)
            
            # 2. Extract the JSON string
            json_str = response[json_start:json_end+1]
            
            # 3. Clean up common LLM mistakes
            # Remove trailing commas before a closing bracket or brace
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
            
            # 4. Parse the JSON
            data = json.loads(json_str)
            # --- END ROBUST JSON PARSING ---
            
            structure = {}
            if "structure" not in data or not isinstance(data["structure"], dict):
                raise ValueError("JSON response missing 'structure' dictionary or it's not a dict.")

            for directory, files_data in data["structure"].items():
                if not isinstance(files_data, list):
                    logger.warning(f"Skipping malformed directory entry for '{directory}': not a list.")
                    continue
                    
                file_specs = []
                for file_data in files_data:
                    if not isinstance(file_data, dict):
                        logger.warning(f"Skipping malformed file entry (not a dict) in '{directory}': {str(file_data)[:100]}...")
                        continue
                    
                    file_name = file_data.get("name") or file_data.get("filename")
                    
                    if not file_name:
                        logger.warning(f"Skipping malformed file entry (no 'name' or 'filename') in '{directory}': {str(file_data)[:100]}...")
                        continue

                    dir_path = Path(directory)
                    if directory.lower() == 'root':
                        file_path = str(file_name).replace("\\", "/")
                    else:
                        file_path = str(dir_path / file_name).replace("\\", "/")

                    components_list = file_data.get("components", [])
                    if components_list and isinstance(components_list[0], str):
                        components_data = [{"name": comp, "description": "N/A", "type": "unknown"} for comp in components_list]
                    else:
                        components_data = components_list

                    spec = FileSpec(
                        path=file_path,
                        name=file_name,
                        purpose=file_data.get("purpose", "No purpose provided."),
                        estimated_lines=file_data.get("estimated_lines", 100),
                        dependencies=file_data.get("dependencies", []),
                        exports=file_data.get("exports", []),
                        components=components_data,
                        priority=file_data.get("priority", 1)
                    )
                    file_specs.append(spec)
                structure[directory] = file_specs
            
            if not structure:
                logger.warning("JSON was valid but 'structure' was empty.")
                return None

            return ProjectArchitecture(structure=structure)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse architecture JSON: {e}")
            
            # --- FIX for IndexError ---
            if e.pos < len(e.doc):
                logger.error(f"Offending character: '{e.doc[e.pos]}'")
                logger.error(f"Response snippet: {e.doc[max(0, e.pos-150):e.pos+150]}")
            else:
                logger.error("Error is at the end of the JSON string. The string may be truncated or incomplete.")
            # --- END FIX ---
                
            logger.debug(f"Full response: {response}")
            return None # Return None on failure
        except Exception as e:
            logger.error(f"Failed to parse architecture structure: {e}\n{traceback.format_exc()}")
            logger.debug(f"Response was: {response[:500]}")
            return None # Return None on failure
    
    def _extract_all_dependencies(self, arch: ProjectArchitecture) -> Set[str]:
        """Extract all unique external dependencies from the architecture."""
        all_deps = set()
        for spec in arch.file_specs:
            for dep in spec.dependencies:
                if not dep.startswith("src."): # Filter out internal dependencies
                    all_deps.add(dep)
        return all_deps

    def _calculate_build_order(self, arch: ProjectArchitecture) -> List[str]:
        """
        Calculate build order using topological sort.
        Files with no dependencies built first.
        """
        logger.info("Calculating build order...")
        
        # Build dependency graph
        graph = {}
        all_files_set = {spec.path for spec in arch.file_specs}
        
        for spec in arch.file_specs:
            graph[spec.path] = []
            for dep in spec.dependencies:
                if dep.startswith("src."):
                    # Convert import path to file path
                    # "src.models.backbone" → "src/models/backbone.py"
                    dep_filepath = dep.replace(".", "/") + ".py"
                    graph[spec.path].append(dep_filepath)
        
        # Topological sort
        sorted_files = []
        visited = set()
        recursion_stack = set()
        
        def visit(filepath: str):
            if filepath not in all_files_set:
                logger.warning(f"Dependency '{filepath}' not found in project structure. Ignoring.")
                return

            if filepath in visited:
                return
            if filepath in recursion_stack:
                logger.error(f"Circular dependency detected: {filepath}")
                return
            
            recursion_stack.add(filepath)
            
            for dep_filepath in graph.get(filepath, []):
                visit(dep_filepath)
            
            recursion_stack.remove(filepath)
            visited.add(filepath)
            sorted_files.append(filepath)
        
        all_specs = sorted(arch.file_specs, key=lambda s: s.priority)
        
        for spec in all_specs:
            if spec.path not in visited:
                visit(spec.path)
        
        logger.success(f"Build order calculated: {len(sorted_files)} files")
        return sorted_files
    
    def _create_import_map(self, arch: ProjectArchitecture) -> Dict[str, Dict[str, str]]:
        """
        Create symbol → {module, symbol} mapping.
        """
        logger.info("Creating import map...")
        
        import_map = {}
        
        for spec in arch.file_specs:
            import_path = spec.path.replace("/", ".").removesuffix(".py")
            
            for export in spec.exports:
                if export in import_map:
                    logger.warning(f"Duplicate export: {export} in {import_path} and {import_map[export]['module']}")
                import_map[export] = {
                    "module": import_path,
                    "symbol": export
                }
        
        logger.success(f"Import map created: {len(import_map)} symbols")
        return import_map
    
    def save_architecture(self, arch: ProjectArchitecture, output_path: Path):
        """Save architecture to JSON for persistence."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(arch.to_dict(), f, indent=2)
            logger.success(f"Architecture saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save architecture: {e}")
    
    @staticmethod
    def load_architecture(input_path: Path) -> ProjectArchitecture:
        """Load architecture from JSON."""
        with open(input_path) as f:
            data = json.load(f)
        
        structure = {}
        for directory, files_data in data["structure"].items():
            file_specs = []
            for file_data in files_data:
                components_list = file_data.get("components", [])
                if components_list and isinstance(components_list[0], str):
                    components_data = [{"name": comp, "description": "N/A", "type": "unknown"} for comp in components_list]
                else:
                    components_data = components_list
                
                file_data["components"] = components_data
                file_specs.append(FileSpec(**file_data))
            structure[directory] = file_specs
        
        return ProjectArchitecture(
            structure=structure,
            dependencies=set(data.get("dependencies", [])),
            build_order=data.get("build_order", []),
            import_map=data.get("import_map", {})
        )