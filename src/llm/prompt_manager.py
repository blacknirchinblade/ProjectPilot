

"""
Prompt Manager - Load and manage prompts for different agents

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""


import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class PromptManager:
    """Manage prompts for different agents"""
    
    def __init__(self, prompts_dir: str = "config/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_cache: Dict[str, Dict[str, Any]] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt files"""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return
        
        for prompt_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    category = prompt_file.stem
                    self.prompts_cache[category] = yaml.safe_load(f)
                    logger.info(f"Loaded prompts from {prompt_file.name}")
            except Exception as e:
                logger.error(f"Error loading {prompt_file}: {str(e)}")
    
    def get_prompt(
        self, 
        category: str, 
        prompt_name: str, 
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get a prompt template and fill in variables
        
        Args:
            category: Prompt category (e.g., 'planning', 'coding')
            prompt_name: Name of the prompt
            variables: Variables to format the prompt
        
        Returns:
            Formatted prompt string
        """
        if category not in self.prompts_cache:
            raise ValueError(f"Prompt category '{category}' not found")
        
        if prompt_name not in self.prompts_cache[category]:
            raise ValueError(f"Prompt '{prompt_name}' not found in category '{category}'")
        
        prompt_obj = self.prompts_cache[category][prompt_name]

        # Extract prompt template from different possible structures
        if isinstance(prompt_obj, dict):
            if 'prompt' in prompt_obj:
                prompt_template = prompt_obj['prompt']
            else:
                # If it's a dict but no 'prompt' key, try to use the first string value
                string_values = [v for v in prompt_obj.values() if isinstance(v, str)]
                if string_values:
                    prompt_template = string_values[0]
                else:
                    raise ValueError(f"No prompt string found in prompt object for '{prompt_name}'")
        elif isinstance(prompt_obj, str):
            prompt_template = prompt_obj
        else:
            raise ValueError(f"Invalid prompt format for '{prompt_name}' in category '{category}'. Expected string or dict.")
        
        # Format the prompt with variables
        if variables:
            try:
                # Convert any non-string values to JSON strings for better formatting
                formatted_vars = {}
                for key, value in variables.items():
                    if isinstance(value, (dict, list)):
                        formatted_vars[key] = json.dumps(value, indent=2)
                    else:
                        formatted_vars[key] = value
                
                return prompt_template.format(**formatted_vars)
            except KeyError as e:
                logger.error(f"Missing variable in prompt: {e}")
                # Provide a fallback by replacing variables manually
                formatted_prompt = prompt_template
                for key, value in formatted_vars.items():
                    placeholder = "{" + key + "}"
                    formatted_prompt = formatted_prompt.replace(placeholder, str(value))
                return formatted_prompt
            except Exception as e:
                logger.error(f"Error formatting prompt: {e}")
                return prompt_template  # Return unformatted prompt as fallback
        
        return prompt_template
    
    def get_system_instruction(self, agent_type: str) -> str:
        """Get system instruction for agent type"""
        system_prompts = self.prompts_cache.get("system_instructions", {})
        return system_prompts.get(agent_type, "You are a helpful AI coding assistant.")