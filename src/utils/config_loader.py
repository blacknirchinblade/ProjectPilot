"""
Config Loader - Load configuration from .env and config files

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from loguru import logger


class ConfigLoader:
    """Load and manage configuration"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize config loader
        
        Args:
            env_file: Path to .env file (default: .env in project root)
        """
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in current or parent directories
            current = Path.cwd()
            while current != current.parent:
                env_path = current / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from {env_path}")
                    break
                current = current.parent
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get configuration value from environment
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        return os.getenv(key, default)
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default {default}")
            return default
    
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_required(key: str) -> str:
        """
        Get required configuration value
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration value
        
        Raises:
            ValueError: If key not found
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required configuration '{key}' not found in environment")
        return value
