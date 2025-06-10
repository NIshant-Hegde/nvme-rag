"""
Configuration manager for NVMe RAG system.
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from nvme_rag.config.models import NVMeRAGConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    DEFAULT_CONFIG_FILE = Path.home() / ".nvme-rag" / "config.yml"
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file. 
                        If None, uses default location.
        """
        self.config_file = Path(config_file) if config_file else self.DEFAULT_CONFIG_FILE
        self._config: Optional[NVMeRAGConfig] = None
        
        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def config(self) -> NVMeRAGConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def load(self) -> NVMeRAGConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                
                logger.info(f"Loaded configuration from {self.config_file}")
                return NVMeRAGConfig.from_dict(data)
                
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                logger.info("Using default configuration")
        
        # Create default configuration
        config = NVMeRAGConfig()
        self.save(config)
        return config
    
    def save(self, config: Optional[NVMeRAGConfig] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save. If None, saves current config.
        """
        config = config or self.config
        
        try:
            # Convert config to dictionary for YAML serialization
            config_dict = config.to_dict()
            
            # Add header comment
            yaml_content = self._generate_yaml_header() + yaml.dump(
                config_dict, 
                default_flow_style=False, 
                sort_keys=True,
                indent=2
            )
            
            with open(self.config_file, 'w') as f:
                f.write(yaml_content)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reset_to_defaults(self) -> NVMeRAGConfig:
        """Reset configuration to defaults."""
        logger.info("Resetting configuration to defaults")
        self._config = NVMeRAGConfig()
        self.save()
        return self._config
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update a specific configuration section.
        
        Args:
            section: Section name (e.g., 'ollama', 'embedding')
            updates: Dictionary of updates to apply
        """
        config = self.config
        
        if not hasattr(config, section):
            raise ValueError(f"Unknown configuration section: {section}")
        
        section_obj = getattr(config, section)
        
        for key, value in updates.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                logger.info(f"Updated {section}.{key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {section}.{key}")
        
        self.save()
    
    def get_section(self, section: str) -> Any:
        """Get a specific configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section object
        """
        config = self.config
        if not hasattr(config, section):
            raise ValueError(f"Unknown configuration section: {section}")
        return getattr(config, section)
    
    def validate(self) -> Dict[str, Any]:
        """Validate current configuration.
        
        Returns:
            Dictionary with validation results
        """
        config = self.config
        issues = []
        warnings = []
        
        # Validate system configuration
        if not config.system.data_dir:
            issues.append("System data directory not configured")
        elif not Path(config.system.data_dir).parent.exists():
            warnings.append(f"Parent directory of data_dir does not exist: {config.system.data_dir}")
        
        # Validate Ollama configuration
        if not config.ollama.base_url:
            issues.append("Ollama base URL not configured")
        
        if not config.ollama.model:
            issues.append("Ollama model not specified")
        
        # Validate embedding configuration
        if not config.embedding.model_name:
            issues.append("Embedding model name not specified")
        
        # Validate processing configuration
        if config.processing.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        
        if config.processing.chunk_overlap >= config.processing.chunk_size:
            issues.append("Chunk overlap must be less than chunk size")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def setup_environment(self) -> None:
        """Set up environment based on configuration."""
        config = self.config
        
        # Create necessary directories
        config.create_directories()
        
        # Set environment variables if needed
        if config.ollama.base_url:
            os.environ["OLLAMA_HOST"] = config.ollama.base_url
        
        logger.info("Environment setup completed")
    
    def _generate_yaml_header(self) -> str:
        """Generate header comment for YAML file."""
        return """# NVMe RAG Configuration File
# This file contains all configuration settings for the NVMe RAG system.
# You can edit this file directly or use 'nvme-rag configure' command.

"""
    
    def export_config(self, output_file: Path) -> None:
        """Export configuration to a file.
        
        Args:
            output_file: Path to output file
        """
        config_dict = self.config.to_dict()
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Configuration exported to {output_file}")
    
    def import_config(self, input_file: Path) -> None:
        """Import configuration from a file.
        
        Args:
            input_file: Path to input file
        """
        with open(input_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self._config = NVMeRAGConfig.from_dict(data)
        self.save()
        
        logger.info(f"Configuration imported from {input_file}")
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration with environment variable overrides."""
        config_dict = self.config.to_dict()
        
        # Apply environment variable overrides
        env_overrides = {
            "NVME_RAG_DATA_DIR": ("system", "data_dir"),
            "NVME_RAG_LOG_LEVEL": ("system", "log_level"),
            "OLLAMA_HOST": ("ollama", "base_url"),
            "OLLAMA_MODEL": ("ollama", "model"),
        }
        
        for env_var, (section, key) in env_overrides.items():
            if env_var in os.environ:
                if section in config_dict and key in config_dict[section]:
                    config_dict[section][key] = os.environ[env_var]
        
        return config_dict