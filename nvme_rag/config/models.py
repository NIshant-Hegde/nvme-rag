"""
Configuration models for NVMe RAG system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import os


@dataclass
class SystemConfig:
    """System-level configuration."""
    data_dir: Path = field(default_factory=lambda: Path.home() / ".nvme-rag" / "data")
    log_level: str = "INFO"
    max_parallel_jobs: int = 4
    temp_dir: Optional[Path] = None
    
    def __post_init__(self):
        # Ensure paths are Path objects
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if self.temp_dir and isinstance(self.temp_dir, str):
            self.temp_dir = Path(self.temp_dir)


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    base_url: str = "http://localhost:11434"
    model: str = "gemma3:12b-it-qat"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 30
    health_check_timeout: int = 5
    auto_start: bool = True


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    device: str = "cpu"
    cache_embeddings: bool = True
    batch_size: int = 32
    max_seq_length: int = 512
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        if self.cache_dir and isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    strategy: str = "hybrid"  # semantic, hybrid, filtered, reranked
    max_results: int = 10
    confidence_threshold: float = 0.7
    rerank_results: bool = True
    similarity_threshold: float = 0.5
    context_window_size: int = 3
    enable_filtering: bool = True


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_structure: bool = True
    extract_metadata: bool = True
    supported_formats: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".md"])


@dataclass
class UIConfig:
    """User interface configuration."""
    color_output: bool = True
    progress_bars: bool = True
    verbose_errors: bool = True
    table_format: str = "rich"
    pager: bool = True
    auto_confirm: bool = False


@dataclass
class NVMeRAGConfig:
    """Main configuration class for NVMe RAG system."""
    system: SystemConfig = field(default_factory=SystemConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    def __post_init__(self):
        """Set up derived paths after initialization."""
        # Set embedding cache dir relative to data dir
        if not self.embedding.cache_dir:
            self.embedding.cache_dir = self.system.data_dir / "embeddings_cache"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NVMeRAGConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        # Re-run post init to set derived values
        config.__post_init__()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in ["system", "ollama", "embedding", "retrieval", "processing", "ui"]:
            section = getattr(self, field_name)
            section_dict = {}
            
            for attr_name in dir(section):
                if not attr_name.startswith("_"):
                    value = getattr(section, attr_name)
                    # Convert Path objects to strings
                    if isinstance(value, Path):
                        value = str(value)
                    elif not callable(value):
                        section_dict[attr_name] = value
            
            result[field_name] = section_dict
        
        return result
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all important data paths."""
        base = self.system.data_dir
        return {
            "data_dir": base,
            "vector_store": base / "vector_store",
            "processed": base / "processed",
            "raw": base / "raw",
            "embeddings_cache": self.embedding.cache_dir,
            "logs": base / "logs",
            "config": Path.home() / ".nvme-rag" / "config.yml"
        }
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        paths = self.get_data_paths()
        for path in paths.values():
            if path.suffix not in [".yml", ".yaml", ".json"]:  # Skip files
                path.mkdir(parents=True, exist_ok=True)