from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.models.document import ProcessedChunk

class DistanceMetric(Enum):
    """Distance metrics for vector similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

@dataclass
class SearchResult:
    """Result from vector search"""
    chunk: ProcessedChunk
    score: float
    distance: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chunk": self.chunk.to_dict(),
            "score": float(self.score),
            "distance": float(self.distance),
            "metadata": self.metadata
        }

@dataclass
class SearchQuery:
    """Vector search query configuration"""
    query_text: str
    top_k: int = 10
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    filters: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = None
    include_metadata: bool = True
    
class VectorStoreBase(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[ProcessedChunk]) -> List[str]:
        """Add processed chunks to vector store"""
        pass
    
    @abstractmethod
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform vector similarity search"""
        pass
    
    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ProcessedChunk]:
        """Retrieve chunk by ID"""
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete chunks by IDs, return count deleted"""
        pass
    
    @abstractmethod
    def update_chunk(self, chunk_id: str, chunk: ProcessedChunk) -> bool:
        """Update existing chunk"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> bool:
        """Save vector index to disk"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> bool:
        """Load vector index from disk"""
        pass