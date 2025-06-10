import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel
import hashlib
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from src.models.document import ProcessedChunk

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation
    """
    
    #for every embedding generating model, comment the one not being used and state results in the comments
    
    #model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  #TODO: experiment with different models
    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"   
    max_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    cache_dir: str = "data/embeddings_cache"
    device: str = "auto"

class EmbeddingGenerator:
    """
    Professional embedding generator with caching and batch processing
    Generates high-quality embeddings for vector search
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        self.embedding_cache = {}
        self._setup_models()
        self._setup_cache()
    
    def _setup_device(self) -> torch.device:
        """
        Setup optimal device for processing
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Embedding generator using device: {device}")
        return device
    
    def _setup_models(self):
        """
        Initialize embedding models
        """
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info("Embedding models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def _setup_cache(self):
        """
        Setup embedding cache if enabled
        """
        if self.config.cache_embeddings:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / f"{self._get_model_hash()}_embeddings.json"  #naming convention for cache files
            self._load_cache()
    
    def _get_model_hash(self) -> str:
        """
        Get hash for model configuration for cache identification
        """
        config_str = f"{self.config.model_name}_{self.config.max_length}_{self.config.normalize_embeddings}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _load_cache(self):
        """
        Load embedding cache from disk
        """
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                self.embedding_cache = {k: np.array(v) for k, v in cache_data.items()}
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """
        Save embedding cache to disk
        """
        if not self.config.cache_embeddings:
            return
        
        try:
            cache_data = {k: v.tolist() for k, v in self.embedding_cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """
        Get hash of text for caching
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _generate_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        """
        # Tokenize batch
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize if requested
            if self.config.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching and batching
        
        Args:
            texts: List of input texts
            
        Returns:
            numpy array of embeddings [num_texts, embedding_dim]
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if self.config.cache_embeddings and text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            
            for i in range(0, len(uncached_texts), self.config.batch_size):
                batch_texts = uncached_texts[i:i + self.config.batch_size]
                batch_indices = uncached_indices[i:i + self.config.batch_size]
                
                batch_embeddings = self._generate_embedding_batch(batch_texts)
                
                # Store in results and cache
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    result_idx = batch_indices[j]
                    embeddings[result_idx] = embedding
                    
                    # Cache the embedding
                    if self.config.cache_embeddings:
                        text_hash = self._get_text_hash(text)
                        self.embedding_cache[text_hash] = embedding
            
            # Save cache after generating new embeddings
            self._save_cache()
        
        return np.array(embeddings)
    
    def generate_chunk_embeddings(self, chunks: List[ProcessedChunk]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for processed chunks
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Dictionary mapping chunk_id to embedding
        """
        if not chunks:
            return {}
        
        # Extract texts and chunk IDs
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create chunk ID to embedding mapping
        chunk_embeddings = {}
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            chunk_embeddings[chunk_id] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunk_embeddings
    
    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query_text: Query string
            
        Returns:
            Query embedding as numpy array
        """
        embeddings = self.generate_embeddings([query_text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model
        """
        # Generate a test embedding to get dimension
        test_embedding = self.generate_embeddings(["test"])
        return test_embedding.shape[1] if len(test_embedding) > 0 else 0
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        """
        return asdict(self.config)
    
    def cleanup(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Embedding generator cleanup completed")