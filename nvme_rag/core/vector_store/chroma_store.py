import chromadb
#from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import chromadb.errors  # Add this import for proper error handling

from nvme_rag.core.vector_store.base import VectorStoreBase, SearchResult, SearchQuery, DistanceMetric
from nvme_rag.core.vector_store.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from nvme_rag.core.models.document import ProcessedChunk

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStoreBase):
    """
    ChromaDB-based vector store implementation
    Provides high-performance vector storage and similarity search
    """
    
    def __init__(self, 
                 collection_name: str = "nvme_rag_chunks",
                 persist_directory: str = "data/vector_store",
                 embedding_config: EmbeddingConfig = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_config: Configuration for embedding generation
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        
        # Initialize ChromaDB client
        self._setup_chroma_client()
        
        # Get or create collection
        self._setup_collection()
        
        logger.info(f"ChromaDB vector store initialized with collection: {collection_name}")
    
    def _setup_chroma_client(self):
        """Setup ChromaDB client with persistence"""
        try:
            # Use the new ChromaDB client configuration (replaces deprecated Settings)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            logger.info(f"ChromaDB client initialized with persistence at: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    

    def _setup_collection(self):
        """Setup or get ChromaDB collection"""
        try:
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function()
                )
                logger.info(f"Using existing collection: {self.collection_name}")
            except chromadb.errors.NotFoundError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={"description": "NVME RAG chunks with semantic embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _get_embedding_function(self):
        """Get ChromaDB-compatible embedding function"""
        class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, generator: EmbeddingGenerator):
                self.generator = generator
            
            def __call__(self, texts: List[str]) -> List[List[float]]:
                embeddings = self.generator.generate_embeddings(texts)
                return embeddings.tolist()
        
        return CustomEmbeddingFunction(self.embedding_generator)
    
    def add_chunks(self, chunks: List[ProcessedChunk]) -> List[str]:
        """
        Add processed chunks to ChromaDB vector store
        
        Args:
            chunks: List of processed chunks to add
            
        Returns:
            List of chunk IDs that were added
        """
        if not chunks:
            return []
        
        try:
            # Prepare data for ChromaDB
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            
            # Prepare metadata for each chunk
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "parent_doc_id": chunk.parent_doc_id,
                    "section_header": chunk.section_header,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type,
                    "semantic_density": float(chunk.semantic_density),
                    "character_length": len(chunk.content),
                    "paragraph_count": chunk.metadata.get("paragraph_count", 1),
                    "extraction_method": chunk.metadata.get("extraction_method", "unknown"),
                    "processing_timestamp": chunk.metadata.get("processing_timestamp", datetime.now().isoformat()),
                    "source": chunk.metadata.get("source", "unknown")
                }
                metadatas.append(metadata)
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=chunk_ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform vector similarity search using ChromaDB
        
        Args:
            query: Search query configuration
            
        Returns:
            List of search results ranked by similarity
        """
        try:
            # Prepare ChromaDB query parameters
            where_clause = None
            if query.filters:
                where_clause = self._build_where_clause(query.filters)
            
            # Execute search
            results = self.collection.query(
                query_texts=[query.query_text],
                n_results=query.top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert ChromaDB results to SearchResult objects
            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    document = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    # Fix: Handle distance metric properly - check if it's an enum or string
                    distance_metric = query.distance_metric
                    if hasattr(distance_metric, 'value'):
                        # It's an enum, use the value
                        metric_value = distance_metric.value
                    else:
                        # It's already a string
                        metric_value = distance_metric

                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = 1.0 - distance if metric_value == "COSINE" else distance
                    
                    # Skip results below minimum score threshold
                    if query.min_score is not None and score < query.min_score:
                        continue
                    
                    # Reconstruct ProcessedChunk from stored data
                    chunk = ProcessedChunk(
                        content=document,
                        metadata=metadata,
                        chunk_id=chunk_id,
                        parent_doc_id=metadata.get("parent_doc_id", ""),
                        section_header=metadata.get("section_header", ""),
                        page_number=metadata.get("page_number", 0),
                        chunk_type=metadata.get("chunk_type", "text"),
                        semantic_density=metadata.get("semantic_density", 0.0)
                    )
                    
                    search_result = SearchResult(
                        chunk=chunk,
                        score=score,
                        distance=distance,
                        metadata=metadata if query.include_metadata else {}
                    )
                    
                    search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} results for query: {query.query_text[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, dict) and "$in" in value:
                # Handle $in operator for multiple values
                where_clause[key] = {"$in": value["$in"]}
            elif isinstance(value, dict) and "$gte" in value:
                # Handle greater than or equal
                where_clause[key] = {"$gte": value["$gte"]}
            elif isinstance(value, dict) and "$lte" in value:
                # Handle less than or equal
                where_clause[key] = {"$lte": value["$lte"]}
            else:
                # Direct equality
                where_clause[key] = value
        
        return where_clause
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ProcessedChunk]:
        """
        Retrieve chunk by ID from ChromaDB
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            ProcessedChunk if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                document = results["documents"][0]
                metadata = results["metadatas"][0]
                
                chunk = ProcessedChunk(
                    content=document,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    parent_doc_id=metadata.get("parent_doc_id", ""),
                    section_header=metadata.get("section_header", ""),
                    page_number=metadata.get("page_number", 0),
                    chunk_type=metadata.get("chunk_type", "text"),
                    semantic_density=metadata.get("semantic_density", 0.0)
                )
                
                return chunk
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk by ID {chunk_id}: {e}")
            return None
    
    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by IDs from ChromaDB
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Number of chunks successfully deleted
        """
        try:
            # Check which chunks exist before deletion
            existing_chunks = self.collection.get(ids=chunk_ids, include=[])
            existing_ids = existing_chunks["ids"]
            
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Deleted {len(existing_ids)} chunks from ChromaDB")
                return len(existing_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return 0
    
    def update_chunk(self, chunk_id: str, chunk: ProcessedChunk) -> bool:
        """
        Update existing chunk in ChromaDB
        
        Args:
            chunk_id: ID of chunk to update
            chunk: Updated chunk data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # ChromaDB doesn't have direct update, so we delete and re-add
            deleted_count = self.delete_chunks([chunk_id])
            if deleted_count > 0:
                added_ids = self.add_chunks([chunk])
                return len(added_ids) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get ChromaDB vector store statistics
        
        Returns:
            Dictionary with store statistics
        """
        try:
            # Get collection count
            count_result = self.collection.count()
            
            # Get a sample of chunks to analyze
            sample_results = self.collection.peek(limit=100)
            
            # Calculate statistics
            stats = {
                "total_chunks": count_result,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "embedding_model": self.embedding_generator.config.model_name,
                "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
                "sample_metadata": sample_results["metadatas"][:5] if sample_results["metadatas"] else []
            }
            
            # Analyze metadata if available
            if sample_results["metadatas"]:
                extraction_methods = {}
                chunk_types = {}
                parent_docs = set()
                
                for metadata in sample_results["metadatas"]:
                    # Count extraction methods
                    method = metadata.get("extraction_method", "unknown")
                    extraction_methods[method] = extraction_methods.get(method, 0) + 1
                    
                    # Count chunk types
                    chunk_type = metadata.get("chunk_type", "unknown")
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    # Count parent documents
                    parent_docs.add(metadata.get("parent_doc_id", "unknown"))
                
                stats.update({
                    "extraction_methods": extraction_methods,
                    "chunk_types": chunk_types,
                    "unique_documents": len(parent_docs)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def save_index(self, path: str) -> bool:
        """
        Save ChromaDB index (ChromaDB auto-persists)
        
        Args:
            path: Path to save index (not used in ChromaDB)
            
        Returns:
            True if successful
        """
        try:
            # ChromaDB automatically persists, but we can trigger explicit persistence
            self.client.persist()
            logger.info("ChromaDB index persisted")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """
        Load ChromaDB index (automatically loaded on client initialization)
        
        Args:
            path: Path to load index from (not used in ChromaDB)
            
        Returns:
            True if successful
        """
        try:
            # ChromaDB automatically loads persisted data
            stats = self.get_stats()
            logger.info(f"ChromaDB index loaded with {stats.get('total_chunks', 0)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset (delete and recreate) the collection
        
        Returns:
            True if successful
        """
        try:
            # Delete existing collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate collection
            self._setup_collection()
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'embedding_generator'):
            self.embedding_generator.cleanup()
        logger.info("ChromaDB vector store cleanup completed")