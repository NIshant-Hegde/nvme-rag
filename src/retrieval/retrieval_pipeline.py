import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

from src.vector_store.base import SearchQuery, SearchResult, DistanceMetric
from src.vector_store.chroma_store import ChromaVectorStore
from src.llm.ollama_client import OllamaClient, ChatMessage, OllamaConfig
from src.models.document import ProcessedChunk

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    FILTERED = "filtered"
    RERANKED = "reranked"

@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline"""
    strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_ONLY
    top_k: int = 10
    min_score: float = 0.7
    rerank_top_k: int = 20
    enable_query_enhancement: bool = True
    enable_context_filtering: bool = True
    max_context_length: int = 4000
    include_section_context: bool = True

@dataclass
class QueryContext:
    """Context for query processing"""
    original_query: str
    enhanced_query: str
    chat_history: List[ChatMessage]
    filters: Dict[str, Any]
    user_preferences: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Result from retrieval pipeline"""
    query_context: QueryContext
    search_results: List[SearchResult]
    context_chunks: List[ProcessedChunk]
    total_context_length: int
    retrieval_stats: Dict[str, Any]

class QueryEnhancer:
    """
    Enhances queries using LLM for better retrieval
    Implements query rewriting and expansion techniques
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
    
    def enhance_query(self, query: str, chat_history: List[ChatMessage] = None) -> str:
        """
        Enhance query using chat history and context understanding
        
        Args:
            query: Original user query
            chat_history: Previous conversation messages
            
        Returns:
            Enhanced query string
        """
        try:
            # Build context from chat history
            context_str = ""
            if chat_history:
                recent_messages = chat_history[-4:]  # Last 4 messages for context
                context_str = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
            
            # Create enhancement prompt
            enhancement_prompt = self._build_enhancement_prompt(query, context_str)
            
            # Get enhanced query from LLM
            messages = [ChatMessage(role="user", content=enhancement_prompt)]
            response = self.llm_client.chat(messages)
            
            enhanced_query = response.content.strip()
            
            # Fallback to original if enhancement fails
            if not enhanced_query or len(enhanced_query) < 3:
                enhanced_query = query
            
            logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.warning(f"Query enhancement failed, using original: {e}")
            return query
    
    def _build_enhancement_prompt(self, query: str, context: str) -> str:
        """Build prompt for query enhancement"""
        prompt = f"""You are a query enhancement specialist for a technical document retrieval system. Your task is to improve search queries to better find relevant information.

Given the user's query and conversation context, rewrite the query to be more specific and likely to retrieve relevant technical information.

Guidelines:
- Expand abbreviations and technical terms
- Add relevant technical context
- Replace vague pronouns with specific terms
- Keep the core intent unchanged
- Make it more searchable while staying concise

Conversation Context:
{context}

Original Query: {query}

Enhanced Query (return only the enhanced query, no explanation):"""
        
        return prompt
    
    def generate_subqueries(self, query: str) -> List[str]:
        """
        Generate multiple sub-queries for comprehensive retrieval
        
        Args:
            query: Original query
            
        Returns:
            List of sub-queries
        """
        try:
            subquery_prompt = f"""Break down this technical query into 2-3 focused sub-queries that together would comprehensively answer the original question.

Original Query: {query}

Generate focused sub-queries (one per line, no numbering or bullets):"""
            
            messages = [ChatMessage(role="user", content=subquery_prompt)]
            response = self.llm_client.chat(messages)
            
            # Parse sub-queries from response
            subqueries = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and len(line) > 3:
                    # Clean up any numbering or bullet points
                    cleaned = re.sub(r'^[\d\.\-\*\+]\s*', '', line)
                    if cleaned:
                        subqueries.append(cleaned)
            
            # Include original query if no subqueries generated
            if not subqueries:
                subqueries = [query]
            
            logger.info(f"Generated {len(subqueries)} sub-queries for: {query}")
            return subqueries
            
        except Exception as e:
            logger.warning(f"Sub-query generation failed: {e}")
            return [query]

class ContextFilter:
    """
    Filters and ranks retrieved context for relevance
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
    
    def filter_relevant_chunks(self, chunks: List[ProcessedChunk], query: str, max_chunks: int = 5) -> List[ProcessedChunk]:
        """
        Filter chunks for relevance to query using LLM
        
        Args:
            chunks: List of retrieved chunks
            query: Original query
            max_chunks: Maximum chunks to return
            
        Returns:
            Filtered list of most relevant chunks
        """
        if len(chunks) <= max_chunks:
            return chunks
        
        try:
            # Create relevance scoring prompt
            chunks_text = "\n\n".join([
                f"[CHUNK {i+1}]\nSection: {chunk.section_header}\nContent: {chunk.content[:500]}..."
                for i, chunk in enumerate(chunks[:10])  # Limit to first 10 for LLM processing
            ])
            
            relevance_prompt = f"""Rate the relevance of each chunk to the user's query. Respond with only the chunk numbers (1-{min(len(chunks), 10)}) of the {max_chunks} most relevant chunks, separated by commas.

Query: {query}

Chunks:
{chunks_text}

Most relevant chunk numbers (top {max_chunks}):"""
            
            messages = [ChatMessage(role="user", content=relevance_prompt)]
            response = self.llm_client.chat(messages)
            
            # Parse chunk numbers from response
            relevant_indices = []
            numbers = re.findall(r'\d+', response.content)
            for num_str in numbers[:max_chunks]:
                try:
                    idx = int(num_str) - 1  # Convert to 0-based index
                    if 0 <= idx < len(chunks):
                        relevant_indices.append(idx)
                except ValueError:
                    continue
            
            # Return filtered chunks, fallback to score-based if parsing fails
            if relevant_indices:
                filtered_chunks = [chunks[i] for i in relevant_indices]
                logger.info(f"LLM filtered {len(chunks)} chunks to {len(filtered_chunks)}")
                return filtered_chunks
            else:
                logger.warning("LLM filtering failed, using top chunks by score")
                return chunks[:max_chunks]
                
        except Exception as e:
            logger.warning(f"Context filtering failed: {e}")
            return chunks[:max_chunks]

class RetrievalPipeline:
    """
    Main retrieval pipeline that orchestrates query enhancement,
    vector search, and context filtering
    """
    
    def __init__(self, 
                 vector_store: ChromaVectorStore,
                 llm_client: OllamaClient,
                 config: RetrievalConfig = None):
        """
        Initialize retrieval pipeline
        
        Args:
            vector_store: ChromaDB vector store
            llm_client: Ollama LLM client
            config: Retrieval configuration
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or RetrievalConfig()
        
        # Initialize components
        self.query_enhancer = QueryEnhancer(llm_client)
        self.context_filter = ContextFilter(llm_client)
        
        logger.info("Retrieval pipeline initialized")
    
    def retrieve(self, 
                 query: str, 
                 chat_history: List[ChatMessage] = None,
                 filters: Dict[str, Any] = None,
                 user_preferences: Dict[str, Any] = None) -> RetrievalResult:
        """
        Main retrieval method with full pipeline
        
        Args:
            query: User query
            chat_history: Previous conversation
            filters: Metadata filters
            user_preferences: User preferences for retrieval
            
        Returns:
            RetrievalResult with context and metadata
        """
        logger.info(f"Starting retrieval for query: {query}")
        
        # Step 1: Query Enhancement
        enhanced_query = query
        if self.config.enable_query_enhancement:
            enhanced_query = self.query_enhancer.enhance_query(query, chat_history)
        
        # Create query context
        query_context = QueryContext(
            original_query=query,
            enhanced_query=enhanced_query,
            chat_history=chat_history or [],
            filters=filters or {},
            user_preferences=user_preferences or {}
        )
        
        # Step 2: Vector Search
        search_results = self._perform_search(enhanced_query, filters)
        
        # Step 3: Context Filtering and Ranking
        context_chunks = self._process_search_results(search_results, query)
        
        # Step 4: Context Length Management
        final_chunks, total_length = self._manage_context_length(context_chunks)
        
        # Step 5: Generate retrieval statistics
        retrieval_stats = self._generate_stats(search_results, final_chunks)
        
        result = RetrievalResult(
            query_context=query_context,
            search_results=search_results,
            context_chunks=final_chunks,
            total_context_length=total_length,
            retrieval_stats=retrieval_stats
        )
        
        logger.info(f"Retrieval completed: {len(final_chunks)} chunks, {total_length} chars")
        return result
    
    def _perform_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform vector search with strategy-specific logic"""
        try:
            if self.config.strategy == RetrievalStrategy.SEMANTIC_ONLY:
                return self._semantic_search(query, filters)
            elif self.config.strategy == RetrievalStrategy.HYBRID:
                return self._hybrid_search(query, filters)
            elif self.config.strategy == RetrievalStrategy.FILTERED:
                return self._filtered_search(query, filters)
            elif self.config.strategy == RetrievalStrategy.RERANKED:
                return self._reranked_search(query, filters)
            else:
                return self._semantic_search(query, filters)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _semantic_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform basic semantic search"""
        search_query = SearchQuery(
            query_text=query,
            top_k=self.config.top_k,
            distance_metric=DistanceMetric.COSINE,
            filters=filters,
            min_score=self.config.min_score
        )
        
        return self.vector_store.search(search_query)
    
    def _hybrid_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform hybrid search using multiple strategies"""
        # Generate sub-queries for diverse retrieval
        subqueries = self.query_enhancer.generate_subqueries(query)
        
        all_results = []
        seen_chunk_ids = set()
        
        # Search with each sub-query
        for subquery in subqueries:
            search_query = SearchQuery(
                query_text=subquery,
                top_k=self.config.top_k // len(subqueries),
                distance_metric=DistanceMetric.COSINE,
                filters=filters,
                min_score=self.config.min_score * 0.8  # Lower threshold for sub-queries
            )
            
            results = self.vector_store.search(search_query)
            
            # Deduplicate results
            for result in results:
                if result.chunk.chunk_id not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk.chunk_id)
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:self.config.top_k]
    
    def _filtered_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search with enhanced filtering"""
        # Start with broader search
        search_query = SearchQuery(
            query_text=query,
            top_k=self.config.top_k * 2,  # Get more results to filter
            distance_metric=DistanceMetric.COSINE,
            filters=filters,
            min_score=self.config.min_score * 0.7  # Lower initial threshold
        )
        
        results = self.vector_store.search(search_query)
        
        # Apply additional filtering logic here
        # For now, return top results
        return results[:self.config.top_k]
    
    def _reranked_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search with LLM-based reranking"""
        # Get more results for reranking
        search_query = SearchQuery(
            query_text=query,
            top_k=self.config.rerank_top_k,
            distance_metric=DistanceMetric.COSINE,
            filters=filters,
            min_score=self.config.min_score * 0.6
        )
        
        results = self.vector_store.search(search_query)
        
        # Always call filter_relevant_chunks for test expectation
        chunks = [result.chunk for result in results]
        reranked_chunks = self.context_filter.filter_relevant_chunks(
            chunks, query, self.config.top_k
        )
        
        # Reconstruct search results with reranked order
        chunk_id_to_result = {result.chunk.chunk_id: result for result in results}
        reranked_results = []
        
        for chunk in reranked_chunks:
            if chunk.chunk_id in chunk_id_to_result:
                reranked_results.append(chunk_id_to_result[chunk.chunk_id])
        
        return reranked_results
    
    def _process_search_results(self, search_results: List[SearchResult], query: str) -> List[ProcessedChunk]:
        """Process search results to extract context chunks"""
        chunks = [result.chunk for result in search_results]
        
        if self.config.enable_context_filtering and len(chunks) > self.config.top_k:
            # Use LLM to filter most relevant chunks
            chunks = self.context_filter.filter_relevant_chunks(chunks, query, self.config.top_k)
        
        return chunks
    
    def _manage_context_length(self, chunks: List[ProcessedChunk]) -> Tuple[List[ProcessedChunk], int]:
        """Manage context length to fit within limits"""
        total_length = 0
        final_chunks = []
        
        for chunk in chunks:
            chunk_length = len(chunk.content)
            
            # Add section context if enabled
            context_addition = 0
            if self.config.include_section_context:
                context_addition = len(f"\n\nSection: {chunk.section_header}\n")
            
            # Ensure we never exceed max_context_length
            if total_length + chunk_length + context_addition > self.config.max_context_length:
                # Try to fit partial chunk if there's significant remaining space
                remaining_space = self.config.max_context_length - total_length - context_addition
                if remaining_space > 200:  # Minimum meaningful chunk size
                    truncated_content = chunk.content[:remaining_space] + "..."
                    truncated_chunk = ProcessedChunk(
                        content=truncated_content,
                        metadata=chunk.metadata,
                        chunk_id=chunk.chunk_id + "_truncated",
                        parent_doc_id=chunk.parent_doc_id,
                        section_header=chunk.section_header,
                        page_number=chunk.page_number,
                        chunk_type=chunk.chunk_type,
                        semantic_density=chunk.semantic_density
                    )
                    final_chunks.append(truncated_chunk)
                    total_length += len(truncated_content) + context_addition
                break
            else:
                final_chunks.append(chunk)
                total_length += chunk_length + context_addition
        
        # Ensure total_length does not exceed max_context_length
        if total_length > self.config.max_context_length and final_chunks:
            # Remove last chunk if it caused overflow
            last_chunk = final_chunks.pop()
            if self.config.include_section_context:
                total_length -= len(last_chunk.content) + len(f"\n\nSection: {last_chunk.section_header}\n")
            else:
                total_length -= len(last_chunk.content)
        
        return final_chunks, total_length
    
    def _generate_stats(self, search_results: List[SearchResult], final_chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Generate retrieval statistics"""
        return {
            "search_results_count": len(search_results),
            "final_chunks_count": len(final_chunks),
            "average_score": sum(r.score for r in search_results) / len(search_results) if search_results else 0,
            "score_range": {
                "min": min(r.score for r in search_results) if search_results else 0,
                "max": max(r.score for r in search_results) if search_results else 0
            },
            "sections_covered": list(set(chunk.section_header for chunk in final_chunks)),
            "documents_covered": list(set(chunk.parent_doc_id for chunk in final_chunks)),
            "strategy_used": self.config.strategy.value,
            "filters_applied": bool(self.config.enable_context_filtering)
        }
    
    def update_config(self, **kwargs):
        """Update retrieval configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated retrieval config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return asdict(self.config)