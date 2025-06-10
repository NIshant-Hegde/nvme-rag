import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import requests

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.ollama_client import OllamaClient, OllamaConfig, ChatMessage, LLMResponse, OllamaModel
from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline, RetrievalConfig, QueryEnhancer, ContextFilter, 
    RetrievalStrategy, QueryContext, RetrievalResult
)
from src.models.document import ProcessedChunk

class TestOllamaClient(unittest.TestCase):
    """Test Ollama LLM client functionality"""
    
    def setUp(self):
        self.config = OllamaConfig(
            base_url="http://localhost:11434",
            model=OllamaModel.MISTRAL_7B.value,
            temperature=0.1,
            timeout=30
        )
    
    @patch('src.llm.ollama_client.requests.Session')
    def test_ollama_client_initialization(self, mock_session_class):
        """Test Ollama client initialization and connection verification"""
        # Mock session and responses
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock successful connection
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "mistral:7b"}, {"name": "llama2:7b"}]
        }
        mock_session.get.return_value = mock_response
        
        # Test initialization
        with patch.object(OllamaClient, '_pull_model'):
            client = OllamaClient(self.config)
            
            self.assertEqual(client.config.model, "mistral:7b")
            self.assertEqual(client.config.base_url, "http://localhost:11434")
            mock_session.get.assert_called()
    
    @patch('src.llm.ollama_client.requests.Session')
    def test_chat_functionality(self, mock_session_class):
        """Test chat functionality with non-streaming response"""
        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock connection verification
        mock_tags_response = Mock()
        mock_tags_response.json.return_value = {"models": [{"name": "mistral:7b"}]}
        
        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.json.return_value = {
            "message": {"content": "This is a test response"},
            "model": "mistral:7b",
            "done": True,
            "total_duration": 1000000,
            "eval_count": 10
        }
        
        # Configure session.get and session.post
        mock_session.get.return_value = mock_tags_response
        mock_session.post.return_value = mock_chat_response
        
        with patch.object(OllamaClient, '_pull_model'):
            client = OllamaClient(self.config)
            
            # Test chat
            messages = [ChatMessage(role="user", content="Hello, how are you?")]
            response = client.chat(messages)
            
            self.assertIsInstance(response, LLMResponse)
            self.assertEqual(response.content, "This is a test response")
            self.assertEqual(response.model, "mistral:7b")
            self.assertTrue(response.done)
    
    @patch('src.llm.ollama_client.requests.Session')
    def test_streaming_chat(self, mock_session_class):
        """Test streaming chat functionality"""
        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock connection verification
        mock_tags_response = Mock()
        mock_tags_response.json.return_value = {"models": [{"name": "mistral:7b"}]}
        mock_session.get.return_value = mock_tags_response
        
        # Mock streaming response
        mock_stream_response = Mock()
        mock_stream_lines = [
            b'{"message": {"content": "Hello"}, "done": false}',
            b'{"message": {"content": " there"}, "done": false}',
            b'{"message": {"content": "!"}, "done": true, "total_duration": 1000000}'
        ]
        mock_stream_response.iter_lines.return_value = mock_stream_lines
        mock_session.post.return_value = mock_stream_response
        
        with patch.object(OllamaClient, '_pull_model'):
            client = OllamaClient(self.config)
            
            # Test streaming chat
            messages = [ChatMessage(role="user", content="Hello")]
            response_stream = client.chat(messages, stream=True)
            
            # Collect streaming responses
            responses = list(response_stream)
            
            self.assertEqual(len(responses), 3)
            self.assertEqual(responses[0].content, "Hello")
            self.assertEqual(responses[1].content, " there")
            self.assertEqual(responses[2].content, "!")
            self.assertTrue(responses[2].done)
    
    @patch('src.llm.ollama_client.requests.Session')
    def test_generate_completion(self, mock_session_class):
        """Test simple completion generation"""
        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock responses
        mock_tags_response = Mock()
        mock_tags_response.json.return_value = {"models": [{"name": "mistral:7b"}]}
        
        mock_completion_response = Mock()
        mock_completion_response.json.return_value = {
            "message": {"content": "Completion response"},
            "model": "mistral:7b",
            "done": True
        }
        
        mock_session.get.return_value = mock_tags_response
        mock_session.post.return_value = mock_completion_response
        
        with patch.object(OllamaClient, '_pull_model'):
            client = OllamaClient(self.config)
            
            response = client.generate_completion("Complete this sentence")
            
            self.assertIsInstance(response, LLMResponse)
            self.assertEqual(response.content, "Completion response")
    
    @patch('src.llm.ollama_client.requests.Session')
    def test_health_check(self, mock_session_class):
        """Test health check functionality"""
        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock successful health check
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "mistral:7b"}, {"name": "llama2:7b"}]
        }
        mock_session.get.return_value = mock_response
        
        with patch.object(OllamaClient, '_pull_model'):
            client = OllamaClient(self.config)
            
            health = client.health_check()
            
            self.assertEqual(health["status"], "healthy")
            self.assertIn("response_time_seconds", health)
            self.assertTrue(health["model_available"])
            self.assertEqual(health["total_models"], 2)
    
    def test_chat_message_creation(self):
        """Test ChatMessage creation and serialization"""
        message = ChatMessage(role="user", content="Test message")
        
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Test message")
        
        message_dict = message.to_dict()
        expected = {"role": "user", "content": "Test message"}
        self.assertEqual(message_dict, expected)
    
    def test_llm_response_creation(self):
        """Test LLMResponse creation and serialization"""
        response = LLMResponse(
            content="Test response",
            model="mistral:7b",
            total_duration=1000000,
            eval_count=10,
            done=True
        )
        
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "mistral:7b")
        self.assertTrue(response.done)
        
        response_dict = response.to_dict()
        self.assertIn("content", response_dict)
        self.assertIn("model", response_dict)
        self.assertIn("done", response_dict)

class TestQueryEnhancer(unittest.TestCase):
    """Test query enhancement functionality"""
    
    def setUp(self):
        # Mock LLM client
        self.mock_llm_client = Mock()
        self.query_enhancer = QueryEnhancer(self.mock_llm_client)
    
    def test_enhance_query_basic(self):
        """Test basic query enhancement"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Enhanced query about NVMe controller implementation details"
        self.mock_llm_client.chat.return_value = mock_response
        
        original_query = "How does NVMe work?"
        enhanced_query = self.query_enhancer.enhance_query(original_query)
        
        self.assertEqual(enhanced_query, "Enhanced query about NVMe controller implementation details")
        self.mock_llm_client.chat.assert_called_once()
    
    def test_enhance_query_with_history(self):
        """Test query enhancement with chat history"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Enhanced query with context"
        self.mock_llm_client.chat.return_value = mock_response
        
        chat_history = [
            ChatMessage(role="user", content="Tell me about storage"),
            ChatMessage(role="assistant", content="Storage refers to data persistence...")
        ]
        
        original_query = "What about the interface?"
        enhanced_query = self.query_enhancer.enhance_query(original_query, chat_history)
        
        self.assertEqual(enhanced_query, "Enhanced query with context")
        self.mock_llm_client.chat.assert_called_once()
    
    def test_enhance_query_fallback(self):
        """Test query enhancement fallback when LLM fails"""
        # Mock LLM failure
        self.mock_llm_client.chat.side_effect = Exception("LLM error")
        
        original_query = "Test query"
        enhanced_query = self.query_enhancer.enhance_query(original_query)
        
        # Should fallback to original query
        self.assertEqual(enhanced_query, original_query)
    
    def test_generate_subqueries(self):
        """Test sub-query generation"""
        # Mock LLM response with sub-queries
        mock_response = Mock()
        mock_response.content = """What are NVMe controllers?
How do NVMe controllers work?
What are the benefits of NVMe controllers?"""
        self.mock_llm_client.chat.return_value = mock_response
        
        original_query = "Tell me about NVMe controllers"
        subqueries = self.query_enhancer.generate_subqueries(original_query)
        
        self.assertEqual(len(subqueries), 3)
        self.assertIn("What are NVMe controllers?", subqueries)
        self.assertIn("How do NVMe controllers work?", subqueries)
        self.assertIn("What are the benefits of NVMe controllers?", subqueries)

class TestContextFilter(unittest.TestCase):
    """Test context filtering functionality"""
    
    def setUp(self):
        # Mock LLM client
        self.mock_llm_client = Mock()
        self.context_filter = ContextFilter(self.mock_llm_client)
        
        # Create test chunks
        self.test_chunks = [
            ProcessedChunk(
                content="NVMe controllers manage data transfer operations.",
                metadata={},
                chunk_id="chunk_1",
                parent_doc_id="doc_1",
                section_header="NVMe Controllers",
                page_number=1,
                chunk_type="text",
                semantic_density=0.8
            ),
            ProcessedChunk(
                content="PCIe interface provides high-speed communication.",
                metadata={},
                chunk_id="chunk_2",
                parent_doc_id="doc_1",
                section_header="PCIe Interface",
                page_number=2,
                chunk_type="text",
                semantic_density=0.9
            ),
            ProcessedChunk(
                content="Storage capacity considerations for enterprise use.",
                metadata={},
                chunk_id="chunk_3",
                parent_doc_id="doc_1",
                section_header="Storage Capacity",
                page_number=3,
                chunk_type="text",
                semantic_density=0.7
            )
        ]
    
    def test_filter_relevant_chunks(self):
        """Test filtering chunks for relevance"""
        # Mock LLM response with relevant chunk numbers
        mock_response = Mock()
        mock_response.content = "1, 3"  # Choose chunks 1 and 3 as most relevant
        self.mock_llm_client.chat.return_value = mock_response
        
        query = "How do NVMe controllers work?"
        filtered_chunks = self.context_filter.filter_relevant_chunks(
            self.test_chunks, query, max_chunks=2
        )
        
        self.assertEqual(len(filtered_chunks), 2)
        self.assertEqual(filtered_chunks[0].chunk_id, "chunk_1")  # First chunk
        self.assertEqual(filtered_chunks[1].chunk_id, "chunk_3")  # Third chunk
    
    def test_filter_relevant_chunks_fallback(self):
        """Test filtering fallback when LLM fails"""
        # Mock LLM failure
        self.mock_llm_client.chat.side_effect = Exception("LLM error")
        
        query = "Test query"
        filtered_chunks = self.context_filter.filter_relevant_chunks(
            self.test_chunks, query, max_chunks=2
        )
        
        # Should fallback to top chunks by order
        self.assertEqual(len(filtered_chunks), 2)
        self.assertEqual(filtered_chunks[0].chunk_id, "chunk_1")
        self.assertEqual(filtered_chunks[1].chunk_id, "chunk_2")
    
    def test_filter_fewer_chunks_than_max(self):
        """Test filtering when fewer chunks than max requested"""
        mock_response = Mock()
        mock_response.content = "1, 2"
        self.mock_llm_client.chat.return_value = mock_response
        
        query = "Test query"
        filtered_chunks = self.context_filter.filter_relevant_chunks(
            self.test_chunks[:2], query, max_chunks=5  # Request more than available
        )
        
        # Should return all available chunks
        self.assertEqual(len(filtered_chunks), 2)

class TestRetrievalPipeline(unittest.TestCase):
    """Test complete retrieval pipeline functionality"""
    
    def setUp(self):
        # Mock vector store
        self.mock_vector_store = Mock()
        
        # Mock LLM client
        self.mock_llm_client = Mock()
        
        # Create retrieval config
        self.config = RetrievalConfig(
            strategy=RetrievalStrategy.SEMANTIC_ONLY,
            top_k=5,
            min_score=0.7,
            enable_query_enhancement=True,
            enable_context_filtering=True
        )
        
        # Create test search results
        self.test_chunks = [
            ProcessedChunk(
                content="NVMe controller implementation details.",
                metadata={"score": 0.95},
                chunk_id="chunk_1",
                parent_doc_id="doc_1",
                section_header="NVMe Controllers",
                page_number=1,
                chunk_type="text",
                semantic_density=0.8
            ),
            ProcessedChunk(
                content="PCIe interface specifications.",
                metadata={"score": 0.88},
                chunk_id="chunk_2",
                parent_doc_id="doc_1",
                section_header="PCIe Interface",
                page_number=2,
                chunk_type="text",
                semantic_density=0.9
            )
        ]
        
        from src.vector_store.base import SearchResult
        self.test_search_results = [
            SearchResult(
                chunk=chunk,
                score=chunk.metadata["score"],
                distance=1.0 - chunk.metadata["score"],
                metadata=chunk.metadata
            )
            for chunk in self.test_chunks
        ]
    
    @patch('src.retrieval.retrieval_pipeline.QueryEnhancer')
    @patch('src.retrieval.retrieval_pipeline.ContextFilter')
    def test_retrieval_pipeline_initialization(self, mock_context_filter, mock_query_enhancer):
        """Test retrieval pipeline initialization"""
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        self.assertEqual(pipeline.vector_store, self.mock_vector_store)
        self.assertEqual(pipeline.llm_client, self.mock_llm_client)
        self.assertEqual(pipeline.config.strategy, RetrievalStrategy.SEMANTIC_ONLY)
        mock_query_enhancer.assert_called_once()
        mock_context_filter.assert_called_once()
    
    @patch('src.retrieval.retrieval_pipeline.QueryEnhancer')
    @patch('src.retrieval.retrieval_pipeline.ContextFilter')
    def test_semantic_search_strategy(self, mock_context_filter, mock_query_enhancer):
        """Test semantic search strategy"""
        # Mock query enhancer
        mock_enhancer_instance = Mock()
        mock_enhancer_instance.enhance_query.return_value = "enhanced query"
        mock_query_enhancer.return_value = mock_enhancer_instance
        
        # Mock context filter
        mock_filter_instance = Mock()
        mock_filter_instance.filter_relevant_chunks.return_value = self.test_chunks
        mock_context_filter.return_value = mock_filter_instance
        
        # Mock vector store search
        self.mock_vector_store.search.return_value = self.test_search_results
        
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        # Test retrieval
        result = pipeline.retrieve("test query")
        
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.query_context.original_query, "test query")
        self.assertEqual(result.query_context.enhanced_query, "enhanced query")
        self.assertEqual(len(result.context_chunks), 2)
        self.assertGreater(result.total_context_length, 0)
        
        # Verify method calls
        mock_enhancer_instance.enhance_query.assert_called_once()
        self.mock_vector_store.search.assert_called_once()
    
    @patch('src.retrieval.retrieval_pipeline.QueryEnhancer')
    @patch('src.retrieval.retrieval_pipeline.ContextFilter')
    def test_hybrid_search_strategy(self, mock_context_filter, mock_query_enhancer):
        """Test hybrid search strategy"""
        # Configure for hybrid search
        self.config.strategy = RetrievalStrategy.HYBRID
        
        # Mock query enhancer
        mock_enhancer_instance = Mock()
        mock_enhancer_instance.enhance_query.return_value = "enhanced query"
        mock_enhancer_instance.generate_subqueries.return_value = [
            "subquery 1", "subquery 2"
        ]
        mock_query_enhancer.return_value = mock_enhancer_instance
        
        # Mock context filter
        mock_filter_instance = Mock()
        mock_filter_instance.filter_relevant_chunks.return_value = self.test_chunks
        mock_context_filter.return_value = mock_filter_instance
        
        # Mock vector store search (return different results for different subqueries)
        self.mock_vector_store.search.side_effect = [
            [self.test_search_results[0]],  # First subquery result
            [self.test_search_results[1]]   # Second subquery result
        ]
        
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        # Test hybrid retrieval
        result = pipeline.retrieve("test query")
        
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(len(result.search_results), 2)  # Combined results
        
        # Verify multiple searches were called (one per subquery)
        self.assertEqual(self.mock_vector_store.search.call_count, 2)
        mock_enhancer_instance.generate_subqueries.assert_called_once()
    
    @patch('src.retrieval.retrieval_pipeline.QueryEnhancer')
    @patch('src.retrieval.retrieval_pipeline.ContextFilter')
    def test_reranked_search_strategy(self, mock_context_filter, mock_query_enhancer):
        """Test reranked search strategy"""
        # Configure for reranked search
        self.config.strategy = RetrievalStrategy.RERANKED
        self.config.rerank_top_k = 10
        
        # Mock query enhancer
        mock_enhancer_instance = Mock()
        mock_enhancer_instance.enhance_query.return_value = "enhanced query"
        mock_query_enhancer.return_value = mock_enhancer_instance
        
        # Mock context filter with reranking
        mock_filter_instance = Mock()
        mock_filter_instance.filter_relevant_chunks.return_value = [
            self.test_chunks[1], self.test_chunks[0]  # Reranked order
        ]
        mock_context_filter.return_value = mock_filter_instance
        
        # Mock vector store search
        self.mock_vector_store.search.return_value = self.test_search_results
        
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        # Test reranked retrieval
        result = pipeline.retrieve("test query")
        
        self.assertIsInstance(result, RetrievalResult)
        # Results should be reranked by context filter
        mock_filter_instance.filter_relevant_chunks.assert_called_once()
    
    @patch('src.retrieval.retrieval_pipeline.QueryEnhancer')
    @patch('src.retrieval.retrieval_pipeline.ContextFilter')
    def test_context_length_management(self, mock_context_filter, mock_query_enhancer):
        """Test context length management"""
        # Create large chunks that exceed context limit
        large_chunks = [
            ProcessedChunk(
                content="A" * 2000,  # Large content
                metadata={},
                chunk_id=f"large_chunk_{i}",
                parent_doc_id="doc_1",
                section_header="Large Section",
                page_number=i,
                chunk_type="text",
                semantic_density=0.8
            )
            for i in range(5)
        ]
        
        # Configure small context limit
        self.config.max_context_length = 3000
        
        # Mock components
        mock_enhancer_instance = Mock()
        mock_enhancer_instance.enhance_query.return_value = "enhanced query"
        mock_query_enhancer.return_value = mock_enhancer_instance
        
        mock_filter_instance = Mock()
        mock_filter_instance.filter_relevant_chunks.return_value = large_chunks
        mock_context_filter.return_value = mock_filter_instance
        
        from src.vector_store.base import SearchResult
        large_search_results = [
            SearchResult(chunk=chunk, score=0.9, distance=0.1, metadata={})
            for chunk in large_chunks
        ]
        self.mock_vector_store.search.return_value = large_search_results
        
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        # Test retrieval with context length management
        result = pipeline.retrieve("test query")
        
        # Should limit chunks to fit within context length
        self.assertLessEqual(result.total_context_length, self.config.max_context_length)
        self.assertLess(len(result.context_chunks), 5)  # Not all chunks should fit
    
    @patch('src.retrieval.retrieval_pipeline.QueryEnhancer')
    @patch('src.retrieval.retrieval_pipeline.ContextFilter')
    def test_retrieval_with_filters(self, mock_context_filter, mock_query_enhancer):
        """Test retrieval with metadata filters"""
        # Mock components
        mock_enhancer_instance = Mock()
        mock_enhancer_instance.enhance_query.return_value = "enhanced query"
        mock_query_enhancer.return_value = mock_enhancer_instance
        
        mock_filter_instance = Mock()
        mock_filter_instance.filter_relevant_chunks.return_value = self.test_chunks
        mock_context_filter.return_value = mock_filter_instance
        
        self.mock_vector_store.search.return_value = self.test_search_results
        
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        # Test retrieval with filters
        filters = {"section_header": "NVMe Controllers"}
        result = pipeline.retrieve("test query", filters=filters)
        
        self.assertEqual(result.query_context.filters, filters)
        
        # Verify search was called with filters
        search_call_args = self.mock_vector_store.search.call_args[0][0]
        self.assertEqual(search_call_args.filters, filters)
    
    def test_retrieval_config_updates(self):
        """Test updating retrieval configuration"""
        pipeline = RetrievalPipeline(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            config=self.config
        )
        
        # Test config updates
        pipeline.update_config(top_k=10, min_score=0.8)
        
        self.assertEqual(pipeline.config.top_k, 10)
        self.assertEqual(pipeline.config.min_score, 0.8)
        
        # Test getting config
        config_dict = pipeline.get_config()
        self.assertIn("top_k", config_dict)
        self.assertIn("min_score", config_dict)
        self.assertEqual(config_dict["top_k"], 10)

class TestRetrievalDataStructures(unittest.TestCase):
    """Test retrieval data structures"""
    
    def test_query_context_creation(self):
        """Test QueryContext creation"""
        chat_history = [ChatMessage(role="user", content="Previous question")]
        filters = {"section": "test"}
        preferences = {"language": "en"}
        
        context = QueryContext(
            original_query="original",
            enhanced_query="enhanced",
            chat_history=chat_history,
            filters=filters,
            user_preferences=preferences
        )
        
        self.assertEqual(context.original_query, "original")
        self.assertEqual(context.enhanced_query, "enhanced")
        self.assertEqual(len(context.chat_history), 1)
        self.assertEqual(context.filters, filters)
        self.assertEqual(context.user_preferences, preferences)
    
    def test_retrieval_result_creation(self):
        """Test RetrievalResult creation"""
        query_context = QueryContext(
            original_query="test",
            enhanced_query="enhanced test",
            chat_history=[],
            filters={},
            user_preferences={}
        )
        
        chunks = [
            ProcessedChunk(
                content="Test content",
                metadata={},
                chunk_id="test_chunk",
                parent_doc_id="test_doc",
                section_header="Test Section",
                page_number=1,
                chunk_type="text",
                semantic_density=0.8
            )
        ]
        
        stats = {"total_results": 1, "strategy": "semantic"}
        
        result = RetrievalResult(
            query_context=query_context,
            search_results=[],
            context_chunks=chunks,
            total_context_length=100,
            retrieval_stats=stats
        )
        
        self.assertEqual(result.query_context.original_query, "test")
        self.assertEqual(len(result.context_chunks), 1)
        self.assertEqual(result.total_context_length, 100)
        self.assertEqual(result.retrieval_stats, stats)

if __name__ == '__main__':
    unittest.main(verbosity=2)