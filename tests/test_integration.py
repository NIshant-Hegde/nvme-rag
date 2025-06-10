import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.integration import RAGPipelineIntegration
from src.models.document import ProcessedChunk, ProcessingResult, DocumentMetadata
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaConfig, ChatMessage
from src.retrieval.retrieval_pipeline import RetrievalConfig

class TestRAGPipelineIntegration(unittest.TestCase):
    """Test complete RAG pipeline integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configurations
        self.embedding_config = EmbeddingConfig(
            device="cpu",
            cache_embeddings=False,
            cache_dir=self.temp_dir
        )
        
        self.ollama_config = OllamaConfig(
            model="mistral:7b",
            temperature=0.1,
            timeout=30
        )
        
        self.retrieval_config = RetrievalConfig(
            top_k=5,
            min_score=0.7
        )
        
        # Create test chunks
        self.test_chunks = [
            ProcessedChunk(
                content="NVMe controllers provide high-performance storage interface.",
                metadata={"test": "metadata1"},
                chunk_id="test_chunk_1",
                parent_doc_id="test_doc",
                section_header="NVMe Controllers",
                page_number=1,
                chunk_type="text",
                semantic_density=0.8
            ),
            ProcessedChunk(
                content="PCIe Gen4 interface supports up to 64 GB/s bandwidth.",
                metadata={"test": "metadata2"},
                chunk_id="test_chunk_2",
                parent_doc_id="test_doc",
                section_header="PCIe Interface",
                page_number=2,
                chunk_type="text",
                semantic_density=0.9
            )
        ]
        
        # Create test processing result
        self.test_metadata = DocumentMetadata(
            source_path="test.pdf",
            document_type="pdf",
            page_count=10,
            processing_timestamp="2025-01-01T00:00:00",
            file_hash="test_hash",
            extraction_method="test",
            has_images=False,
            has_tables=True
        )
        
        self.test_processing_result = ProcessingResult(
            chunks=self.test_chunks,
            metadata=self.test_metadata,
            processing_stats={"processing_time": 5.0, "total_chunks": 2}
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('src.pipeline.integration.DocumentProcessor')
    @patch('src.pipeline.integration.ChromaVectorStore')
    @patch('src.pipeline.integration.OllamaClient')
    @patch('src.pipeline.integration.RetrievalPipeline')
    def test_pipeline_initialization(self, mock_retrieval, mock_ollama, mock_vector_store, mock_doc_processor):
        """Test RAG pipeline initialization"""
        # Mock components
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_ollama_instance = Mock()
        mock_retrieval_instance = Mock()
        
        mock_doc_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ollama.return_value = mock_ollama_instance
        mock_retrieval.return_value = mock_retrieval_instance
        
        # Test initialization
        pipeline = RAGPipelineIntegration(
            vector_store_path=self.temp_dir,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            retrieval_config=self.retrieval_config
        )
        
        # Verify components were initialized
        mock_doc_processor.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_ollama.assert_called_once()
        mock_retrieval.assert_called_once()
        
        self.assertIsNotNone(pipeline.document_processor)
        self.assertIsNotNone(pipeline.vector_store)
        self.assertIsNotNone(pipeline.llm_client)
        self.assertIsNotNone(pipeline.retrieval_pipeline)
    
    @patch('src.pipeline.integration.DocumentProcessor')
    @patch('src.pipeline.integration.ChromaVectorStore')
    @patch('src.pipeline.integration.OllamaClient')
    @patch('src.pipeline.integration.RetrievalPipeline')
    def test_process_and_index_document(self, mock_retrieval, mock_ollama, mock_vector_store, mock_doc_processor):
        """Test complete document processing and indexing"""
        # Mock components
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_ollama_instance = Mock()
        mock_retrieval_instance = Mock()
        
        mock_doc_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ollama.return_value = mock_ollama_instance
        mock_retrieval.return_value = mock_retrieval_instance
        
        # Mock document processing
        mock_doc_processor_instance.process_document.return_value = self.test_processing_result
        
        # Mock vector store operations
        mock_vector_store_instance.add_chunks.return_value = ["test_chunk_1", "test_chunk_2"]
        mock_vector_store_instance.get_stats.return_value = {
            "total_chunks": 2,
            "collection_name": "test_collection"
        }
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=self.temp_dir,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            retrieval_config=self.retrieval_config
        )
        
        # Test document processing and indexing
        pdf_path = Path("test.pdf")
        result = pipeline.process_and_index_document(pdf_path)
        
        # Verify successful processing
        self.assertTrue(result["success"])
        self.assertEqual(result["chunks_processed"], 2)
        self.assertEqual(result["chunks_indexed"], 2)
        self.assertIn("processing_stats", result)
        self.assertIn("document_metadata", result)
        self.assertIn("vector_store_stats", result)
        
        # Verify method calls
        mock_doc_processor_instance.process_document.assert_called_once_with(pdf_path)
        mock_vector_store_instance.add_chunks.assert_called_once_with(self.test_chunks)
    
    @patch('src.pipeline.integration.DocumentProcessor')
    @patch('src.pipeline.integration.ChromaVectorStore')
    @patch('src.pipeline.integration.OllamaClient')
    @patch('src.pipeline.integration.RetrievalPipeline')
    def test_search_and_retrieve(self, mock_retrieval, mock_ollama, mock_vector_store, mock_doc_processor):
        """Test search and retrieval functionality"""
        # Mock components
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_ollama_instance = Mock()
        mock_retrieval_instance = Mock()
        
        mock_doc_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ollama.return_value = mock_ollama_instance
        mock_retrieval.return_value = mock_retrieval_instance
        
        # Mock retrieval result
        from src.retrieval.retrieval_pipeline import RetrievalResult, QueryContext
        mock_query_context = QueryContext(
            original_query="test query",
            enhanced_query="enhanced test query",
            chat_history=[],
            filters={},
            user_preferences={}
        )
        
        mock_retrieval_result = RetrievalResult(
            query_context=mock_query_context,
            search_results=[],
            context_chunks=self.test_chunks,
            total_context_length=100,
            retrieval_stats={"total_results": 2}
        )
        
        mock_retrieval_instance.retrieve.return_value = mock_retrieval_result
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=self.temp_dir,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            retrieval_config=self.retrieval_config
        )
        
        # Test search and retrieval
        query = "How do NVMe controllers work?"
        chat_history = [ChatMessage(role="user", content="Previous question")]
        filters = {"section_header": "NVMe Controllers"}
        
        result = pipeline.search_and_retrieve(
            query=query,
            chat_history=chat_history,
            filters=filters
        )
        
        # Verify successful retrieval
        self.assertTrue(result["success"])
        self.assertEqual(result["query"]["original"], "test query")
        self.assertEqual(result["query"]["enhanced"], "enhanced test query")
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(result["total_context_length"], 100)
        self.assertIn("context", result)
        self.assertIn("stats", result)
        
        # Verify method calls
        mock_retrieval_instance.retrieve.assert_called_once_with(
            query=query,
            chat_history=chat_history,
            filters=filters
        )
    
    @patch('src.pipeline.integration.DocumentProcessor')
    @patch('src.pipeline.integration.ChromaVectorStore')
    @patch('src.pipeline.integration.OllamaClient')
    @patch('src.pipeline.integration.RetrievalPipeline')
    def test_get_pipeline_status(self, mock_retrieval, mock_ollama, mock_vector_store, mock_doc_processor):
        """Test getting pipeline status"""
        # Mock components
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_ollama_instance = Mock()
        mock_retrieval_instance = Mock()
        
        mock_doc_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ollama.return_value = mock_ollama_instance
        mock_retrieval.return_value = mock_retrieval_instance
        
        # Mock semantic chunker device
        mock_semantic_chunker = Mock()
        mock_semantic_chunker.device = "cpu"
        mock_doc_processor_instance.semantic_chunker = mock_semantic_chunker
        
        # Mock vector store stats
        mock_vector_store_instance.get_stats.return_value = {
            "total_chunks": 100,
            "collection_name": "test_collection",
            "embedding_model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        }
        
        # Mock LLM health check
        mock_ollama_instance.health_check.return_value = {
            "status": "healthy",
            "model": "mistral:7b",
            "response_time_seconds": 0.5
        }
        
        # Mock retrieval config
        mock_retrieval_instance.get_config.return_value = {
            "strategy": "semantic_only",
            "top_k": 5
        }
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=self.temp_dir,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            retrieval_config=self.retrieval_config
        )
        
        # Test status retrieval
        status = pipeline.get_pipeline_status()
        
        # Verify status structure
        self.assertIn("timestamp", status)
        self.assertIn("components", status)
        self.assertIn("overall_status", status)
        
        # Verify component statuses
        components = status["components"]
        self.assertIn("document_processor", components)
        self.assertIn("vector_store", components)
        self.assertIn("llm_client", components)
        self.assertIn("retrieval_pipeline", components)
        
        # Verify specific component details
        self.assertEqual(components["document_processor"]["device"], "cpu")
        self.assertEqual(components["vector_store"]["total_chunks"], 100)
        self.assertEqual(components["llm_client"]["status"], "healthy")
        self.assertEqual(components["retrieval_pipeline"]["strategy"], "semantic_only")
        
        self.assertEqual(status["overall_status"], "ready")
    
    @patch('src.pipeline.integration.DocumentProcessor')
    @patch('src.pipeline.integration.ChromaVectorStore')
    @patch('src.pipeline.integration.OllamaClient')
    @patch('src.pipeline.integration.RetrievalPipeline')
    def test_error_handling(self, mock_retrieval, mock_ollama, mock_vector_store, mock_doc_processor):
        """Test error handling in pipeline operations"""
        # Mock components
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_ollama_instance = Mock()
        mock_retrieval_instance = Mock()
        
        mock_doc_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ollama.return_value = mock_ollama_instance
        mock_retrieval.return_value = mock_retrieval_instance
        
        # Mock document processing failure
        mock_doc_processor_instance.process_document.side_effect = Exception("Processing failed")
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=self.temp_dir,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            retrieval_config=self.retrieval_config
        )
        
        # Test error handling in document processing
        pdf_path = Path("test.pdf")
        result = pipeline.process_and_index_document(pdf_path)
        
        # Verify error handling
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Processing failed")

if __name__ == '__main__':
    unittest.main(verbosity=2)