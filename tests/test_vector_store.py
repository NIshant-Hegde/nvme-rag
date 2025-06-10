import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store.base import SearchQuery, SearchResult, DistanceMetric
from src.vector_store.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.vector_store.chroma_store import ChromaVectorStore
from src.models.document import ProcessedChunk

class TestEmbeddingGenerator(unittest.TestCase):
    """Test embedding generation functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = EmbeddingConfig(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            device="cpu",
            batch_size=2,
            cache_dir=self.temp_dir
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_embedding_generator_initialization(self, mock_model, mock_tokenizer):
        """Test embedding generator initialization"""
        # Mock the model and tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        generator = EmbeddingGenerator(self.config)
        
        self.assertIsNotNone(generator.tokenizer)
        self.assertIsNotNone(generator.model)
        self.assertEqual(generator.device.type, "cpu")
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_generate_embeddings(self, mock_model, mock_tokenizer):
        """Test embedding generation for texts"""
        # Setup mocks
        # Mock tokenizer instances
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock generator's generate_embeddings to return predictable embeddings
        with patch.object(EmbeddingGenerator, 'generate_embeddings') as mock_generate:
            import numpy as np
            mock_generate.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Mock embedding generation
        with patch.object(EmbeddingGenerator, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            
            generator = EmbeddingGenerator(self.config)
        
        # Mock the mean pooling method to return predictable embeddings
        texts = ["This is test text 1", "This is test text 2"]
        embeddings = generator.generate_embeddings(texts)
        
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 3)
        self.assertTrue(isinstance(embeddings, np.ndarray))
    
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_embedding_caching(self, mock_model, mock_tokenizer):
        """Test embedding caching functionality"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        generator = EmbeddingGenerator(self.config)
        
        # Mock embedding generation
        with patch.object(EmbeddingGenerator, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # First call should generate embedding
            text = "Test text for caching"
            # First call should generate embedding
            embeddings1 = generator.generate_embeddings([text])
            
            # Second call should use cache
            embeddings2 = generator.generate_embeddings([text])
            
            # Should only call generate once due to caching
            mock_generate.assert_called_once()
            np.testing.assert_array_equal(embeddings1, embeddings2)
    
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_generate_chunk_embeddings(self, mock_model, mock_tokenizer):
        """Test embedding generation for ProcessedChunk objects"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        generator = EmbeddingGenerator(self.config)
        
        # Mock embedding generation
        with patch.object(EmbeddingGenerator, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Create test chunks
        chunks = [
            ProcessedChunk(
                content="Test content 1",
                metadata={},
                chunk_id="chunk_1",
                parent_doc_id="doc_1",
                section_header="Section 1",
                page_number=1,
                chunk_type="text",
                semantic_density=0.8
            ),
            ProcessedChunk(
                content="Test content 2",
                metadata={},
                chunk_id="chunk_2",
                parent_doc_id="doc_1",
                section_header="Section 2",
                page_number=1,
                chunk_type="text",
                semantic_density=0.9
            )
        ]
        
        with patch.object(generator, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            
            chunk_embeddings = generator.generate_chunk_embeddings(chunks)
            
            self.assertEqual(len(chunk_embeddings), 2)
            self.assertIn("chunk_1", chunk_embeddings)
            self.assertIn("chunk_2", chunk_embeddings)
            mock_generate.assert_called_once()

class TestChromaVectorStore(unittest.TestCase):
    """Test ChromaDB vector store functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_config = EmbeddingConfig(cache_embeddings=False)
        
        # Create test chunks
        self.test_chunks = [
            ProcessedChunk(
                content="This is about NVMe controllers and their implementation.",
                metadata={"test": "metadata1"},
                chunk_id="test_chunk_1",
                parent_doc_id="test_doc",
                section_header="NVMe Controllers",
                page_number=1,
                chunk_type="text",
                semantic_density=0.8
            ),
            ProcessedChunk(
                content="PCIe interface specifications for NVMe devices.",
                metadata={"test": "metadata2"},
                chunk_id="test_chunk_2",
                parent_doc_id="test_doc",
                section_header="PCIe Interface",
                page_number=2,
                chunk_type="text",
                semantic_density=0.9
            )
        ]
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('src.vector_store.chroma_store.chromadb.Client')
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_chroma_store_initialization(self, mock_model, mock_tokenizer, mock_client):
        """Test ChromaDB store initialization"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = ValueError("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the embedding function
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.return_value = np.array([[0.1, 0.2, 0.3]])
        self.embedding_config.model_name = mock_embedding_fn
        
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir,
            embedding_config=self.embedding_config
        )
        
        self.assertIsNotNone(store.collection)
        self.assertEqual(store.collection_name, "test_collection")
        mock_client_instance.create_collection.assert_called_once()
    
    @patch('src.vector_store.chroma_store.chromadb.Client')
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_add_chunks(self, mock_model, mock_tokenizer, mock_client):
        """Test adding chunks to ChromaDB"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = ValueError("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the embedding function
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.return_value = np.array([[0.1, 0.2, 0.3]])
        self.embedding_config.model_name = mock_embedding_fn
        
        store = ChromaVectorStore(
            persist_directory=self.temp_dir,
            embedding_config=self.embedding_config
        )
        
        # Test adding chunks
        chunk_ids = store.add_chunks(self.test_chunks)
        
        self.assertEqual(len(chunk_ids), 2)
        self.assertIn("test_chunk_1", chunk_ids)
        self.assertIn("test_chunk_2", chunk_ids)
        mock_collection.add.assert_called_once()
    
    @patch('src.vector_store.chroma_store.chromadb.Client')
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_search_functionality(self, mock_model, mock_tokenizer, mock_client):
        """Test search functionality"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["test_chunk_1", "test_chunk_2"]],
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[
                {"parent_doc_id": "doc1", "section_header": "Section 1", "chunk_type": "text"},
                {"parent_doc_id": "doc1", "section_header": "Section 2", "chunk_type": "text"}
            ]],
            "distances": [[0.1, 0.2]]
        }
        
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the embedding function
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.return_value = np.array([[0.1, 0.2, 0.3]])
        self.embedding_config.model_name = mock_embedding_fn
        
        store = ChromaVectorStore(
            persist_directory=self.temp_dir,
            embedding_config=self.embedding_config
        )
        
        # Test search
        query = SearchQuery(
            query_text="test query",
            top_k=5,
            distance_metric=DistanceMetric.COSINE
        )
        
        results = store.search(query)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].chunk.chunk_id, "test_chunk_1")
        mock_collection.query.assert_called_once()
    
    @patch('src.vector_store.chroma_store.chromadb.Client')
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_get_chunk_by_id(self, mock_model, mock_tokenizer, mock_client):
        """Test retrieving chunk by ID"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock get results
        mock_collection.get.return_value = {
            "ids": ["test_chunk_1"],
            "documents": ["Test document content"],
            "metadatas": [{
                "parent_doc_id": "doc1",
                "section_header": "Section 1",
                "page_number": 1,
                "chunk_type": "text",
                "semantic_density": 0.8
            }]
        }
        
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the embedding function
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.return_value = np.array([[0.1, 0.2, 0.3]])
        self.embedding_config.model_name = mock_embedding_fn
        
        store = ChromaVectorStore(
            persist_directory=self.temp_dir,
            embedding_config=self.embedding_config
        )
        
        # Test get chunk by ID
        chunk = store.get_chunk_by_id("test_chunk_1")
        
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.chunk_id, "test_chunk_1")
        self.assertEqual(chunk.content, "Test document content")
        mock_collection.get.assert_called_once()
    
    @patch('src.vector_store.chroma_store.chromadb.Client')
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_delete_chunks(self, mock_model, mock_tokenizer, mock_client):
        """Test deleting chunks"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock existing chunks check
        mock_collection.get.return_value = {
            "ids": ["test_chunk_1", "test_chunk_2"]
        }
        
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the embedding function
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.return_value = np.array([[0.1, 0.2, 0.3]])
        self.embedding_config.model_name = mock_embedding_fn
        
        store = ChromaVectorStore(
            persist_directory=self.temp_dir,
            embedding_config=self.embedding_config
        )
        
        # Test delete
        deleted_count = store.delete_chunks(["test_chunk_1", "test_chunk_2"])
        
        self.assertEqual(deleted_count, 2)
        mock_collection.delete.assert_called_once()
    
    @patch('src.vector_store.chroma_store.chromadb.Client')
    @patch('src.vector_store.embedding_generator.AutoTokenizer')
    @patch('src.vector_store.embedding_generator.AutoModel')
    def test_get_stats(self, mock_model, mock_tokenizer, mock_client):
        """Test getting vector store statistics"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = mock_collection
        
        mock_collection.count.return_value = 100
        mock_collection.peek.return_value = {
            "metadatas": [
                {"extraction_method": "marker", "chunk_type": "text", "parent_doc_id": "doc1"},
                {"extraction_method": "pymupdf", "chunk_type": "text", "parent_doc_id": "doc2"}
            ]
        }
        
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock embedding generator dimension
        with patch.object(EmbeddingGenerator, 'get_embedding_dimension', return_value=384):
            store = ChromaVectorStore(
                persist_directory=self.temp_dir,
                embedding_config=self.embedding_config
            )
            
            stats = store.get_stats()
            
            self.assertEqual(stats["total_chunks"], 100)
            self.assertIn("embedding_dimension", stats)
            self.assertIn("extraction_methods", stats)

class TestSearchQuery(unittest.TestCase):
    """Test search query functionality"""
    
    def test_search_query_creation(self):
        """Test creating search queries with different parameters"""
        # Basic query
        query = SearchQuery(query_text="test query")
        self.assertEqual(query.query_text, "test query")
        self.assertEqual(query.top_k, 10)
        self.assertEqual(query.distance_metric, DistanceMetric.COSINE)
        self.assertIsNone(query.filters)
        
        # Advanced query with filters
        filters = {"section_header": "NVMe Controllers"}
        query = SearchQuery(
            query_text="controller implementation",
            top_k=5,
            distance_metric=DistanceMetric.EUCLIDEAN,
            filters=filters,
            min_score=0.8
        )
        
        self.assertEqual(query.top_k, 5)
        self.assertEqual(query.distance_metric, DistanceMetric.EUCLIDEAN)
        self.assertEqual(query.filters, filters)
        self.assertEqual(query.min_score, 0.8)

class TestSearchResult(unittest.TestCase):
    """Test search result functionality"""
    
    def test_search_result_creation(self):
        """Test creating and serializing search results"""
        chunk = ProcessedChunk(
            content="Test content",
            metadata={"test": "value"},
            chunk_id="test_chunk",
            parent_doc_id="test_doc",
            section_header="Test Section",
            page_number=1,
            chunk_type="text",
            semantic_density=0.8
        )
        
        result = SearchResult(
            chunk=chunk,
            score=0.95,
            distance=0.05,
            metadata={"extra": "metadata"}
        )
        
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.distance, 0.05)
        self.assertEqual(result.chunk.chunk_id, "test_chunk")
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIn("chunk", result_dict)
        self.assertIn("score", result_dict)
        self.assertIn("distance", result_dict)
        self.assertIn("metadata", result_dict)

if __name__ == '__main__':
    unittest.main(verbosity=2)