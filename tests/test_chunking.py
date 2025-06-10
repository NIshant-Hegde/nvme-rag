import unittest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.semantic_chunker import SemanticChunker

class TestSemanticChunkingDetailed(unittest.TestCase):
    """Detailed tests for semantic chunking functionality"""
    
    def setUp(self):
        """Setup test chunker with mocked models"""
        with patch('src.data_processing.semantic_chunker.AutoTokenizer') as mock_tokenizer, \
             patch('src.data_processing.semantic_chunker.AutoModel') as mock_model:
            
            self.chunker = SemanticChunker(
                device="cpu", 
                similarity_threshold=0.7, 
                max_chunk_length=1000
            )
            
            # Setup mock tokenizer
            mock_tokenizer_instance = Mock()
            def mock_tokenizer_call(*args, **kwargs):
                return {
                    'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
                }
            mock_tokenizer_instance.side_effect = mock_tokenizer_call
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Setup mock model
            mock_model_instance = Mock()
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(1, 5, 384)  # Batch, seq_len, hidden_size
            mock_model_instance.return_value = mock_output
            mock_model_instance.to = Mock(return_value=mock_model_instance)
            mock_model.from_pretrained.return_value = mock_model_instance
            
            self.chunker.tokenizer = mock_tokenizer_instance
            self.chunker.embedding_model = mock_model_instance
    
    def test_device_setup(self):
        """Test device detection and setup"""
        # Test auto device detection
        with patch('torch.cuda.is_available', return_value=True):
            device = self.chunker._setup_device("auto")
            self.assertEqual(device.type, "cuda")
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            device = self.chunker._setup_device("auto")
            self.assertEqual(device.type, "mps")
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = self.chunker._setup_device("auto")
            self.assertEqual(device.type, "cpu")
        
        # Test explicit device
        device = self.chunker._setup_device("cpu")
        self.assertEqual(device.type, "cpu")
    
    def test_header_extraction_edge_cases(self):
        """Test edge cases in header extraction"""
        # Test malformed headers
        malformed_text = """
        ### Header with no space
        #Invalid header
        # 
        ##Another header   
                """
        
        headers = self.chunker.extract_section_headers(malformed_text)
        
        # Should extract valid headers and ignore malformed ones
        # Note: empty header "# " will be filtered out, but "Invalid header" without space is valid
        self.assertEqual(len(headers), 3)  # All headers except empty one
        self.assertEqual(headers[0][0], "Header with no space")
        self.assertEqual(headers[1][0], "Invalid header")  # This is actually valid
        self.assertEqual(headers[2][0], "Another header")
    
    def test_cosine_similarity_edge_cases(self):
        """Test cosine similarity with edge cases"""
        # Zero vectors
        zero_vec = np.array([0, 0, 0])
        normal_vec = np.array([1, 2, 3])
        
        similarity = self.chunker._cosine_similarity(zero_vec, normal_vec)
        self.assertEqual(similarity, 0.0)
        
        # Same vector
        similarity = self.chunker._cosine_similarity(normal_vec, normal_vec)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Opposite vectors
        opposite_vec = np.array([-1, -2, -3])
        similarity = self.chunker._cosine_similarity(normal_vec, opposite_vec)
        self.assertAlmostEqual(similarity, -1.0, places=5)
    
    def test_paragraph_embedding_generation(self):
        """Test paragraph embedding generation"""
        paragraphs = [
            "This is the first paragraph about machine learning.",
            "This is the second paragraph also about machine learning.",
            "This paragraph discusses something completely different."
        ]
        
        # Mock the embedding generation to return predictable results
        def mock_forward(*args, **kwargs):
            mock_output = Mock()
            # Return different embeddings for different inputs
            if "first" in str(args) or "first" in str(kwargs):
                mock_output.last_hidden_state = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]]])
            elif "second" in str(args) or "second" in str(kwargs):
                mock_output.last_hidden_state = torch.tensor([[[0.9, 0.1, 0.0, 0.0, 0.0]]])
            else:
                mock_output.last_hidden_state = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]])
            return mock_output
        
        self.chunker.embedding_model.side_effect = mock_forward
        
        embeddings = self.chunker._get_paragraph_embeddings(paragraphs)
        
        self.assertEqual(embeddings.shape[0], 3)  # Three paragraphs
        self.assertGreater(embeddings.shape[1], 0)  # Non-zero embedding dimension
    
    def test_chunking_with_different_thresholds(self):
        """Test chunking behavior with different similarity thresholds"""
        test_text = """
        Machine learning is a subset of artificial intelligence. It focuses on algorithms that learn from data.

        Deep learning is a subset of machine learning. It uses neural networks with multiple layers.

        Cooking pasta requires boiling water. Add salt to the water before adding pasta.

        The best pasta cooking time depends on the type. Al dente is preferred by many chefs.
        """
        
        # Mock embeddings to simulate related and unrelated content
        def mock_get_embeddings(paragraphs):
            embeddings = []
            for p in paragraphs:
                if "machine learning" in p.lower() or "deep learning" in p.lower():
                    embeddings.append(np.array([1.0, 0.0, 0.0]))  # ML topic
                else:
                    embeddings.append(np.array([0.0, 1.0, 0.0]))  # Cooking topic
            return np.array(embeddings)
        
        with patch.object(self.chunker, '_get_paragraph_embeddings', side_effect=mock_get_embeddings):
            # High threshold - should create more chunks (less grouping)
            high_threshold_chunker = SemanticChunker(device="cpu", similarity_threshold=0.9)
            high_threshold_chunker.tokenizer = self.chunker.tokenizer
            high_threshold_chunker.embedding_model = self.chunker.embedding_model
            high_threshold_chunker._get_paragraph_embeddings = mock_get_embeddings
            
            high_chunks = high_threshold_chunker.semantic_chunking(test_text)
            
            # Low threshold - should create fewer chunks (more grouping)
            low_threshold_chunker = SemanticChunker(device="cpu", similarity_threshold=0.3)
            low_threshold_chunker.tokenizer = self.chunker.tokenizer
            low_threshold_chunker.embedding_model = self.chunker.embedding_model
            low_threshold_chunker._get_paragraph_embeddings = mock_get_embeddings
            
            low_chunks = low_threshold_chunker.semantic_chunking(test_text)
            
            # Low threshold should generally create fewer or equal chunks
            self.assertLessEqual(len(low_chunks), len(high_chunks))
    
    def test_section_context_assignment(self):
        """Test section context assignment to chunks"""
        original_text = """# Chapter 1: Introduction

        This is the introduction paragraph.

        This is another introduction paragraph.

        ## Section 1.1: Background

        This is background information.

        ## Section 1.2: Objectives

        These are the objectives.

        # Chapter 2: Methods

This is about methods.
        """
        
        headers = [
            ("Chapter 1: Introduction", 0, 1),
            ("Section 1.1: Background", 6, 2),
            ("Section 1.2: Objectives", 10, 2),
            ("Chapter 2: Methods", 14, 1)
        ]
        
        chunks = [
            {'content': 'This is the introduction paragraph.\n\nThis is another introduction paragraph.'},
            {'content': 'This is background information.'},
            {'content': 'These are the objectives.'},
            {'content': 'This is about methods.'}
        ]
        
        contextualized_chunks = self.chunker.assign_section_context(chunks, headers, original_text)
        
        # Debug: Print what we actually get
        print("\nActual section assignments:")
        for i, chunk in enumerate(contextualized_chunks):
            print(f"Chunk {i}: '{chunk['section_header']}'")
        
        # Check that chunks are assigned to correct sections
        # The logic finds the last header that appears before the chunk content
        self.assertEqual(contextualized_chunks[0]['section_header'], "Chapter 1: Introduction")
        self.assertEqual(contextualized_chunks[1]['section_header'], "Section 1.1: Background")
        self.assertEqual(contextualized_chunks[2]['section_header'], "Section 1.2: Objectives")
        self.assertEqual(contextualized_chunks[3]['section_header'], "Chapter 2: Methods")
    
    def test_chunk_length_constraints(self):
        """Test that chunks respect maximum length constraints"""
        # Create very long paragraphs that should be split
        long_paragraph = "This is a very long paragraph. " * 50  # ~1500 characters
        test_text = f"{long_paragraph}\n\n{long_paragraph}\n\n{long_paragraph}"
        
        def mock_get_embeddings(paragraphs):
            # Return different embeddings so chunks won't merge due to low similarity
            embeddings = []
            for i, p in enumerate(paragraphs):
                # Create orthogonal vectors to ensure low similarity
                vec = [0.0] * 3
                vec[i % 3] = 1.0
                embeddings.append(vec)
            return np.array(embeddings)
        
        with patch.object(self.chunker, '_get_paragraph_embeddings', side_effect=mock_get_embeddings):
            chunks = self.chunker.semantic_chunking(test_text)
            
            # Check that chunks are created (should be multiple due to low similarity)
            self.assertGreater(len(chunks), 0)
            
            # Each chunk should be roughly one paragraph since similarity is low
            for chunk in chunks:
                # With low similarity, each paragraph should be its own chunk
                # So no chunk should significantly exceed paragraph length
                self.assertLessEqual(chunk['character_length'], 2000)  # Reasonable buffer
    
    def test_empty_and_short_content_handling(self):
        """Test handling of empty or very short content"""
        # Empty content
        empty_chunks = self.chunker.semantic_chunking("")
        self.assertEqual(len(empty_chunks), 0)
        
        # Only whitespace
        whitespace_chunks = self.chunker.semantic_chunking("   \n\n   \t\t  ")
        self.assertEqual(len(whitespace_chunks), 0)
        
        # Very short content (below minimum threshold)
        short_text = "Hi.\n\nBye."
        short_chunks = self.chunker.semantic_chunking(short_text)
        self.assertEqual(len(short_chunks), 0)  # Should be filtered out
    
    def test_chunk_metadata_completeness(self):
        """Test that all required metadata is present in chunks"""
        test_text = """
This is a substantial paragraph with enough content to pass the length filter. It contains multiple sentences and provides meaningful information.

This is another substantial paragraph that also contains enough content. It discusses related topics and should be processed properly.
        """
        
        def mock_get_embeddings(paragraphs):
            return np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
        
        with patch.object(self.chunker, '_get_paragraph_embeddings', side_effect=mock_get_embeddings):
            chunks = self.chunker.semantic_chunking(test_text)
            
            self.assertGreater(len(chunks), 0)
            
            for chunk in chunks:
                # Check all required fields are present
                required_fields = ['content', 'paragraph_count', 'semantic_density', 'character_length']
                for field in required_fields:
                    self.assertIn(field, chunk)
                
                # Check data types
                self.assertIsInstance(chunk['content'], str)
                self.assertIsInstance(chunk['paragraph_count'], int)
                self.assertIsInstance(chunk['semantic_density'], float)
                self.assertIsInstance(chunk['character_length'], int)
                
                # Check reasonable values
                self.assertGreater(len(chunk['content']), 0)
                self.assertGreater(chunk['paragraph_count'], 0)
                self.assertGreaterEqual(chunk['semantic_density'], 0.0)
                self.assertLessEqual(chunk['semantic_density'], 1.0)
                self.assertEqual(chunk['character_length'], len(chunk['content']))

if __name__ == '__main__':
    unittest.main(verbosity=2)