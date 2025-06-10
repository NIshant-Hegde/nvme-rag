import unittest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.document import DocumentMetadata, ProcessedChunk, ProcessingResult
from src.data_processing.pdf_processor import PDFProcessor
from src.data_processing.semantic_chunker import SemanticChunker
from src.data_processing.document_processor import DocumentProcessor

class TestDocumentModels(unittest.TestCase):
    """Test document data models"""
    
    def setUp(self):
        self.sample_metadata = DocumentMetadata(
            source_path="/test/document.pdf",
            document_type="pdf",
            page_count=10,
            processing_timestamp="2025-01-01T00:00:00",
            file_hash="abc123def456",
            extraction_method="marker",
            has_images=True,
            has_tables=False,
            language="en"
        )
        
        self.sample_chunk = ProcessedChunk(
            content="This is test content for the chunk.",
            metadata={"test_key": "test_value"},
            chunk_id="abc123_0001",
            parent_doc_id="abc123",
            section_header="Test Section",
            page_number=1,
            chunk_type="text",
            semantic_density=0.85
        )
    
    def test_document_metadata_serialization(self):
        """Test metadata to_dict and from_dict"""
        metadata_dict = self.sample_metadata.to_dict()
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict["source_path"], "/test/document.pdf")
        
        # Test round-trip
        restored_metadata = DocumentMetadata.from_dict(metadata_dict)
        self.assertEqual(restored_metadata.source_path, self.sample_metadata.source_path)
        self.assertEqual(restored_metadata.file_hash, self.sample_metadata.file_hash)
    
    def test_processed_chunk_serialization(self):
        """Test chunk to_dict and from_dict"""
        chunk_dict = self.sample_chunk.to_dict()
        self.assertIsInstance(chunk_dict, dict)
        self.assertEqual(chunk_dict["content"], "This is test content for the chunk.")
        
        # Test round-trip
        restored_chunk = ProcessedChunk.from_dict(chunk_dict)
        self.assertEqual(restored_chunk.content, self.sample_chunk.content)
        self.assertEqual(restored_chunk.chunk_id, self.sample_chunk.chunk_id)
    
    def test_processing_result_save_load(self):
        """Test ProcessingResult save and load functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create processing result
            chunks = [self.sample_chunk]
            stats = {"test_stat": 42, "numpy_float": np.float32(3.14)}
            result = ProcessingResult(chunks, self.sample_metadata, stats)
            
            # Save to files
            result.save_to_files(temp_dir)
            
            # Check files were created
            doc_id = self.sample_metadata.file_hash[:12]
            chunks_file = Path(temp_dir) / f"{doc_id}_chunks.json"
            metadata_file = Path(temp_dir) / f"{doc_id}_metadata.json"
            stats_file = Path(temp_dir) / f"{doc_id}_stats.json"
            
            self.assertTrue(chunks_file.exists())
            self.assertTrue(metadata_file.exists())
            self.assertTrue(stats_file.exists())
            
            # Load and verify
            loaded_result = ProcessingResult.load_from_files(temp_dir, doc_id)
            self.assertEqual(len(loaded_result.chunks), 1)
            self.assertEqual(loaded_result.chunks[0].content, self.sample_chunk.content)
            self.assertEqual(loaded_result.metadata.source_path, self.sample_metadata.source_path)

class TestPDFProcessor(unittest.TestCase):
    """Test PDF processing functionality"""
    
    def setUp(self):
        self.processor = PDFProcessor()
    
    def test_file_hash_calculation(self):
        """Test file hash calculation"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("test content")
            temp_path = Path(temp_file.name)
        
        try:
            hash1 = self.processor.calculate_file_hash(temp_path)
            hash2 = self.processor.calculate_file_hash(temp_path)
            
            # Same file should produce same hash
            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 64)  # SHA256 produces 64 char hex string
        finally:
            temp_path.unlink()
    
    def test_table_detection(self):
        """Test table detection in markdown"""
        # Text with table
        table_text = """
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
        """
        self.assertTrue(self.processor._detect_tables(table_text))
        
        # Text without table
        no_table_text = "This is just regular text without any tables."
        self.assertFalse(self.processor._detect_tables(no_table_text))
        
        # Text with partial table markers but not a real table
        fake_table_text = "Use | for pipes and | symbols in text."
        self.assertFalse(self.processor._detect_tables(fake_table_text))
    
    @patch('src.data_processing.pdf_processor.pymupdf4llm.to_markdown')
    def test_pymupdf_extraction(self, mock_to_markdown):
        """Test PyMuPDF extraction method"""
        mock_to_markdown.return_value = "# Test Document\n\nThis is test content."
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            text, info = self.processor.extract_with_pymupdf(temp_path)
            
            self.assertEqual(text, "# Test Document\n\nThis is test content.")
            self.assertEqual(info["extraction_method"], "pymupdf4llm")
            self.assertTrue(info["success"])
            mock_to_markdown.assert_called_once()
        finally:
            temp_path.unlink()

class TestSemanticChunker(unittest.TestCase):
    """Test semantic chunking functionality"""
    
    def setUp(self):
        # Mock the models to avoid loading in tests
        with patch('src.data_processing.semantic_chunker.AutoTokenizer'), \
             patch('src.data_processing.semantic_chunker.AutoModel'):
            self.chunker = SemanticChunker(device="cpu", similarity_threshold=0.7)
            # Mock the models
            self.chunker.tokenizer = Mock()
            self.chunker.embedding_model = Mock()
    
    def test_header_extraction(self):
        """Test markdown header extraction"""
        markdown_text = """
# Main Title
Some content here.

## Section 1
More content.

### Subsection 1.1
Even more content.

## Section 2
Final content.
        """
        
        headers = self.chunker.extract_section_headers(markdown_text)
        
        self.assertEqual(len(headers), 4)
        self.assertEqual(headers[0][0], "Main Title")  # header text
        self.assertEqual(headers[0][2], 1)  # level
        self.assertEqual(headers[1][0], "Section 1")
        self.assertEqual(headers[1][2], 2)  # level
        self.assertEqual(headers[2][0], "Subsection 1.1")
        self.assertEqual(headers[2][2], 3)  # level
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec_a = np.array([1, 0, 0])
        vec_b = np.array([0, 1, 0])
        vec_c = np.array([1, 0, 0])
        
        # Orthogonal vectors should have similarity 0
        similarity_ab = self.chunker._cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity_ab, 0.0, places=5)
        
        # Identical vectors should have similarity 1
        similarity_ac = self.chunker._cosine_similarity(vec_a, vec_c)
        self.assertAlmostEqual(similarity_ac, 1.0, places=5)
    
    @patch.object(SemanticChunker, '_get_paragraph_embeddings')
    def test_semantic_chunking(self, mock_embeddings):
        """Test semantic chunking logic"""
        # Mock embeddings - similar pairs should be grouped
        mock_embeddings.return_value = np.array([
            [1, 0, 0],  # Similar to next
            [0.9, 0.1, 0],  # Similar to previous
            [0, 1, 0],  # Different from others
            [0, 0.9, 0.1]  # Similar to previous
        ])
        
        test_text = """
Paragraph one about topic A. This has some content.

Paragraph two also about topic A. This is related content.

Paragraph three about topic B. This is different content.

Paragraph four also about topic B. This is also different but related.
        """
        
        chunks = self.chunker.semantic_chunking(test_text)
        
        # Should create fewer chunks than paragraphs due to grouping
        self.assertGreater(len(chunks), 0)
        self.assertLessEqual(len(chunks), 4)  # At most 4 (one per paragraph)
        
        # Check chunk structure
        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('paragraph_count', chunk)
            self.assertIn('semantic_density', chunk)
            self.assertIn('character_length', chunk)

class TestDocumentProcessor(unittest.TestCase):
    """Test complete document processing pipeline"""
    
    def setUp(self):
        # Mock the internal components
        with patch('src.data_processing.document_processor.PDFProcessor'), \
             patch('src.data_processing.document_processor.SemanticChunker'):
            self.processor = DocumentProcessor()
    
    @patch('src.data_processing.document_processor.Path.exists')
    def test_processing_pipeline_integration(self, mock_exists):
        """Test the complete processing pipeline"""
        mock_exists.return_value = True
        
        # Mock PDF processor
        self.processor.pdf_processor.extract_text = Mock(return_value=(
            "# Test Document\n\nThis is test content.",
            {"extraction_method": "test", "has_images": False, "has_tables": False}
        ))
        self.processor.pdf_processor.calculate_file_hash = Mock(return_value="abc123def456")
        self.processor.pdf_processor.get_page_count = Mock(return_value=1)
        
        # Mock semantic chunker
        self.processor.semantic_chunker.extract_section_headers = Mock(return_value=[
            ("Test Document", 0, 1)
        ])
        self.processor.semantic_chunker.semantic_chunking = Mock(return_value=[
            {
                'content': 'This is test content.',
                'paragraph_count': 1,
                'semantic_density': 1.0,
                'character_length': 21
            }
        ])
        self.processor.semantic_chunker.assign_section_context = Mock(side_effect=lambda chunks, headers, text: [
            {**chunk, 'section_header': 'Test Document', 'section_level': 1}
            for chunk in chunks
        ])
        
        # Test processing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            result = self.processor.process_document(temp_path)
            
            # Verify result structure
            self.assertIsInstance(result, ProcessingResult)
            self.assertEqual(len(result.chunks), 1)
            self.assertEqual(result.chunks[0].content, 'This is test content.')
            self.assertEqual(result.chunks[0].section_header, 'Test Document')
            self.assertIn('processing_time_seconds', result.processing_stats)
        finally:
            temp_path.unlink()

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_file_handling(self):
        """Test handling of missing PDF files"""
        processor = DocumentProcessor()
        non_existent_path = Path("non_existent_file.pdf")
        
        with self.assertRaises(FileNotFoundError):
            processor.process_document(non_existent_path)
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text"""
        with patch('src.data_processing.semantic_chunker.AutoTokenizer'), \
             patch('src.data_processing.semantic_chunker.AutoModel'):
            chunker = SemanticChunker(device="cpu")
            chunker.tokenizer = Mock()
            chunker.embedding_model = Mock()
            
            # Empty text should return empty chunks
            chunks = chunker.semantic_chunking("")
            self.assertEqual(len(chunks), 0)
            
            # Very short text should also return empty chunks
            chunks = chunker.semantic_chunking("short")
            self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    # Create a test suite
    unittest.main(verbosity=2)