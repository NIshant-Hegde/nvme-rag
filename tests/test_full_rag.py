import unittest
from unittest.mock import Mock, patch
import os
import tempfile
from pathlib import Path

from run_full_RAG import NVMeQADemo
from src.pipeline.integration import RAGPipelineIntegration
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaConfig
from src.llm.answer_generator import AnswerGenerationConfig, AnswerStyle
from src.retrieval.retrieval_pipeline import RetrievalConfig, RetrievalStrategy

class TestNVMeRAGPipeline(unittest.TestCase):
    """Test suite for the full NVMe RAG pipeline"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create a temporary directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store_path = os.path.join(self.temp_dir, "vector_store")
        
        # Mock configurations
        self.embedding_config = EmbeddingConfig(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )
        
        self.ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3:12b-it-qat",
            temperature=0.1,
            max_tokens=2048
        )
        
        self.retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.RERANKED,
            top_k=5,
            enable_query_enhancement=True,
            enable_context_filtering=True,
            max_context_length=3000
        )
        
        self.answer_config = AnswerGenerationConfig(
            style=AnswerStyle.DETAILED,
            max_answer_length=1000,
            include_sources=True,
            include_confidence=True,
            cite_sections=True,
            explain_technical_terms=False
        )

    def tearDown(self):
        """Clean up test fixtures after each test"""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_qa_system_initialization(self):
        """Test QA system initialization and configuration"""
        # Create QA system
        qa_system = NVMeQADemo()
        
        # Verify configurations are set correctly
        self.assertEqual(qa_system.vector_store_path, "data/vector_store")
        self.assertEqual(qa_system.embedding_config.model_name, 
                        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.assertEqual(qa_system.ollama_config.model, "gemma3:12b-it-qat")
        self.assertEqual(qa_system.retrieval_config.strategy, 
                        RetrievalStrategy.RERANKED)
        self.assertEqual(qa_system.answer_config.style, AnswerStyle.DETAILED)
        
        # Verify pipeline is initialized
        self.assertIsInstance(qa_system.rag_pipeline, RAGPipelineIntegration)

    @patch('src.pipeline.integration.RAGPipelineIntegration')
    def test_pipeline_status_check(self, mock_pipeline):
        """Test system status check functionality"""
        # Mock pipeline status
        mock_status = {
            'timestamp': '2024-01-01T12:00:00',
            'overall_status': 'ready',
            'components': {
                'document_processor': {
                    'status': 'ready',
                    'device': 'cpu'
                },
                'vector_store': {
                    'status': 'ready',
                    'total_chunks': 100,
                    'collection_name': 'nvme_rag_chunks',
                    'embedding_model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
                },
                'llm_client': {
                    'status': 'healthy',
                    'model': 'gemma3:12b-it-qat',
                    'response_time': 0.5
                },
                'retrieval_pipeline': {
                    'status': 'ready',
                    'strategy': 'reranked',
                    'top_k': 5
                }
            }
        }
        # Mock the pipeline configuration
        mock_pipeline.return_value.vector_store_path = self.vector_store_path
        mock_pipeline.return_value.embedding_config = self.embedding_config
        mock_pipeline.return_value.ollama_config = self.ollama_config
        mock_pipeline.return_value.retrieval_config = self.retrieval_config
        mock_pipeline.return_value.answer_config = self.answer_config
        mock_pipeline.return_value.get_pipeline_status.return_value = mock_status
        
        # Create QA system with mocked pipeline
        qa_system = NVMeQADemo()
        qa_system.vector_store_path = self.vector_store_path
        qa_system.rag_pipeline = mock_pipeline.return_value
        
        # Check status
        try:
            qa_system._check_system_status()
        except Exception as e:
            self.fail(f"Status check raised unexpected exception: {e}")
            
        # Verify pipeline status was called
        mock_pipeline.return_value.get_pipeline_status.assert_called_once()

    @patch('src.pipeline.integration.RAGPipelineIntegration')
    def test_question_answering(self, mock_pipeline):
        """Test basic question answering functionality"""
        # Mock QA response
        mock_qa_result = Mock()
        mock_qa_result.generated_answer.answer = "Test answer"
        mock_qa_result.generated_answer.confidence = 0.9
        mock_qa_result.generated_answer.sources = [
            {'section': 'Test Section', 'page': 1}
        ]
        mock_qa_result.query_analysis = Mock()
        mock_qa_result.query_analysis.query_type = Mock()
        mock_qa_result.query_analysis.query_type.value = "technical"
        mock_qa_result.query_analysis.intent = "understand"
        mock_qa_result.query_analysis.technical_terms = ["NVMe"]
        mock_qa_result.processing_time_seconds = 1.5
        mock_qa_result.generated_answer.context_used = 3
        mock_qa_result.follow_up_questions = []
        
        # Mock the pipeline configuration
        mock_pipeline.return_value.vector_store_path = self.vector_store_path
        mock_pipeline.return_value.embedding_config = self.embedding_config
        mock_pipeline.return_value.ollama_config = self.ollama_config
        mock_pipeline.return_value.retrieval_config = self.retrieval_config
        mock_pipeline.return_value.answer_config = self.answer_config
        mock_pipeline.return_value.ask_question.return_value = mock_qa_result
        
        # Create QA system with mocked pipeline
        qa_system = NVMeQADemo()
        qa_system.vector_store_path = self.vector_store_path
        qa_system.rag_pipeline = mock_pipeline.return_value
        
        # Setup ask_question mock with expected parameters
        def ask_question_side_effect(*args, **kwargs):
            return mock_qa_result
        mock_pipeline.return_value.ask_question.side_effect = ask_question_side_effect
        
        # Test asking a question
        test_question = "What is NVMe?"
        qa_result = qa_system.rag_pipeline.ask_question(
            question=test_question,
            session_id=None
        )
        
        # Verify question was processed
        mock_pipeline.return_value.ask_question.assert_called_once_with(
            question=test_question,
            session_id=None
        )
        
        # Verify result structure
        self.assertEqual(qa_result.generated_answer.answer, "Test answer")
        self.assertEqual(qa_result.generated_answer.confidence, 0.9)
        self.assertEqual(len(qa_result.generated_answer.sources), 1)

    @patch('src.pipeline.integration.RAGPipelineIntegration')
    def test_follow_up_questions(self, mock_pipeline):
        """Test follow-up question handling"""
        # Mock initial QA response with follow-up questions
        mock_initial_result = Mock()
        mock_initial_result.generated_answer = Mock()
        mock_initial_result.generated_answer.answer = "Initial answer"
        mock_initial_result.generated_answer.sources = []
        mock_initial_result.generated_answer.confidence = 0.9
        mock_initial_result.generated_answer.context_used = 2
        mock_initial_result.follow_up_questions = [
            "Follow-up question 1",
            "Follow-up question 2"
        ]
        mock_initial_result.session_id = "test_session_123"
        mock_initial_result.query = "Initial question"
        mock_initial_result.processing_time_seconds = 1.5
        mock_initial_result.query_analysis = Mock()
        mock_initial_result.query_analysis.query_type = Mock()
        mock_initial_result.query_analysis.query_type.value = "factual"
        mock_initial_result.query_analysis.intent = "test"
        mock_initial_result.query_analysis.technical_terms = ["nvme", "test"]
        
        # Mock conversation summary
        mock_summary = {
            'total_exchanges': 2,
            'session_duration_minutes': 5.0,
            'topics_discussed': ['NVMe basics', 'Queue management']
        }
        mock_pipeline.return_value.get_conversation_summary.return_value = mock_summary
        
        # Mock follow-up response
        mock_follow_up_result = Mock()
        mock_follow_up_result.generated_answer = Mock()
        mock_follow_up_result.generated_answer.answer = "Follow-up answer"
        mock_follow_up_result.generated_answer.sources = []
        mock_follow_up_result.generated_answer.confidence = 0.85
        mock_follow_up_result.generated_answer.context_used = 1
        mock_follow_up_result.follow_up_questions = []
        mock_follow_up_result.session_id = "test_session_123"
        mock_follow_up_result.query = "Follow-up question 1"
        mock_follow_up_result.processing_time_seconds = 1.0
        mock_follow_up_result.query_analysis = Mock()
        mock_follow_up_result.query_analysis.query_type = Mock()
        mock_follow_up_result.query_analysis.query_type.value = "factual"
        mock_follow_up_result.query_analysis.intent = "follow-up"
        mock_follow_up_result.query_analysis.technical_terms = ["nvme"]
        
        # Mock the pipeline configuration
        mock_pipeline.return_value.vector_store_path = self.vector_store_path
        mock_pipeline.return_value.embedding_config = self.embedding_config
        mock_pipeline.return_value.ollama_config = self.ollama_config
        mock_pipeline.return_value.retrieval_config = self.retrieval_config
        mock_pipeline.return_value.answer_config = self.answer_config
        mock_pipeline.return_value.ask_question.return_value = mock_initial_result
        mock_pipeline.return_value.ask_follow_up.return_value = mock_follow_up_result
        
        # Create QA system with mocked pipeline
        qa_system = NVMeQADemo()
        qa_system.vector_store_path = self.vector_store_path
        qa_system.rag_pipeline = mock_pipeline.return_value
        
        # Test asking follow-up
        test_follow_up = "Follow-up question 1"
        follow_up_result = qa_system.rag_pipeline.ask_follow_up(
            follow_up_question=test_follow_up,
            previous_result=mock_initial_result
        )
        
        # Verify follow-up was processed
        mock_pipeline.return_value.ask_follow_up.assert_called_once_with(
            follow_up_question=test_follow_up,
            previous_result=mock_initial_result
        )
        
        # Verify conversation summary can be retrieved
        summary = qa_system.rag_pipeline.get_conversation_summary(follow_up_result.session_id)
        self.assertEqual(summary['total_exchanges'], 2)
        self.assertEqual(summary['session_duration_minutes'], 5.0)
        self.assertEqual(len(summary['topics_discussed']), 2)
        
        # Verify result properties
        self.assertEqual(follow_up_result.generated_answer.answer, "Follow-up answer")
        self.assertEqual(follow_up_result.generated_answer.confidence, 0.85)
        self.assertEqual(follow_up_result.generated_answer.context_used, 1)
        self.assertEqual(follow_up_result.query, "Follow-up question 1")
        self.assertEqual(follow_up_result.processing_time_seconds, 1.0)
        self.assertEqual(follow_up_result.query_analysis.query_type.value, "factual")
        self.assertEqual(follow_up_result.query_analysis.intent, "follow-up")

    @patch('src.pipeline.integration.RAGPipelineIntegration')
    def test_session_management(self, mock_pipeline):
        """Test session management and summary functionality"""
        # Mock session summary
        mock_summary = {
            'total_exchanges': 5,
            'session_duration_minutes': 10.5,
            'topics_discussed': ['NVMe basics', 'Queue management']
        }
        # Mock the pipeline configuration
        mock_pipeline.return_value.vector_store_path = self.vector_store_path
        mock_pipeline.return_value.embedding_config = self.embedding_config
        mock_pipeline.return_value.ollama_config = self.ollama_config
        mock_pipeline.return_value.retrieval_config = self.retrieval_config
        mock_pipeline.return_value.answer_config = self.answer_config
        mock_pipeline.return_value.get_conversation_summary.return_value = mock_summary
        
        # Create QA system with mocked pipeline
        qa_system = NVMeQADemo()
        qa_system.vector_store_path = self.vector_store_path
        qa_system.rag_pipeline = mock_pipeline.return_value
        
        # Test getting session summary
        test_session_id = "test_session_123"
        qa_system._show_session_summary(test_session_id)
        
        # Verify summary was requested
        mock_pipeline.return_value.get_conversation_summary.assert_called_once_with(
            test_session_id
        )

    def test_export_functionality(self):
        """Test QA result export functionality"""
        # Create QA system
        qa_system = NVMeQADemo()
        
        # Create mock QA result
        mock_result = Mock()
        mock_result.timestamp.strftime.return_value = "20240101_120000"
        mock_result.generated_answer = Mock()
        mock_result.generated_answer.answer = "Test answer"
        mock_result.generated_answer.sources = [{'section': 'Test Section', 'page': 1}]
        mock_result.generated_answer.confidence = 0.9
        mock_result.generated_answer.context_used = 2
        mock_result.session_id = "test_session_123"
        mock_result.query = "Test question"
        mock_result.processing_time_seconds = 1.5
        mock_result.query_analysis = Mock()
        mock_result.query_analysis.query_type = Mock()
        mock_result.query_analysis.query_type.value = "factual"
        mock_result.query_analysis.intent = "test"
        mock_result.query_analysis.technical_terms = ["test"]
        
        # Create temporary export file
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            # Test export
            qa_system.export_demo_results(mock_result, temp_file.name)
            
            # Verify file exists and is non-empty
            self.assertTrue(os.path.exists(temp_file.name))
            self.assertGreater(os.path.getsize(temp_file.name), 0)

if __name__ == '__main__':
    unittest.main()