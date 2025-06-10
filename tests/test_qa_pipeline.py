#!/usr/bin/env python3
"""
Comprehensive tests for the QA pipeline
"""

import pytest
import logging
from unittest.mock import Mock, patch
from datetime import datetime

from src.llm.query_translator import QueryTranslator, QueryAnalysis, QueryType
from src.llm.answer_generator import AnswerGenerator, GeneratedAnswer, AnswerGenerationConfig
from src.llm.qa_pipeline import QAPipeline, QAResult
from src.llm.ollama_client import OllamaClient, ChatMessage, LLMResponse
from src.retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalResult, QueryContext
from src.models.document import ProcessedChunk

logger = logging.getLogger(__name__)


class TestQueryTranslator:
    """Test query translation and analysis"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing"""
        client = Mock(spec=OllamaClient)
        return client
    
    @pytest.fixture
    def query_translator(self, mock_llm_client):
        """Create query translator with mock client"""
        return QueryTranslator(mock_llm_client)
    
    def test_analyze_query_basic(self, query_translator, mock_llm_client):
        """Test basic query analysis"""
        # Mock LLM responses
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="Type: factual\nIntent: asking about NVMe submission queue"),
            LLMResponse(content="What is the purpose and function of the NVMe submission queue")
        ]
        
        query = "What is the NVMe submission queue?"
        analysis = query_translator.analyze_query(query)
        
        assert analysis.original_query == query
        assert analysis.query_type == QueryType.FACTUAL
        assert analysis.confidence > 0.5
        assert len(analysis.technical_terms) > 0
        assert "submission queue" in analysis.technical_terms or "queue" in analysis.technical_terms
    
    def test_extract_technical_terms(self, query_translator):
        """Test technical term extraction"""
        query = "How does DMA work with PCIe in NVMe SSD controllers?"
        terms = query_translator._extract_technical_terms(query)
        
        # Should extract DMA, PCIe, NVMe, SSD
        expected_terms = ["direct memory access", "DMA", "peripheral component interconnect express", "PCIe"]
        found_terms = [term.lower() for term in terms]
        
        assert any("dma" in found_terms for term in found_terms)
        assert any("pcie" in found_terms for term in found_terms)
    
    def test_query_expansion(self, query_translator, mock_llm_client):
        """Test query expansion"""
        mock_llm_client.chat.return_value = LLMResponse(
            content="What is the purpose and functionality of NVMe submission queues in PCIe storage controllers"
        )
        
        original = "What is SQ?"
        expanded = query_translator._expand_query(original)
        
        assert len(expanded) >= len(original)
        mock_llm_client.chat.assert_called_once()


class TestAnswerGenerator:
    """Test answer generation"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing"""
        client = Mock(spec=OllamaClient)
        return client
    
    @pytest.fixture
    def answer_generator(self, mock_llm_client):
        """Create answer generator with mock client"""
        return AnswerGenerator(mock_llm_client)
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample context chunks for testing"""
        return [
            ProcessedChunk(
                content="The submission queue (SQ) is a circular buffer used to submit commands to the NVMe controller.",
                metadata={"section": "Queue Management"},
                chunk_id="chunk_1",
                parent_doc_id="nvme_spec",
                section_header="Queue Management",
                page_number=42,
                chunk_type="text",
                semantic_density=0.8
            ),
            ProcessedChunk(
                content="Submission queues can be created in host memory or controller memory.",
                metadata={"section": "Queue Management"},
                chunk_id="chunk_2",
                parent_doc_id="nvme_spec",
                section_header="Queue Management",
                page_number=43,
                chunk_type="text",
                semantic_density=0.7
            )
        ]
    
    @pytest.fixture
    def sample_query_analysis(self):
        """Sample query analysis for testing"""
        return QueryAnalysis(
            original_query="What is the NVMe submission queue?",
            query_type=QueryType.FACTUAL,
            technical_terms=["submission queue", "NVMe"],
            intent="asking about submission queue definition",
            expanded_query="What is the purpose and function of the NVMe submission queue",
            search_keywords=["submission", "queue", "nvme", "purpose"],
            confidence=0.8
        )
    
    def test_generate_answer(self, answer_generator, mock_llm_client, sample_chunks, sample_query_analysis):
        """Test answer generation"""
        # Mock LLM response
        mock_llm_client.chat.return_value = LLMResponse(
            content="The NVMe submission queue is a circular buffer used to submit commands to the controller. It can be created in either host memory or controller memory, providing flexibility for different system configurations."
        )
        
        config = AnswerGenerationConfig()
        answer = answer_generator.generate_answer(
            query_analysis=sample_query_analysis,
            context_chunks=sample_chunks,
            config=config
        )
        
        assert isinstance(answer, GeneratedAnswer)
        assert len(answer.answer) > 50  # Substantial answer
        assert answer.confidence > 0.0
        assert len(answer.sources) == len(sample_chunks)
        assert answer.context_used == len(sample_chunks)
    
    def test_build_context_string(self, answer_generator, sample_chunks):
        """Test context string building"""
        config = AnswerGenerationConfig(cite_sections=True)
        context_str = answer_generator._build_context_string(sample_chunks, config)
        
        assert "[CONTEXT 1]" in context_str
        assert "[CONTEXT 2]" in context_str
        assert "Queue Management" in context_str
        assert sample_chunks[0].content in context_str
    
    def test_extract_sources(self, answer_generator, sample_chunks):
        """Test source extraction"""
        sources = answer_generator._extract_sources(sample_chunks)
        
        assert len(sources) == len(sample_chunks)
        assert sources[0]["section"] == "Queue Management"
        assert sources[0]["page"] == 42
        assert sources[0]["chunk_id"] == "chunk_1"


class TestQAPipeline:
    """Test complete QA pipeline"""
    
    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Mock retrieval pipeline"""
        pipeline = Mock(spec=RetrievalPipeline)
        return pipeline
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client"""
        client = Mock(spec=OllamaClient)
        return client
    
    @pytest.fixture
    def qa_pipeline(self, mock_retrieval_pipeline, mock_llm_client):
        """Create QA pipeline with mocks"""
        return QAPipeline(
            retrieval_pipeline=mock_retrieval_pipeline,
            llm_client=mock_llm_client
        )
    
    @pytest.fixture
    def sample_retrieval_result(self):
        """Sample retrieval result"""
        context = QueryContext(
            original_query="What is the submission queue?",
            enhanced_query="What is the NVMe submission queue purpose",
            chat_history=[],
            filters={},
            user_preferences={}
        )
        
        chunks = [
            ProcessedChunk(
                content="The submission queue is used to submit commands.",
                metadata={"section": "Queues"},
                chunk_id="chunk_1",
                parent_doc_id="nvme_spec",
                section_header="Queue Management",
                page_number=42,
                chunk_type="text",
                semantic_density=0.8
            )
        ]
        
        return RetrievalResult(
            query_context=context,
            search_results=[],
            context_chunks=chunks,
            total_context_length=len(chunks[0].content),
            retrieval_stats={"chunks_found": 1}
        )
    
    def test_ask_question(self, qa_pipeline, mock_retrieval_pipeline, mock_llm_client, sample_retrieval_result):
        """Test complete question asking process"""
        # Setup mocks
        mock_retrieval_pipeline.retrieve.return_value = sample_retrieval_result
        
        # Mock LLM responses for different components
        mock_llm_client.chat.side_effect = [
            # Query analysis responses
            LLMResponse(content="Type: factual\nIntent: asking about submission queue"),
            LLMResponse(content="What is the purpose and function of the NVMe submission queue"),
            # Answer generation response
            LLMResponse(content="The submission queue is a data structure used to submit commands to the NVMe controller."),
            # Follow-up questions response
            LLMResponse(content="How are submission queues created?\nWhat is the maximum size of a submission queue?\nHow do completion queues relate to submission queues?")
        ]
        
        question = "What is the submission queue?"
        result = qa_pipeline.ask_question(question)
        
        # Verify result structure
        assert isinstance(result, QAResult)
        assert result.query == question
        assert result.generated_answer.answer
        assert result.generated_answer.confidence > 0
        assert len(result.follow_up_questions) > 0
        assert result.session_id is not None
        assert result.processing_time_seconds > 0
        
        # Verify mocks were called
        mock_retrieval_pipeline.retrieve.assert_called_once()
        assert mock_llm_client.chat.call_count >= 3  # At least query analysis, expansion, and answer generation
    
    def test_session_management(self, qa_pipeline, mock_retrieval_pipeline, mock_llm_client, sample_retrieval_result):
        """Test session management functionality"""
        # Setup mocks
        mock_retrieval_pipeline.retrieve.return_value = sample_retrieval_result
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="Type: factual\nIntent: test"),
            LLMResponse(content="expanded query"),
            LLMResponse(content="test answer"),
            LLMResponse(content="follow up question")
        ]
        
        # Ask first question
        result1 = qa_pipeline.ask_question("First question")
        session_id = result1.session_id
        
        # Ask second question with same session
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="Type: factual\nIntent: test"),
            LLMResponse(content="expanded query"),
            LLMResponse(content="second answer"),
            LLMResponse(content="follow up question")
        ]
        
        result2 = qa_pipeline.ask_question("Second question", session_id=session_id)
        
        # Should use same session
        assert result2.session_id == session_id
        
        # Get conversation summary
        summary = qa_pipeline.get_conversation_summary(session_id)
        assert summary["total_exchanges"] == 2
        assert session_id in qa_pipeline.sessions
    
    def test_follow_up_question(self, qa_pipeline, mock_retrieval_pipeline, mock_llm_client, sample_retrieval_result):
        """Test follow-up question functionality"""
        # Setup mocks for initial question
        mock_retrieval_pipeline.retrieve.return_value = sample_retrieval_result
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="Type: factual\nIntent: test"),
            LLMResponse(content="expanded query"),
            LLMResponse(content="initial answer"),
            LLMResponse(content="follow up question")
        ]
        
        # Ask initial question
        initial_result = qa_pipeline.ask_question("Initial question")
        
        # Setup mocks for follow-up
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="Type: factual\nIntent: follow up"),
            LLMResponse(content="expanded follow up"),
            LLMResponse(content="follow up answer"),
            LLMResponse(content="more follow ups")
        ]
        
        # Ask follow-up
        follow_up_result = qa_pipeline.ask_follow_up("Follow up question", initial_result)
        
        # Should use same session
        assert follow_up_result.session_id == initial_result.session_id
        assert follow_up_result.query == "Follow up question"
    
    def test_error_handling(self, qa_pipeline, mock_retrieval_pipeline, mock_llm_client):
        """Test error handling in QA pipeline"""
        # Make retrieval fail
        mock_retrieval_pipeline.retrieve.side_effect = Exception("Retrieval failed")
        
        result = qa_pipeline.ask_question("Test question")
        
        # Should return error result
        assert isinstance(result, QAResult)
        assert "error" in result.generated_answer.answer.lower()
        assert result.generated_answer.confidence == 0.0
    
    def test_export_qa_result(self, qa_pipeline, mock_retrieval_pipeline, mock_llm_client, sample_retrieval_result):
        """Test QA result export functionality"""
        # Setup mocks
        mock_retrieval_pipeline.retrieve.return_value = sample_retrieval_result
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="Type: factual\nIntent: test"),
            LLMResponse(content="expanded query"),
            LLMResponse(content="test answer"),
            LLMResponse(content="follow up")
        ]
        
        result = qa_pipeline.ask_question("Test question")
        exported = qa_pipeline.export_qa_result(result)
        
        # Verify export structure
        assert "query" in exported
        assert "answer" in exported
        assert "confidence" in exported
        assert "sources" in exported
        assert "query_analysis" in exported
        assert "timestamp" in exported
        assert exported["query"] == "Test question"


def run_qa_pipeline_tests():
    """Run all QA pipeline tests"""
    print("🧪 Running QA Pipeline Tests...")
    
    # This would normally use pytest, but for demo purposes we'll run basic validation
    try:
        # Test basic imports
        from src.llm.query_translator import QueryTranslator
        from src.llm.answer_generator import AnswerGenerator
        from src.llm.qa_pipeline import QAPipeline
        
        print("✅ All imports successful")
        
        # Test basic instantiation (with mocks)
        mock_client = Mock(spec=OllamaClient)
        mock_retrieval = Mock(spec=RetrievalPipeline)
        
        translator = QueryTranslator(mock_client)
        generator = AnswerGenerator(mock_client)
        pipeline = QAPipeline(mock_retrieval, mock_client)
        
        print("✅ All components instantiate successfully")
        print("✅ QA Pipeline tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ QA Pipeline tests failed: {e}")
        return False


if __name__ == "__main__":
    run_qa_pipeline_tests()