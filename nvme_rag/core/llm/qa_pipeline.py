import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .ollama_client import OllamaClient, ChatMessage
from .query_translator import QueryTranslator, QueryAnalysis
from .answer_generator import AnswerGenerator, GeneratedAnswer, AnswerGenerationConfig
from ..retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalResult
from ..models.document import ProcessedChunk

logger = logging.getLogger(__name__)


@dataclass
class QASession:
    """Represents a question-answering session"""
    session_id: str
    chat_history: List[ChatMessage]
    created_at: datetime
    last_updated: datetime


@dataclass
class QAResult:
    """Complete question-answering result"""
    query: str
    query_analysis: QueryAnalysis
    retrieval_result: RetrievalResult
    generated_answer: GeneratedAnswer
    follow_up_questions: List[str]
    session_id: Optional[str]
    timestamp: datetime
    processing_time_seconds: float


class QAPipeline:
    """
    Complete Question-Answering pipeline that orchestrates:
    1. Query translation and analysis
    2. Context retrieval
    3. Answer generation
    4. Follow-up question generation
    """
    
    def __init__(self,
                 retrieval_pipeline: RetrievalPipeline,
                 llm_client: OllamaClient,
                 default_answer_config: AnswerGenerationConfig = None):
        """
        Initialize QA pipeline
        
        Args:
            retrieval_pipeline: Configured retrieval pipeline
            llm_client: Ollama LLM client
            default_answer_config: Default answer generation configuration
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_client = llm_client
        self.query_translator = QueryTranslator(llm_client)
        self.answer_generator = AnswerGenerator(llm_client)
        self.default_answer_config = default_answer_config or AnswerGenerationConfig()
        
        # Session management
        self.sessions: Dict[str, QASession] = {}
        
        logger.info("QA Pipeline initialized")
    
    def ask_question(self,
                    question: str,
                    session_id: Optional[str] = None,
                    answer_config: AnswerGenerationConfig = None,
                    retrieval_filters: Dict[str, Any] = None) -> QAResult:
        """
        Process a complete question-answering request
        
        Args:
            question: User's question
            session_id: Optional session ID for conversation continuity
            answer_config: Answer generation configuration
            retrieval_filters: Optional filters for retrieval
            
        Returns:
            Complete QA result with answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get or create session
            session = self._get_or_create_session(session_id)
            
            # Step 1: Analyze and translate query
            logger.info(f"Analyzing query: {question[:50]}...")
            query_analysis = self.query_translator.analyze_query(
                question, session.chat_history
            )
            
            # Step 2: Retrieve relevant context
            logger.info("Retrieving context...")
            retrieval_result = self.retrieval_pipeline.retrieve(
                query=query_analysis.expanded_query,
                chat_history=session.chat_history,
                filters=retrieval_filters
            )
            
            # Step 3: Generate answer
            logger.info("Generating answer...")
            config = answer_config or self.default_answer_config
            generated_answer = self.answer_generator.generate_answer(
                query_analysis=query_analysis,
                context_chunks=retrieval_result.context_chunks,
                chat_history=session.chat_history,
                config=config
            )
            
            # Step 4: Generate follow-up questions
            logger.info("Generating follow-up questions...")
            follow_up_questions = self.answer_generator.generate_follow_up_questions(
                query_analysis, generated_answer
            )
            
            # Step 5: Update session history
            self._update_session_history(session, question, generated_answer.answer)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            qa_result = QAResult(
                query=question,
                query_analysis=query_analysis,
                retrieval_result=retrieval_result,
                generated_answer=generated_answer,
                follow_up_questions=follow_up_questions,
                session_id=session.session_id,
                timestamp=datetime.now(),
                processing_time_seconds=processing_time
            )
            
            logger.info(f"QA completed in {processing_time:.2f}s, confidence={generated_answer.confidence:.2f}")
            return qa_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"QA pipeline failed: {e}")
            
            # Return error result
            return self._create_error_result(
                question, str(e), session_id, processing_time
            )
    
    def ask_follow_up(self,
                     follow_up_question: str,
                     previous_result: QAResult,
                     answer_config: AnswerGenerationConfig = None) -> QAResult:
        """
        Ask a follow-up question in the context of a previous result
        
        Args:
            follow_up_question: Follow-up question
            previous_result: Previous QA result for context
            answer_config: Answer generation configuration
            
        Returns:
            QA result for the follow-up question
        """
        # Use the same session to maintain context
        return self.ask_question(
            question=follow_up_question,
            session_id=previous_result.session_id,
            answer_config=answer_config
        )
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation in a session
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Conversation summary
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        # Count questions and answers
        questions = [msg for msg in session.chat_history if msg.role == "user"]
        answers = [msg for msg in session.chat_history if msg.role == "assistant"]
        
        summary = {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "total_exchanges": len(questions),
            "conversation_length": len(session.chat_history),
            "topics_discussed": self._extract_topics(session.chat_history),
            "session_duration_minutes": (session.last_updated - session.created_at).total_seconds() / 60
        }
        
        return summary
    
    def _get_or_create_session(self, session_id: Optional[str]) -> QASession:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        # Create new session
        import uuid
        new_session_id = session_id or str(uuid.uuid4())
        
        session = QASession(
            session_id=new_session_id,
            chat_history=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.sessions[new_session_id] = session
        logger.info(f"Created new QA session: {new_session_id}")
        
        return session
    
    def _update_session_history(self, session: QASession, question: str, answer: str):
        """Update session chat history"""
        session.chat_history.append(ChatMessage(role="user", content=question))
        session.chat_history.append(ChatMessage(role="assistant", content=answer))
        session.last_updated = datetime.now()
        
        # Keep only recent history to prevent context overflow
        max_history = 20  # Keep last 20 messages (10 exchanges)
        if len(session.chat_history) > max_history:
            session.chat_history = session.chat_history[-max_history:]
    
    def _extract_topics(self, chat_history: List[ChatMessage]) -> List[str]:
        """Extract main topics from chat history"""
        # Simple topic extraction based on common NVMe terms
        topics = set()
        
        all_content = " ".join([msg.content.lower() for msg in chat_history])
        
        topic_keywords = {
            "queues": ["queue", "submission", "completion", "doorbell"],
            "commands": ["command", "opcode", "execution", "admin"],
            "memory": ["memory", "dma", "prp", "sgl", "address"],
            "namespaces": ["namespace", "nsid", "format", "block"],
            "controller": ["controller", "identify", "features", "register"],
            "specification": ["spec", "standard", "compliance", "version"],
            "performance": ["performance", "latency", "throughput", "optimization"],
            "troubleshooting": ["error", "fail", "problem", "issue", "debug"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_content for keyword in keywords):
                topics.add(topic)
        
        return list(topics)
    
    def _create_error_result(self,
                           question: str,
                           error_message: str,
                           session_id: Optional[str],
                           processing_time: float) -> QAResult:
        """Create error result when QA pipeline fails"""
        
        # Create minimal query analysis
        from .query_translator import QueryAnalysis, QueryType
        error_analysis = QueryAnalysis(
            original_query=question,
            query_type=QueryType.FACTUAL,
            technical_terms=[],
            intent="error occurred",
            expanded_query=question,
            search_keywords=[],
            confidence=0.0
        )
        
        # Create empty retrieval result
        from ..retrieval.retrieval_pipeline import RetrievalResult, QueryContext
        error_context = QueryContext(
            original_query=question,
            enhanced_query=question,
            chat_history=[],
            filters={},
            user_preferences={}
        )
        
        error_retrieval = RetrievalResult(
            query_context=error_context,
            search_results=[],
            context_chunks=[],
            total_context_length=0,
            retrieval_stats={"error": error_message}
        )
        
        # Create error answer
        error_answer = GeneratedAnswer(
            answer=f"I apologize, but I encountered an error while processing your question: {error_message}",
            confidence=0.0,
            sources=[],
            reasoning="Error occurred during processing",
            query_analysis=error_analysis,
            context_used=0,
            generation_config=self.default_answer_config
        )
        
        return QAResult(
            query=question,
            query_analysis=error_analysis,
            retrieval_result=error_retrieval,
            generated_answer=error_answer,
            follow_up_questions=[],
            session_id=session_id,
            timestamp=datetime.now(),
            processing_time_seconds=processing_time
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len([s for s in self.sessions.values() 
                                  if (datetime.now() - s.last_updated).total_seconds() < 3600]),
            "total_questions_processed": sum(len([m for m in s.chat_history if m.role == "user"]) 
                                           for s in self.sessions.values()),
            "retrieval_config": self.retrieval_pipeline.get_config(),
            "average_session_length": sum(len(s.chat_history) for s in self.sessions.values()) / max(1, len(self.sessions))
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def clear_all_sessions(self):
        """Clear all sessions"""
        session_count = len(self.sessions)
        self.sessions.clear()
        logger.info(f"Cleared {session_count} sessions")
    
    def export_qa_result(self, qa_result: QAResult) -> Dict[str, Any]:
        """Export QA result to dictionary format"""
        return {
            "query": qa_result.query,
            "answer": qa_result.generated_answer.answer,
            "confidence": qa_result.generated_answer.confidence,
            "sources": qa_result.generated_answer.sources,
            "follow_up_questions": qa_result.follow_up_questions,
            "query_analysis": {
                "type": qa_result.query_analysis.query_type.value,
                "intent": qa_result.query_analysis.intent,
                "technical_terms": qa_result.query_analysis.technical_terms,
                "confidence": qa_result.query_analysis.confidence
            },
            "retrieval_stats": qa_result.retrieval_result.retrieval_stats,
            "context_chunks_used": qa_result.generated_answer.context_used,
            "processing_time_seconds": qa_result.processing_time_seconds,
            "timestamp": qa_result.timestamp.isoformat()
        }