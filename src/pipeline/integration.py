import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.data_processing.document_processor import DocumentProcessor
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaClient, OllamaConfig
from src.retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalConfig
from src.llm.qa_pipeline import QAPipeline, QAResult, AnswerGenerationConfig
from src.models.document import ProcessedChunk, ProcessingResult

logger = logging.getLogger(__name__)

class RAGPipelineIntegration:
    """
    Complete RAG pipeline integration that combines:
    - Document processing (Phase 1)
    - Vector storage and retrieval (Phase 2)
    - Ready for generation capabilities (Phase 3)
    """
    
    def __init__(self,
                 vector_store_path: str = "data/vector_store",
                 embedding_config: EmbeddingConfig = None,
                 ollama_config: OllamaConfig = None,
                 retrieval_config: RetrievalConfig = None,
                 answer_config: AnswerGenerationConfig = None):
        """
        Initialize complete RAG pipeline
        
        Args:
            vector_store_path: Path for ChromaDB persistence
            embedding_config: Embedding generation configuration
            ollama_config: Ollama LLM configuration
            retrieval_config: Retrieval pipeline configuration
            answer_config: Answer generation configuration
        """
        self.vector_store_path = vector_store_path
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            collection_name="nvme_rag_chunks",
            persist_directory=vector_store_path,
            embedding_config=embedding_config
        )
        
        # Initialize LLM client
        self.llm_client = OllamaClient(ollama_config)
        
        # Initialize retrieval pipeline
        self.retrieval_pipeline = RetrievalPipeline(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            config=retrieval_config
        )
        
        # Initialize QA pipeline
        self.qa_pipeline = QAPipeline(
            retrieval_pipeline=self.retrieval_pipeline,
            llm_client=self.llm_client,
            default_answer_config=answer_config
        )
        
        logger.info("RAG pipeline integration initialized")
    
    def process_and_index_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Complete pipeline: process PDF and add to vector store
        
        Args:
            pdf_path: Path to PDF document
            
        Returns:
            Dictionary with processing and indexing results
        """
        logger.info(f"Processing and indexing document: {pdf_path}")
        
        try:
            # Step 1: Process document
            processing_result = self.document_processor.process_document(pdf_path)
            
            # Step 2: Add chunks to vector store
            chunk_ids = self.vector_store.add_chunks(processing_result.chunks)
            
            # Step 3: Get vector store stats
            vector_stats = self.vector_store.get_stats()
            
            result = {
                "processing_stats": processing_result.processing_stats,
                "document_metadata": processing_result.metadata.to_dict(),
                "chunks_processed": len(processing_result.chunks),
                "chunks_indexed": len(chunk_ids),
                "vector_store_stats": vector_stats,
                "success": True
            }
            
            logger.info(f"Successfully processed and indexed {len(chunk_ids)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process and index document: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def process_multiple_documents(self, pdf_paths: List[Path]) -> Dict[str, Any]:
        """
        Process and index multiple documents
        
        Args:
            pdf_paths: List of PDF paths to process
            
        Returns:
            Summary of processing results
        """
        logger.info(f"Processing {len(pdf_paths)} documents")
        
        results = []
        successful_docs = 0
        total_chunks = 0
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_and_index_document(pdf_path)
                results.append({
                    "document": str(pdf_path),
                    "result": result
                })
                
                if result.get("success", False):
                    successful_docs += 1
                    total_chunks += result.get("chunks_indexed", 0)
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results.append({
                    "document": str(pdf_path),
                    "result": {"error": str(e), "success": False}
                })
        
        summary = {
            "total_documents": len(pdf_paths),
            "successful_documents": successful_docs,
            "total_chunks_indexed": total_chunks,
            "processing_results": results,
            "vector_store_stats": self.vector_store.get_stats()
        }
        
        logger.info(f"Batch processing completed: {successful_docs}/{len(pdf_paths)} documents successful")
        return summary
    
    def search_and_retrieve(self, 
                           query: str,
                           chat_history: List = None,
                           filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search and retrieve relevant context for a query
        
        Args:
            query: User query
            chat_history: Previous conversation messages
            filters: Metadata filters for search
            
        Returns:
            Retrieval results with context and metadata
        """
        try:
            # Perform retrieval
            retrieval_result = self.retrieval_pipeline.retrieve(
                query=query,
                chat_history=chat_history,
                filters=filters
            )
            
            # Format context for easy consumption
            formatted_context = self._format_context(retrieval_result.context_chunks)
            
            result = {
                "query": {
                    "original": retrieval_result.query_context.original_query,
                    "enhanced": retrieval_result.query_context.enhanced_query
                },
                "context": formatted_context,
                "chunks": [chunk.to_dict() for chunk in retrieval_result.context_chunks],
                "stats": retrieval_result.retrieval_stats,
                "total_context_length": retrieval_result.total_context_length,
                "success": True
            }
            
            logger.info(f"Retrieved {len(retrieval_result.context_chunks)} chunks for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Search and retrieval failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _format_context(self, chunks: List[ProcessedChunk]) -> str:
        """Format chunks into readable context string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            section_info = f"Section: {chunk.section_header}" if chunk.section_header != "Document Root" else ""
            
            chunk_text = f"""[CONTEXT {i}]
{section_info}
{chunk.content}

"""
            context_parts.append(chunk_text)
        
        return "\n".join(context_parts)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the RAG pipeline
        
        Returns:
            Status information for all components
        """
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_stats()
            
            # Get LLM health check
            llm_health = self.llm_client.health_check()
            
            # Get retrieval config
            retrieval_config = self.retrieval_pipeline.get_config()
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "document_processor": {
                        "status": "ready",
                        "device": str(self.document_processor.semantic_chunker.device)
                    },
                    "vector_store": {
                        "status": "ready",
                        "total_chunks": vector_stats.get("total_chunks", 0),
                        "collection_name": vector_stats.get("collection_name"),
                        "embedding_model": vector_stats.get("embedding_model")
                    },
                    "llm_client": {
                        "status": llm_health.get("status", "unknown"),
                        "model": llm_health.get("model"),
                        "response_time": llm_health.get("response_time_seconds")
                    },
                    "retrieval_pipeline": {
                        "status": "ready",
                        "strategy": retrieval_config.get("strategy"),
                        "top_k": retrieval_config.get("top_k")
                    }
                },
                "overall_status": "ready" if llm_health.get("status") == "healthy" else "degraded"
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    def search_by_section(self, section_name: str, top_k: int = 5) -> List[ProcessedChunk]:
        """
        Search for chunks by section name
        
        Args:
            section_name: Name of section to search
            top_k: Maximum number of chunks to return
            
        Returns:
            List of chunks from the specified section
        """
        try:
            filters = {"section_header": section_name}
            search_result = self.search_and_retrieve(
                query=f"content from {section_name}",
                filters=filters
            )
            
            if search_result.get("success", False):
                chunks_data = search_result.get("chunks", [])
                chunks = [ProcessedChunk.from_dict(chunk_data) for chunk_data in chunks_data[:top_k]]
                return chunks
            
            return []
            
        except Exception as e:
            logger.error(f"Section search failed: {e}")
            return []
    
    def get_document_sections(self, doc_id: str) -> List[str]:
        """
        Get all sections for a specific document
        
        Args:
            doc_id: Document ID to get sections for
            
        Returns:
            List of section names
        """
        try:
            filters = {"parent_doc_id": doc_id}
            search_result = self.search_and_retrieve(
                query="all sections",
                filters=filters
            )
            
            if search_result.get("success", False):
                chunks_data = search_result.get("chunks", [])
                sections = list(set(chunk["metadata"]["section_header"] for chunk in chunks_data))
                return sorted(sections)
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get document sections: {e}")
            return []
    
    def ask_question(self, 
                    question: str,
                    session_id: Optional[str] = None,
                    answer_config: AnswerGenerationConfig = None,
                    retrieval_filters: Dict[str, Any] = None) -> QAResult:
        """
        Ask a question and get a comprehensive answer with context
        
        Args:
            question: User's question
            session_id: Optional session ID for conversation continuity
            answer_config: Answer generation configuration
            retrieval_filters: Optional filters for retrieval
            
        Returns:
            Complete QA result with answer and metadata
        """
        return self.qa_pipeline.ask_question(
            question=question,
            session_id=session_id,
            answer_config=answer_config,
            retrieval_filters=retrieval_filters
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
        return self.qa_pipeline.ask_follow_up(
            follow_up_question=follow_up_question,
            previous_result=previous_result,
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
        return self.qa_pipeline.get_conversation_summary(session_id)
    
    def export_qa_result(self, qa_result: QAResult) -> Dict[str, Any]:
        """
        Export QA result to dictionary format
        
        Args:
            qa_result: QA result to export
            
        Returns:
            Dictionary representation of QA result
        """
        return self.qa_pipeline.export_qa_result(qa_result)
    
    def cleanup(self):
        """Cleanup all pipeline components"""
        try:
            if hasattr(self, 'document_processor'):
                # Cleanup document processor if it has cleanup method
                pass
            
            if hasattr(self, 'qa_pipeline'):
                self.qa_pipeline.clear_all_sessions()
            
            if hasattr(self, 'vector_store'):
                self.vector_store.cleanup()
            
            if hasattr(self, 'llm_client'):
                self.llm_client.cleanup()
            
            logger.info("RAG pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")