"""
Core functionality for NVMe RAG system.
"""

# Lazy imports to avoid heavy dependencies during CLI import
def get_components():
    """Get all core components (lazy import)."""
    from nvme_rag.core.pipeline.integration import RAGPipelineIntegration
    from nvme_rag.core.llm.qa_pipeline import QAPipeline
    from nvme_rag.core.retrieval.retrieval_pipeline import RetrievalPipeline
    from nvme_rag.core.vector_store.chroma_store import ChromaVectorStore
    from nvme_rag.core.data_processing.document_processor import DocumentProcessor
    
    return {
        "RAGPipelineIntegration": RAGPipelineIntegration,
        "QAPipeline": QAPipeline,
        "RetrievalPipeline": RetrievalPipeline,
        "ChromaVectorStore": ChromaVectorStore,
        "DocumentProcessor": DocumentProcessor,
    }

__all__ = ["get_components"]