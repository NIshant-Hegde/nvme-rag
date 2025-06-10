"""
NVMe RAG: A professional Retrieval-Augmented Generation system for NVMe specifications.
"""

__version__ = "1.0.0"
__author__ = "NVMe RAG Development Team"
__email__ = "nvme-rag@example.com"
__description__ = "Professional RAG system for NVMe specification question-answering"

# Lazy imports to avoid heavy dependencies during CLI usage
def get_rag_pipeline():
    """Get RAGPipelineIntegration class (lazy import)."""
    from nvme_rag.core.pipeline.integration import RAGPipelineIntegration
    return RAGPipelineIntegration

def get_qa_pipeline():
    """Get QAPipeline class (lazy import)."""
    from nvme_rag.core.llm.qa_pipeline import QAPipeline
    return QAPipeline

def get_retrieval_pipeline():
    """Get RetrievalPipeline class (lazy import)."""
    from nvme_rag.core.retrieval.retrieval_pipeline import RetrievalPipeline
    return RetrievalPipeline

__all__ = [
    "get_rag_pipeline",
    "get_qa_pipeline", 
    "get_retrieval_pipeline",
    "__version__",
]