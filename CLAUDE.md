# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Setup and Environment
```bash
# Setup virtual environment and dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Alternative minimal setup for testing only
pip install -r requirements_phase2.txt

# Full system setup (includes dependency conflict resolution)
./setup_nvme_rag.sh
```

### Running the System
```bash
# Main interactive RAG demo - complete QA system
python run_full_RAG.py

# Simple retrieval-only demo
python run_retrieval_and_generation.py

# Process NVMe specification documents
python scripts/process_nvme_spec.py

# Setup Phase 2 components (vector store + LLM integration)
python scripts/setup_phase2.py
```

### Testing
```bash
# Run all tests
python tests/run_tests.py

# Run Phase 2 specific tests (vector store + LLM components)
python tests/run_phase2_tests.py

# Run specific test suites
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_full_rag.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/ scripts/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Architecture Overview

This is a modular RAG system for NVMe specification question-answering with three main phases:

**Phase 1: Document Processing**
- `src/data_processing/` - PDF extraction and semantic chunking
- `scripts/process_nvme_spec.py` - Main document processing entry point

**Phase 2: Vector Storage & LLM Integration** 
- `src/vector_store/` - ChromaDB-based vector storage with embedding generation
- `src/llm/` - Ollama client and QA pipeline orchestration
- `scripts/setup_phase2.py` - Sets up vector store and tests LLM connectivity

**Phase 3: Complete RAG Pipeline**
- `src/retrieval/` - Multi-strategy retrieval (semantic, hybrid, reranked)
- `src/pipeline/integration.py` - Unified RAGPipelineIntegration class
- `run_full_RAG.py` - Interactive demo with complete QA workflow

## Key Components

- **RAGPipelineIntegration** (`src/pipeline/integration.py`) - Main orchestrator class that unifies all components
- **QAPipeline** (`src/llm/qa_pipeline.py`) - Complete QA workflow with session management
- **RetrievalPipeline** (`src/retrieval/retrieval_pipeline.py`) - Multi-strategy retrieval with reranking
- **ChromaVectorStore** (`src/vector_store/chroma_store.py`) - Persistent vector storage
- **SemanticChunker** (`src/data_processing/semantic_chunker.py`) - Intelligent document chunking

## Data Flow

1. **Document Processing**: PDF → text extraction → semantic chunking → processed chunks stored in `data/processed/`
2. **Vector Indexing**: Processed chunks → embeddings → ChromaDB storage in `data/vector_store/`
3. **Query Processing**: User query → query enhancement → vector search → context filtering → answer generation

## External Dependencies

- **Ollama**: Local LLM server must be running on port 11434
- **ChromaDB**: Vector database for embeddings storage
- **Sentence Transformers**: For embedding generation (cached in `data/embeddings_cache/`)

## Testing Strategy

- Unit tests for individual components
- Integration tests for component interaction  
- End-to-end tests via `test_full_rag.py`
- Phase-specific test runners for targeted testing
- Mock external dependencies (Ollama) in tests when needed