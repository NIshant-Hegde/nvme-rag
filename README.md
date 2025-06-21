# NVMe RAG System

A professional Retrieval-Augmented Generation (RAG) system specifically designed for the NVMe Base Specification documentation. This system enables intelligent question-answering, context retrieval, and answer generation based on NVMe technical specifications.

![Q2.png](https://github.com/NIshant-Hegde/nvme-rag/blob/main/Q2.png)

## Features

- **Intelligent Document Processing**: Advanced PDF processing with semantic chunking
- **Vector-Based Retrieval**: ChromaDB-powered vector store for efficient similarity search
- **Multi-Model Support**: Sentence transformers and Ollama integration for embeddings and generation
- **Query Translation**: Sophisticated query analysis and translation capabilities
- **Answer Generation**: Context-aware answer generation with follow-up suggestions
- **Session Management**: Conversation history and context management
- **Testing Suite**: Comprehensive test coverage for all components

## Architecture

The system consists of several key components:

- **Data Processing** (`src/data_processing/`): Document processing, PDF parsing, and semantic chunking
- **Vector Store** (`src/vector_store/`): ChromaDB integration and embedding generation
- **Retrieval** (`src/retrieval/`): Context retrieval and ranking pipeline
- **LLM Integration** (`src/llm/`): Ollama client, query translation, and answer generation
- **Pipeline** (`src/pipeline/`): End-to-end RAG pipeline integration

## Requirements

- Python 3.8+
- Ollama server for local LLM inference
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NIshant-Hegde/nvme-rag
   cd nvme-rag
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Pull Gemma 3 (12bit QAT)**
   ```bash
   ollama pull gemma3:12b-it-qat
   ```

2. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

3. **Process NVMe Specification** (if not already done):
   ```bash
   python scripts/process_nvme_spec.py
   ```

4. **Run ChromaDB and Ollama Setup** (vector store initialization):
   ```bash
   python scripts/setup_chromaDB_ollama.py
   ```

## Usage

### Full RAG Pipeline

Run the complete question-answering system:

```bash
python run_full_rag.py
```

This provides an interactive interface for:
- Asking questions about NVMe specifications
- Getting contextual answers with source references
- Receiving follow-up question suggestions

### Retrieval Only

Test retrieval capabilities without answer generation:

```bash
python run_retrieval_and_generation.py
```

## Configuration

The system uses several configuration classes:

- **EmbeddingConfig**: Sentence transformer model configuration
- **OllamaConfig**: Local LLM server configuration
- **RetrievalConfig**: Vector search and ranking parameters
- **AnswerGenerationConfig**: Response generation settings

## Testing - Broken (for now, TODO - fix the tests)

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python tests/run_tests.py        # Unit tests
python tests/run_phase2_tests.py # Integration tests

# Run individual test files
python -m pytest tests/test_full_rag.py
python -m pytest tests/test_data_processing.py
```

## Project Structure

```
nvme-rag/
├── src/                    # Core source code
│   ├── data_processing/    # Document processing utilities
│   ├── llm/                # LLM integration and generation
│   ├── models/             # Data models and schemas
│   ├── pipeline/           # End-to-end pipeline integration
│   ├── retrieval/          # Information retrieval components
│   └── vector_store/       # Vector database integration
├── scripts/                # Utility and deployment scripts
├── tests/                  # Comprehensive test suite
├── data/                   # Vector store and embeddings cache
└── misc/                   # Misc scripts
```

## Author

**Nishant Hegde**
- Email: nishant.hegde13@gmail.com
