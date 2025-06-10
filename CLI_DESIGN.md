# NVMe RAG CLI Tool Design

## Command Structure

### Main Command: `nvme-rag`

```
nvme-rag [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Global Options
```
--verbose, -v          Enable verbose output
--quiet, -q           Suppress non-essential output  
--config FILE         Use specific configuration file
--help, -h            Show help message
--version             Show version information
```

### Command Groups

#### 1. Setup and Configuration
```bash
nvme-rag setup                    # Interactive setup wizard
  --force                         # Force reinstallation
  --models MODEL1,MODEL2         # Specify models to install
  --skip-ollama                  # Skip Ollama installation
  --config-only                 # Only create configuration

nvme-rag configure               # Configuration management
  --show                         # Show current configuration
  --edit                         # Edit configuration interactively
  --reset                        # Reset to defaults
  --export FILE                  # Export configuration
  --import FILE                  # Import configuration

nvme-rag install-models         # Model management
  --list                         # List available models
  --model MODEL                  # Install specific model
  --all                          # Install all recommended models
```

#### 2. Document Management
```bash
nvme-rag add-document PATH       # Add document to system
  --name NAME                    # Custom document name
  --description DESC             # Document description
  --chunk-size SIZE              # Override chunk size
  --overlap SIZE                 # Override chunk overlap

nvme-rag list-documents         # List processed documents
  --format FORMAT                # Output format (table, json, yaml)
  --filter FILTER                # Filter by name/status
  --sort FIELD                   # Sort by field

nvme-rag remove-document ID     # Remove document
  --force                        # Skip confirmation
  --keep-files                   # Keep original files

nvme-rag reindex                # Rebuild vector store
  --force                        # Force full rebuild
  --verify                       # Verify integrity after rebuild
```

#### 3. Query Operations
```bash
nvme-rag ask QUESTION           # Ask a question
  --format FORMAT                # Output format (text, json, markdown)
  --strategy STRATEGY            # Retrieval strategy
  --max-results N                # Maximum results to return
  --confidence THRESHOLD         # Minimum confidence threshold
  --sources                      # Include source citations
  --explain                      # Show reasoning process

nvme-rag search QUERY           # Search for chunks
  --limit N                      # Number of results
  --threshold SCORE              # Similarity threshold
  --format FORMAT                # Output format
  --metadata                     # Include chunk metadata

nvme-rag chat                   # Interactive chat mode
  --session FILE                 # Load/save session
  --model MODEL                  # Use specific model
  --temperature TEMP             # Set generation temperature
```

#### 4. System Management
```bash
nvme-rag status                 # System status check
  --detailed                     # Show detailed status
  --json                         # Output as JSON
  --health-check                 # Run comprehensive health check

nvme-rag start-server           # Start Ollama server
  --wait                         # Wait for server to be ready
  --timeout SECONDS              # Server startup timeout

nvme-rag stop-server            # Stop Ollama server
  --force                        # Force stop

nvme-rag reset                  # Reset system
  --keep-docs                    # Keep document data
  --keep-config                  # Keep configuration
  --confirm                      # Skip confirmation prompt
```

#### 5. Import/Export
```bash
nvme-rag export FORMAT          # Export knowledge base
  --output FILE                  # Output file path
  --include-embeddings           # Include vector embeddings
  --compress                     # Compress output

nvme-rag import FILE            # Import knowledge base
  --merge                        # Merge with existing data
  --overwrite                    # Overwrite existing data
```

## User Experience Flow

### First Time Setup
```bash
# User installs the tool
pip install nvme-rag

# User runs setup
nvme-rag setup
# -> Interactive wizard guides through:
#    - System dependency check
#    - Ollama installation
#    - Model selection and download
#    - Configuration creation
#    - Test document processing

# User adds NVMe specification
nvme-rag add-document /path/to/nvme-spec.pdf
# -> Progress bar shows processing
# -> Success message with statistics

# User asks first question
nvme-rag ask "What is NVMe?"
# -> Displays answer with sources and confidence
```

### Daily Usage
```bash
# Quick question
nvme-rag ask "How does NVMe queue management work?"

# Interactive session
nvme-rag chat
# -> Enters chat mode with prompt
# -> User can ask multiple questions
# -> Type 'exit' to quit

# System maintenance
nvme-rag status
# -> Shows system health
```

## Output Formats

### Question Answering Output
```
Question: What is NVMe?

Answer: [Generated answer with high confidence]

Confidence: 95%

Sources:
┌─────────────────────────────────────────────────────────────────┐
│ Document: NVMe Base Specification v2.0                         │
│ Section: 1.1 Introduction                                       │
│ Page: 15                                                        │
│ Relevance: 0.94                                                │
└─────────────────────────────────────────────────────────────────┘

Processing Time: 1.2s
Retrieved Chunks: 5
Model Used: gemma3:12b-it-qat
```

### Status Output
```
NVMe RAG System Status

System Health: ✅ Healthy
Ollama Server: ✅ Running (localhost:11434)
Vector Store: ✅ Ready (1,234 chunks indexed)
Documents: 3 processed
Last Updated: 2024-01-15 10:30:45

Models:
• gemma3:12b-it-qat ✅ Available
• sentence-transformers/multi-qa-MiniLM-L6-cos-v1 ✅ Loaded

Storage:
• Vector Store: 45.2 MB
• Embeddings Cache: 12.8 MB
• Document Cache: 8.9 MB
```

### Error Handling
```
❌ Error: Ollama server not responding

Suggestions:
1. Start the server: nvme-rag start-server
2. Check if Ollama is installed: ollama --version
3. View detailed logs: nvme-rag status --detailed

Need help? Run: nvme-rag --help
```

## Configuration File Structure

```yaml
# ~/.nvme-rag/config.yml
system:
  data_dir: ~/.nvme-rag/data
  log_level: INFO
  max_parallel_jobs: 4

ollama:
  base_url: http://localhost:11434
  model: gemma3:12b-it-qat
  temperature: 0.1
  max_tokens: 2048
  timeout: 30

embedding:
  model_name: sentence-transformers/multi-qa-MiniLM-L6-cos-v1
  device: cpu
  cache_embeddings: true
  batch_size: 32

retrieval:
  strategy: hybrid
  max_results: 10
  confidence_threshold: 0.7
  rerank_results: true

processing:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100
  max_chunk_size: 2000

ui:
  color_output: true
  progress_bars: true
  verbose_errors: true
```

## Plugin Architecture (Future)

```python
# Example plugin structure
class CustomRetrieverPlugin(RetrieverPlugin):
    name = "custom-retriever"
    version = "1.0.0"
    
    def retrieve(self, query: str, config: dict) -> List[Chunk]:
        # Custom retrieval logic
        pass

# Registration
nvme-rag plugin install custom-retriever
nvme-rag plugin list
nvme-rag plugin enable custom-retriever
```

This design provides a comprehensive, user-friendly CLI interface that follows Unix command-line conventions while providing powerful functionality for NVMe specification querying.