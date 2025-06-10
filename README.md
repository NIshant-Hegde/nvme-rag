# NVMe RAG System

A professional Retrieval-Augmented Generation (RAG) system specifically designed for the NVMe Base Specification documentation.

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

4. Place NVMe specification PDFs in `data/raw/`

## Project Structure

- `config/` - Configuration settings
- `src/` - Main source code
  - `data_processing/` - PDF processing and conversion
  - `chunking/` - Document chunking strategies
  - `retrieval/` - Vector search and retrieval
  - `generation/` - Response generation
  - `evaluation/` - Performance metrics
  - `utils/` - Utility functions
- `data/` - Data storage
- `models/` - ML models
- `api/` - FastAPI application
- `tests/` - Test suite
- `scripts/` - Utility scripts
- `notebooks/` - Jupyter notebooks
- `docker/` - Docker configuration

## Development

Install in development mode:
```bash
pip install -e .
```

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ api/ tests/
```

## License

MIT License
