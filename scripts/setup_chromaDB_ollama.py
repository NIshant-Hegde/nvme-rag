#!/usr/bin/env python3
"""
Phase 2 Setup Script
Sets up the complete RAG pipeline with ChromaDB and Ollama
"""

import sys
import subprocess
import logging
from pathlib import Path
import shutil
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    logger.info("Installing Phase 2 requirements...")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "pymupdf4llm>=0.0.5"
    ]
    
    for requirement in requirements:
        try:
            logger.info(f"Installing {requirement}...")
            subprocess.run([sys.executable, "-m", "pip", "install", requirement], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {requirement}: {e}")
            return False
    
    logger.info("All requirements installed successfully")
    return True

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "data/vector_store",
        "data/embeddings_cache", 
        "data/demo_cache",
        "data/demo_vector_store",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directories created successfully")
    return True

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    logger.info("Checking Ollama installation...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        logger.info(f"Ollama is running with {len(models)} models")
        
        # Check for recommended models
        model_names = [model["name"] for model in models]
        recommended_models = ["mistral:7b", "llama2:7b"]
        
        for model in recommended_models:
            if model in model_names:
                logger.info(f"Found recommended model: {model}")
            else:
                logger.warning(f"Recommended model not found: {model}")
                logger.info(f"Install with: ollama pull {model}")
        
        return True
        
    except Exception as e:
        logger.error("Ollama is not running or not installed")
        logger.info("Install Ollama:")
        logger.info("  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
        logger.info("  Windows: Download from https://ollama.com")
        logger.info("Then run:")
        logger.info("  ollama serve")
        logger.info("  ollama pull mistral:7b")
        return False

def test_phase2_components():
    """Test that Phase 2 components can be imported"""
    logger.info("Testing Phase 2 component imports...")
    
    try:
        # Test vector store components
        from src.vector_store.base import VectorStoreBase, SearchQuery
        from src.vector_store.embedding_generator import EmbeddingGenerator
        logger.info("Vector store components imported successfully")
        
        # Test LLM components  
        from src.llm.ollama_client import OllamaClient, OllamaConfig
        logger.info("LLM components imported successfully")
        
        # Test retrieval components
        from src.retrieval.retrieval_pipeline import RetrievalPipeline
        logger.info("Retrieval components imported successfully")
        
        # Test integration
        from src.pipeline.integration import RAGPipelineIntegration
        logger.info("Integration components imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def create_config_file():
    """Create default configuration file"""
    logger.info("Creating default configuration...")
    
    config = {
        "embedding": {
            "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "device": "auto",
            "batch_size": 32,
            "cache_embeddings": True,
            "cache_dir": "data/embeddings_cache"
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "mistral:7b",
            "temperature": 0.1,
            "max_tokens": 2048,
            "timeout": 120
        },
        "retrieval": {
            "strategy": "hybrid",
            "top_k": 5,
            "min_score": 0.7,
            "enable_query_enhancement": True,
            "enable_context_filtering": True,
            "max_context_length": 4000
        },
        "vector_store": {
            "collection_name": "nvme_rag_chunks",
            "persist_directory": "data/vector_store"
        }
    }
    
    config_path = Path("config/phase2_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    return True

def run_quick_test():
    """Run a quick test of the pipeline"""
    logger.info("Running quick pipeline test...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent.parent))
        
        from src.vector_store.embedding_generator import EmbeddingGenerator, EmbeddingConfig
        from src.llm.ollama_client import OllamaClient, OllamaConfig
        
        # Test embedding generation
        logger.info("Testing embedding generation...")
        config = EmbeddingConfig(device="cpu", cache_embeddings=False)
        generator = EmbeddingGenerator(config)
        
        embeddings = generator.generate_embeddings(["test text"])
        logger.info(f"Generated embedding with shape: {embeddings.shape}")
        
        # Test Ollama connection
        logger.info("Testing Ollama connection...")
        ollama_config = OllamaConfig(timeout=10)
        client = OllamaClient(ollama_config)
        
        health = client.health_check()
        if health["status"] == "healthy":
            logger.info(f"Ollama health check passed: {health['model']}")
        else:
            logger.warning(f"Ollama health check failed: {health.get('error', 'Unknown')}")
        
        # Cleanup
        generator.cleanup()
        client.cleanup()
        
        logger.info("Quick test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Phase 2 RAG Pipeline Setup")
    logger.info("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return 1
    
    # Step 2: Setup directories
    if not setup_directories():
        return 1
    
    # Step 3: Install requirements
    if not install_requirements():
        return 1
    
    # Step 4: Check Ollama
    ollama_available = check_ollama_installation()
    
    # Step 5: Test imports
    if not test_phase2_components():
        return 1
    
    # Step 6: Create config
    if not create_config_file():
        return 1
    
    # Step 7: Quick test
    if ollama_available and not run_quick_test():
        logger.warning("Quick test failed, but setup may still be functional")
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("Phase 2 Setup Complete!")
    logger.info("=" * 50)
    
    if ollama_available:
        logger.info("All components are ready")
        logger.info("\nNext steps:")
        logger.info("  1. Run tests: python tests/run_phase2_tests.py")
        logger.info("  2. Try demo: python scripts/phase2_demo.py")
        logger.info("  3. Process documents: python scripts/process_nvme_spec.py")
    else:
        logger.info("Setup complete, but Ollama needs to be installed")
        logger.info("\nTo complete setup:")
        logger.info("  1. Install Ollama from https://ollama.com")
        logger.info("  2. Run: ollama serve")
        logger.info("  3. Run: ollama pull mistral:7b")
        logger.info("  4. Run tests: python tests/run_phase2_tests.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)