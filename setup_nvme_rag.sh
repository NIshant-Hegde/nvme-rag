#!/bin/bash

# NVMe RAG Package Installation Script
# This script handles package installation with dependency conflict resolution

set -e  # Exit on any error

echo "NVMe RAG Package Installation Script"
echo "===================================="

# Check if we're in the project directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "It's recommended to run this in a virtual environment."
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please activate your virtual environment first:"
        echo "  source venv/bin/activate"
        exit 1
    fi
fi

echo "Checking system dependencies..."

# Check for system dependencies on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Warning: Homebrew not found. Some packages may fail to install."
        echo "Install Homebrew from: https://brew.sh"
    fi
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        echo "cmake not found. Installing via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install cmake
        else
            echo "Please install cmake manually: brew install cmake"
        fi
    fi
    
    # Check for pkg-config
    if ! command -v pkg-config &> /dev/null; then
        echo "pkg-config not found. Installing via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install pkg-config
        else
            echo "Please install pkg-config manually: brew install pkg-config"
        fi
    fi
fi

# Create updated requirements.txt with safer dependencies
echo "Creating safe requirements.txt..."
cat > requirements-safe.txt << 'EOF'
# Core ML/AI Libraries (most stable)
torch
torchvision
transformers
sentence-transformers
accelerate

# Document Processing (basic, no compilation needed)
pymupdf
pytesseract
Pillow
python-docx
openpyxl

# Vector Databases (CPU versions, more stable)
faiss-cpu
chromadb

# Text Processing and NLP
spacy
nltk
beautifulsoup4
markdown

# LangChain Ecosystem (core only)
langchain
langchain-community
langchain-chroma

# API and Web Framework
fastapi
uvicorn[standard]
pydantic
httpx

# Data Processing
pandas
numpy
scikit-learn

# Configuration and Environment
python-dotenv
pyyaml

# Logging and Monitoring
loguru
rich
tqdm

# Development and Testing
pytest
pytest-asyncio
black
flake8

# Utilities
click
requests
aiofiles
psutil
EOF

# Create problematic requirements separately
cat > requirements-advanced.txt << 'EOF'
# Advanced packages that may require system dependencies
# Install these only after system dependencies are ready

# Advanced Document Processing (requires cmake, pkg-config)
marker-pdf
docling
nougat-ocr

# Additional Vector Databases
qdrant-client

# Additional utilities that may cause issues
python-magic
mypy
pathlib2
configparser
elasticsearch
EOF

echo "Installing safe packages first..."
echo "================================"

# Install safe packages with error handling
if pip install -r requirements-safe.txt; then
    echo "✅ Core packages installed successfully!"
else
    echo "❌ Some core packages failed. Trying individual installation..."
    
    # Try installing packages individually
    packages=(
        "torch"
        "transformers"
        "sentence-transformers"
        "pymupdf"
        "Pillow"
        "faiss-cpu"
        "chromadb"
        "spacy"
        "nltk"
        "langchain"
        "fastapi"
        "uvicorn[standard]"
        "pandas"
        "numpy"
        "python-dotenv"
        "loguru"
        "rich"
        "pytest"
        "requests"
    )
    
    failed_packages=()
    
    for package in "${packages[@]}"; do
        echo "Installing $package..."
        if pip install "$package"; then
            echo "✅ $package installed"
        else
            echo "❌ $package failed"
            failed_packages+=("$package")
        fi
    done
    
    if [ ${#failed_packages[@]} -gt 0 ]; then
        echo ""
        echo "Failed packages:"
        printf '%s\n' "${failed_packages[@]}"
    fi
fi

echo ""
echo "Attempting to install advanced packages..."
echo "========================================="

# Try advanced packages with individual error handling
advanced_packages=(
    "marker-pdf"
    "docling"
    "nougat-ocr"
    "qdrant-client"
    "python-magic"
    "mypy"
    "elasticsearch"
)

installed_advanced=()
failed_advanced=()

for package in "${advanced_packages[@]}"; do
    echo "Trying to install $package..."
    if pip install "$package" 2>/dev/null; then
        echo "✅ $package installed"
        installed_advanced+=("$package")
    else
        echo "⚠️  $package failed (skipping)"
        failed_advanced+=("$package")
    fi
done

echo ""
echo "Installation Summary"
echo "==================="
echo "✅ Core packages: Installed"

if [ ${#installed_advanced[@]} -gt 0 ]; then
    echo "✅ Advanced packages installed:"
    printf '  - %s\n' "${installed_advanced[@]}"
fi

if [ ${#failed_advanced[@]} -gt 0 ]; then
    echo "⚠️  Advanced packages that failed:"
    printf '  - %s\n' "${failed_advanced[@]}"
    echo ""
    echo "To install failed packages later:"
    for package in "${failed_advanced[@]}"; do
        echo "  pip install $package"
    done
fi

echo ""
echo "Post-installation setup:"
echo "========================"

# Download spacy model if spacy was installed successfully
if python -c "import spacy" 2>/dev/null; then
    echo "Downloading spacy English model..."
    python -m spacy download en_core_web_sm || echo "⚠️  Spacy model download failed"
fi

# Setup NLTK data
if python -c "import nltk" 2>/dev/null; then
    echo "Setting up NLTK data..."
    python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded')
except:
    print('NLTK data download failed')
" || echo "⚠️  NLTK setup failed"
fi

echo ""
echo "Next steps:"
echo "==========="
echo "1. Copy environment file: cp .env.example .env"
echo "2. Place NVMe PDF files in data/raw/"
echo "3. Test the installation:"
echo "   python -c 'import torch, transformers, langchain; print(\"✅ Installation successful\")'"
echo ""

if [ ${#failed_advanced[@]} -gt 0 ]; then
    echo "For failed packages, you can:"
    echo "- Install system dependencies and retry"
    echo "- Use alternative packages"
    echo "- Skip advanced features for now"
    echo ""
fi

echo "Installation script completed!"
