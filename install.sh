#!/bin/bash

# NVMe RAG CLI Installation Script
# This script installs the NVMe RAG CLI tool and its dependencies

set -e  # Exit on any error

echo "🚀 Installing NVMe RAG CLI Tool..."
echo

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8 or higher is required. You have Python $python_version"
    echo "   Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python $python_version found"

# Check if pip is available
echo "🔍 Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 not found. Please install pip3 and try again."
    exit 1
fi
echo "✅ pip3 found"

# Create virtual environment (optional but recommended)
echo
read -p "📦 Do you want to install in a virtual environment? (recommended) [y/N]: " use_venv
if [[ $use_venv =~ ^[Yy]$ ]]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv nvme-rag-env
    
    echo "🔧 Activating virtual environment..."
    source nvme-rag-env/bin/activate
    
    echo "✅ Virtual environment created and activated"
    echo "   To activate it later, run: source nvme-rag-env/bin/activate"
    echo
fi

# Upgrade pip
echo "🔧 Upgrading pip..."
pip3 install --upgrade pip

# Install the package
echo "📦 Installing NVMe RAG CLI..."
if [[ -f "pyproject.toml" ]]; then
    # Install from local directory
    pip3 install -e .
else
    # Install from PyPI (when available)
    pip3 install nvme-rag
fi

echo
echo "✅ Installation complete!"
echo

# Verify installation
echo "🧪 Verifying installation..."
if command -v nvme-rag &> /dev/null; then
    echo "✅ nvme-rag command is available"
    echo
    
    # Show version and basic info
    echo "📋 Installation Summary:"
    nvme-rag --version 2>/dev/null || echo "Version: 1.0.0"
    echo
    
    echo "🚀 Quick Start:"
    echo "1. Run setup: nvme-rag setup setup"
    echo "2. Check status: nvme-rag system status"
    echo "3. Get help: nvme-rag --help"
    echo
    
    echo "📚 Add a document: nvme-rag document add /path/to/document.pdf"
    echo "🤔 Ask questions: nvme-rag ask 'What is NVMe?'"
    echo "💬 Chat mode: nvme-rag query chat"
    echo
    
    if [[ $use_venv =~ ^[Yy]$ ]]; then
        echo "⚠️  Remember to activate the virtual environment before using:"
        echo "   source nvme-rag-env/bin/activate"
        echo
    fi
    
    echo "🎉 Ready to use! Run 'nvme-rag setup setup' to get started."
    
else
    echo "❌ Installation verification failed. nvme-rag command not found."
    echo "   You may need to restart your terminal or check your PATH."
    exit 1
fi