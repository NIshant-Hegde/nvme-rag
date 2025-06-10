#!/bin/bash

# Build distribution packages for NVMe RAG CLI

set -e

echo "🏗️  Building NVMe RAG CLI Distribution Packages..."
echo

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
echo "✅ Cleaned build directories"

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade pip build wheel twine
echo "✅ Build dependencies installed"

# Build the package
echo "🔨 Building source and wheel distributions..."
python -m build
echo "✅ Packages built successfully"

# List built packages
echo
echo "📋 Built packages:"
ls -la dist/
echo

# Verify the packages
echo "🔍 Verifying packages..."
python -m twine check dist/*
echo "✅ Package verification complete"

echo
echo "🎉 Build complete! Packages are in the dist/ directory."
echo
echo "📤 To upload to PyPI:"
echo "   Test PyPI: python -m twine upload --repository testpypi dist/*"
echo "   Production: python -m twine upload dist/*"
echo
echo "💻 To install locally:"
echo "   pip install dist/nvme_rag-*.whl"