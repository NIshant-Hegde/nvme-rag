# NVMe RAG CLI Deployment Guide

This guide provides multiple deployment options for the NVMe RAG CLI tool, making it easy for your peers to download and use.

## 🚀 Quick Start (Recommended)

### Option 1: One-Command Installation

```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/your-repo/nvme-rag/main/install.sh | bash
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/nvme-rag.git
cd nvme-rag

# Run the installation script
./install.sh
```

### Option 3: Install from PyPI (When Available)

```bash
pip install nvme-rag
```

## 📦 Deployment Options

### 1. Local Installation

#### Prerequisites
- Python 3.8 or higher
- pip3
- 4GB+ RAM recommended
- 2GB+ disk space

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/nvme-rag.git
   cd nvme-rag
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python3 -m venv nvme-rag-env
   source nvme-rag-env/bin/activate  # On Windows: nvme-rag-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   nvme-rag --version
   nvme-rag --help
   ```

5. **Run initial setup:**
   ```bash
   nvme-rag setup setup
   ```

### 2. Docker Deployment

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended

#### Quick Docker Setup

1. **Build and run with Docker:**
   ```bash
   # Build the image
   docker build -t nvme-rag .
   
   # Run the CLI
   docker run -it nvme-rag nvme-rag --help
   
   # Start chat mode
   docker run -it nvme-rag nvme-rag query chat
   ```

2. **Use Docker Compose (Full Setup):**
   ```bash
   # Start all services
   docker-compose up -d
   
   # Enter interactive mode
   docker-compose exec nvme-rag nvme-rag query chat
   
   # View logs
   docker-compose logs -f
   
   # Stop services
   docker-compose down
   ```

#### Docker Helper Script

```bash
# Make the script executable
chmod +x scripts/docker-deploy.sh

# Available commands
./scripts/docker-deploy.sh build      # Build image
./scripts/docker-deploy.sh chat       # Start chat mode
./scripts/docker-deploy.sh dev        # Development environment
./scripts/docker-deploy.sh help       # Show all options
```

### 3. Distribution Package

#### Building Packages

```bash
# Build wheel and source distributions
./scripts/build-package.sh

# This creates files in dist/:
# - nvme_rag-1.0.0-py3-none-any.whl
# - nvme-rag-1.0.0.tar.gz
```

#### Installing from Package

```bash
# Install wheel package
pip install dist/nvme_rag-1.0.0-py3-none-any.whl

# Or install from source distribution
pip install dist/nvme-rag-1.0.0.tar.gz
```

## 🔧 Configuration

### Initial Setup

After installation, run the setup wizard:

```bash
nvme-rag setup setup
```

This will:
- Install required AI models
- Configure Ollama server
- Set up vector database
- Create configuration files
- Verify system requirements

### Configuration Files

The tool stores configuration in:
- **Linux/macOS:** `~/.nvme-rag/config.yml`
- **Windows:** `%USERPROFILE%\.nvme-rag\config.yml`

### Environment Variables

You can override settings with environment variables:

```bash
export NVME_RAG_CONFIG_DIR="/custom/config/path"
export NVME_RAG_DATA_DIR="/custom/data/path"
export OLLAMA_HOST="localhost"
export OLLAMA_PORT="11434"
```

## 🐳 Docker Environment Variables

```bash
# In docker-compose.yml or docker run
environment:
  - NVME_RAG_CONFIG_DIR=/home/nvmerag/.nvme-rag
  - NVME_RAG_DATA_DIR=/app/data
  - OLLAMA_HOST=ollama
  - OLLAMA_PORT=11434
```

## 📋 System Requirements

### Minimum Requirements
- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 5GB free space
- **OS:** Linux, macOS, Windows 10+
- **Python:** 3.8+

### Recommended Requirements
- **CPU:** 8+ cores
- **RAM:** 16GB+
- **Storage:** 20GB+ free space
- **GPU:** CUDA-compatible (optional, for faster processing)

## 🚀 Usage Examples

### Basic Commands

```bash
# Get help
nvme-rag --help

# Check system status
nvme-rag system status

# Add a document
nvme-rag document add /path/to/nvme-spec.pdf

# Ask a question
nvme-rag ask "What is NVMe queue depth?"

# Search documents
nvme-rag query search "PCIe interface"

# Start chat mode
nvme-rag query chat

# List documents
nvme-rag document list

# Configure system
nvme-rag configure configure --show
```

### Advanced Usage

```bash
# Ask with specific format and sources
nvme-rag ask "Explain NVMe commands" --format markdown --sources

# Search with filters
nvme-rag query search "bandwidth" --limit 5 --format json

# Chat with specific model
nvme-rag query chat --model llama3.1:8b --temperature 0.3
```

## 🔍 Troubleshooting

### Common Issues

1. **"Command not found" error:**
   ```bash
   # Ensure the package is installed
   pip list | grep nvme-rag
   
   # Reinstall if needed
   pip install --force-reinstall nvme-rag
   ```

2. **Ollama connection error:**
   ```bash
   # Check Ollama status
   nvme-rag system status
   
   # Start Ollama manually
   ollama serve
   
   # Or use CLI to start
   nvme-rag system start-server
   ```

3. **Memory issues:**
   ```bash
   # Check system resources
   nvme-rag system status
   
   # Use smaller models in config
   nvme-rag configure set ollama model llama3.2:3b
   ```

4. **Permission errors (Linux/macOS):**
   ```bash
   # Fix permissions
   sudo chown -R $USER ~/.nvme-rag
   chmod -R 755 ~/.nvme-rag
   ```

### Docker Troubleshooting

```bash
# Check container logs
docker-compose logs nvme-rag

# Check container status
docker-compose ps

# Restart services
docker-compose restart

# Clean and rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Performance Optimization

### For Better Performance

1. **Use GPU acceleration:**
   ```bash
   # Configure to use GPU
   nvme-rag configure set embedding device cuda
   ```

2. **Optimize batch sizes:**
   ```bash
   # Increase batch size for faster processing
   nvme-rag configure set embedding batch_size 32
   ```

3. **Use faster models:**
   ```bash
   # Use quantized models
   nvme-rag configure set ollama model llama3.1:8b-instruct-q4_0
   ```

## 🔒 Security Considerations

- Store documents in secure locations
- Use virtual environments
- Regularly update dependencies
- Monitor resource usage
- Configure firewall rules for Docker deployments

## 🆕 Updates

### Updating the Tool

```bash
# Update from git
git pull origin main
pip install -e . --upgrade

# Or update from PyPI
pip install --upgrade nvme-rag
```

### Updating Docker Images

```bash
# Rebuild images
docker-compose build --no-cache
docker-compose up -d
```

## 📞 Support

### Getting Help

1. **CLI Help:**
   ```bash
   nvme-rag --help
   nvme-rag <command> --help
   ```

2. **System Status:**
   ```bash
   nvme-rag system status
   ```

3. **Configuration Check:**
   ```bash
   nvme-rag configure validate
   ```

### Reporting Issues

When reporting issues, include:
- Output of `nvme-rag system status`
- Error messages
- Operating system and Python version
- Steps to reproduce

## 🎯 Production Deployment

For production environments:

1. **Use fixed versions:**
   ```bash
   pip install nvme-rag==1.0.0
   ```

2. **Set up monitoring:**
   - Monitor disk usage (vector store grows over time)
   - Monitor memory usage
   - Set up log rotation

3. **Backup data:**
   - Backup `~/.nvme-rag/` directory
   - Backup vector store data
   - Backup processed documents

4. **Use process managers:**
   ```bash
   # With systemd, supervisord, or similar
   # For persistent chat services
   ```

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.