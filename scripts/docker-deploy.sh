#!/bin/bash

# Docker deployment script for NVMe RAG CLI

set -e

echo "🐳 NVMe RAG CLI Docker Deployment"
echo

# Parse command line arguments
COMMAND=${1:-"help"}
MODE=${2:-"production"}

case $COMMAND in
    "build")
        echo "🏗️  Building Docker image..."
        docker build -t nvme-rag:latest --target production .
        echo "✅ Docker image built successfully"
        ;;
        
    "build-dev")
        echo "🏗️  Building development Docker image..."
        docker build -t nvme-rag:dev --target development .
        echo "✅ Development Docker image built successfully"
        ;;
        
    "run")
        echo "🚀 Running NVMe RAG CLI in Docker..."
        docker run -it --rm \
            --name nvme-rag-cli \
            -v nvme-rag-data:/app/data \
            -v nvme-rag-config:/home/nvmerag/.nvme-rag \
            nvme-rag:latest \
            nvme-rag "$@"
        ;;
        
    "chat")
        echo "💬 Starting chat mode in Docker..."
        docker run -it --rm \
            --name nvme-rag-chat \
            -v nvme-rag-data:/app/data \
            -v nvme-rag-config:/home/nvmerag/.nvme-rag \
            nvme-rag:latest \
            nvme-rag query chat
        ;;
        
    "shell")
        echo "🐚 Starting shell in Docker container..."
        docker run -it --rm \
            --name nvme-rag-shell \
            -v nvme-rag-data:/app/data \
            -v nvme-rag-config:/home/nvmerag/.nvme-rag \
            nvme-rag:latest \
            bash
        ;;
        
    "compose-up")
        echo "🚀 Starting services with docker-compose..."
        docker-compose up -d
        echo "✅ Services started"
        echo "📋 Running containers:"
        docker-compose ps
        ;;
        
    "compose-down")
        echo "🛑 Stopping services..."
        docker-compose down
        echo "✅ Services stopped"
        ;;
        
    "compose-logs")
        echo "📋 Showing logs..."
        docker-compose logs -f
        ;;
        
    "dev")
        echo "🔧 Starting development environment..."
        docker-compose --profile dev up -d nvme-rag-dev
        echo "✅ Development environment started"
        echo "🐚 Entering development shell..."
        docker-compose exec nvme-rag-dev bash
        ;;
        
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose down -v
        docker system prune -f
        echo "✅ Cleanup complete"
        ;;
        
    "help"|*)
        echo "📖 Usage: $0 <command> [options]"
        echo
        echo "Available commands:"
        echo "  build          - Build production Docker image"
        echo "  build-dev      - Build development Docker image"
        echo "  run [args]     - Run CLI command in Docker"
        echo "  chat           - Start chat mode in Docker"
        echo "  shell          - Open shell in Docker container"
        echo "  compose-up     - Start all services with docker-compose"
        echo "  compose-down   - Stop all services"
        echo "  compose-logs   - Show service logs"
        echo "  dev            - Start development environment"
        echo "  clean          - Clean up Docker resources"
        echo "  help           - Show this help"
        echo
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 run ask 'What is NVMe?'"
        echo "  $0 chat"
        echo "  $0 dev"
        ;;
esac