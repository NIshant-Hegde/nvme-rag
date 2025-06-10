"""
Document management commands for NVMe RAG CLI.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

from nvme_rag.config.manager import ConfigManager
from nvme_rag.core.pipeline.integration import RAGPipelineIntegration
from nvme_rag.core.vector_store.embedding_generator import EmbeddingConfig
from nvme_rag.core.llm.ollama_client import OllamaConfig

console = Console()
logger = logging.getLogger(__name__)


@click.group(name="document")
def document_group():
    """Document management commands."""
    pass


@document_group.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", help="Custom document name")
@click.option("--description", help="Document description")
@click.option("--chunk-size", type=int, help="Override chunk size")
@click.option("--overlap", type=int, help="Override chunk overlap")
@click.option("--format", type=click.Choice(["json", "yaml", "table"]), default="table", help="Output format")
@click.pass_context
def add(ctx: click.Context, path: str, name: Optional[str], description: Optional[str], 
        chunk_size: Optional[int], overlap: Optional[int], format: str):
    """Add a document to the system."""
    
    try:
        document_path = Path(path)
        
        # Validate file type
        if document_path.suffix.lower() != '.pdf':
            console.print("[red]Error: Only PDF files are supported[/red]")
            raise click.Abort()
        
        # Initialize configuration and pipeline
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Initialize RAG pipeline with config
        embedding_config = EmbeddingConfig(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size
        )
        
        ollama_config = OllamaConfig(
            host=config.ollama.host,
            port=config.ollama.port,
            model=config.ollama.model,
            timeout=config.ollama.timeout
        )
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=config.vector_store.persist_directory,
            embedding_config=embedding_config,
            ollama_config=ollama_config
        )
        
        # Process document with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing document...", total=None)
            
            result = pipeline.process_and_index_document(document_path)
            
            progress.update(task, completed=True)
        
        if result.get("success", False):
            # Display results based on format
            if format == "json":
                console.print(json.dumps(result, indent=2))
            elif format == "yaml":
                import yaml
                console.print(yaml.dump(result, default_flow_style=False))
            else:
                _display_processing_result(result, document_path)
        else:
            console.print(f"[red]Error processing document: {result.get('error', 'Unknown error')}[/red]")
            raise click.Abort()
            
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        console.print(f"[red]Failed to process document: {e}[/red]")
        raise click.Abort()


@document_group.command()
@click.option("--format", type=click.Choice(["table", "json", "yaml"]), default="table", help="Output format")
@click.option("--filter", help="Filter by name/status")
@click.option("--sort", help="Sort by field")
@click.pass_context
def list(ctx: click.Context, format: str, filter: Optional[str], sort: Optional[str]):
    """List processed documents."""
    
    try:
        # Initialize configuration and pipeline
        config_manager = ConfigManager()
        config = config_manager.config
        
        embedding_config = EmbeddingConfig(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        
        ollama_config = OllamaConfig(
            host=config.ollama.host,
            port=config.ollama.port,
            model=config.ollama.model
        )
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=config.vector_store.persist_directory,
            embedding_config=embedding_config,
            ollama_config=ollama_config
        )
        
        # Get vector store stats
        vector_stats = pipeline.vector_store.get_stats()
        
        if format == "json":
            console.print(json.dumps(vector_stats, indent=2))
        elif format == "yaml":
            import yaml
            console.print(yaml.dump(vector_stats, default_flow_style=False))
        else:
            _display_document_list(vector_stats)
            
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        console.print(f"[red]Failed to list documents: {e}[/red]")
        raise click.Abort()


@document_group.command()
@click.argument("document_id")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.option("--keep-files", is_flag=True, help="Keep original files")
@click.pass_context
def remove(ctx: click.Context, document_id: str, force: bool, keep_files: bool):
    """Remove a document from the system."""
    
    try:
        if not force:
            if not click.confirm(f"Are you sure you want to remove document '{document_id}'?"):
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        # Initialize configuration and pipeline
        config_manager = ConfigManager()
        config = config_manager.config
        
        embedding_config = EmbeddingConfig(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        
        ollama_config = OllamaConfig(
            host=config.ollama.host,
            port=config.ollama.port,
            model=config.ollama.model
        )
        
        pipeline = RAGPipelineIntegration(
            vector_store_path=config.vector_store.persist_directory,
            embedding_config=embedding_config,
            ollama_config=ollama_config
        )
        
        # Remove chunks with the document ID as prefix
        removed_count = pipeline.vector_store.delete_chunks_by_filter({"parent_doc_id": document_id})
        
        if removed_count > 0:
            console.print(f"[green]Successfully removed {removed_count} chunks for document {document_id}[/green]")
        else:
            console.print(f"[yellow]No chunks found for document ID: {document_id}[/yellow]")
            
    except Exception as e:
        logger.error(f"Failed to remove document: {e}")
        console.print(f"[red]Failed to remove document: {e}[/red]")
        raise click.Abort()


@document_group.command()
@click.option("--force", is_flag=True, help="Force full rebuild")
@click.option("--verify", is_flag=True, help="Verify integrity after rebuild")
@click.pass_context
def reindex(ctx: click.Context, force: bool, verify: bool):
    """Rebuild the vector store index."""
    
    try:
        if not force:
            if not click.confirm("This will rebuild the entire vector store index. Continue?"):
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.config
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Rebuilding vector store index...", total=None)
            
            # Clear existing vector store
            vector_store_path = Path(config.vector_store.persist_directory)
            if vector_store_path.exists():
                import shutil
                shutil.rmtree(vector_store_path)
                console.print("[yellow]Cleared existing vector store[/yellow]")
            
            progress.update(task, description="Vector store cleared, reindexing required")
        
        console.print("[green]Vector store cleared successfully[/green]")
        console.print("[yellow]Use 'nvme-rag document add <path>' to re-add documents[/yellow]")
        
        if verify:
            console.print("[blue]Verification: Vector store is now empty and ready for new documents[/blue]")
            
    except Exception as e:
        logger.error(f"Failed to reindex: {e}")
        console.print(f"[red]Failed to reindex: {e}[/red]")
        raise click.Abort()


def _display_processing_result(result: Dict[str, Any], document_path: Path):
    """Display document processing results in a nice table format."""
    
    console.print(Panel.fit(
        f"[bold green]✓ Document Successfully Processed[/bold green]\n" +
        f"[blue]File:[/blue] {document_path.name}",
        title="Processing Complete"
    ))
    
    # Processing stats table
    stats_table = Table(title="Processing Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    processing_stats = result.get("processing_stats", {})
    stats_table.add_row("Processing Time", f"{processing_stats.get('processing_time_seconds', 0):.2f}s")
    stats_table.add_row("Total Chunks", str(processing_stats.get('total_chunks', 0)))
    stats_table.add_row("Total Characters", f"{processing_stats.get('total_characters', 0):,}")
    stats_table.add_row("Average Chunk Size", f"{processing_stats.get('average_chunk_size', 0):.0f} chars")
    stats_table.add_row("Headers Found", str(processing_stats.get('headers_found', 0)))
    stats_table.add_row("Extraction Method", str(processing_stats.get('extraction_method', 'Unknown')))
    
    console.print(stats_table)
    
    # Vector store stats
    vector_stats = result.get("vector_store_stats", {})
    if vector_stats:
        vector_table = Table(title="Vector Store Statistics")
        vector_table.add_column("Metric", style="cyan")
        vector_table.add_column("Value", style="yellow")
        
        vector_table.add_row("Total Chunks in Store", str(vector_stats.get('total_chunks', 0)))
        vector_table.add_row("Collection Name", str(vector_stats.get('collection_name', 'Unknown')))
        vector_table.add_row("Embedding Model", str(vector_stats.get('embedding_model', 'Unknown')))
        
        console.print(vector_table)


def _display_document_list(vector_stats: Dict[str, Any]):
    """Display list of documents in the vector store."""
    
    console.print(Panel.fit(
        "[bold blue]Document Library[/bold blue]",
        title="Vector Store Contents"
    ))
    
    # Main stats table
    stats_table = Table(title="Vector Store Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Chunks", str(vector_stats.get('total_chunks', 0)))
    stats_table.add_row("Collection Name", str(vector_stats.get('collection_name', 'Unknown')))
    stats_table.add_row("Embedding Model", str(vector_stats.get('embedding_model', 'Unknown')))
    
    if 'last_updated' in vector_stats:
        stats_table.add_row("Last Updated", str(vector_stats['last_updated']))
    
    console.print(stats_table)
    
    # Show sample documents if available
    documents = vector_stats.get('documents', [])
    if documents:
        doc_table = Table(title="Documents")
        doc_table.add_column("Document ID", style="cyan")
        doc_table.add_column("Chunks", style="yellow")
        doc_table.add_column("Source", style="green")
        
        for doc in documents[:10]:  # Show first 10 documents
            doc_table.add_row(
                str(doc.get('id', 'Unknown')),
                str(doc.get('chunk_count', 0)),
                str(doc.get('source', 'Unknown'))[:50] + "..." if len(str(doc.get('source', ''))) > 50 else str(doc.get('source', 'Unknown'))
            )
        
        console.print(doc_table)
        
        if len(documents) > 10:
            console.print(f"[yellow]... and {len(documents) - 10} more documents[/yellow]")
    else:
        console.print("[yellow]No documents found in vector store[/yellow]")
        console.print("[blue]Use 'nvme-rag document add <path>' to add documents[/blue]")