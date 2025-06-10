"""
Query and search commands for NVMe RAG CLI.
"""

import json
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import print as rprint

from nvme_rag.config.manager import ConfigManager
from nvme_rag.core.pipeline.integration import RAGPipelineIntegration
from nvme_rag.core.vector_store.embedding_generator import EmbeddingConfig
from nvme_rag.core.llm.ollama_client import OllamaConfig
from nvme_rag.core.retrieval.retrieval_pipeline import RetrievalConfig
from nvme_rag.core.llm.qa_pipeline import AnswerGenerationConfig

console = Console()
logger = logging.getLogger(__name__)


@click.group(name="query")
def query_group():
    """Query and search commands."""
    pass


@query_group.command()
@click.argument("question")
@click.option("--format", type=click.Choice(["text", "json", "markdown"]), default="text", help="Output format")
@click.option("--strategy", type=click.Choice(["semantic", "hybrid", "filtered", "reranked"]), help="Retrieval strategy")
@click.option("--max-results", type=int, help="Maximum results to return")
@click.option("--confidence", type=float, help="Minimum confidence threshold")
@click.option("--sources", is_flag=True, help="Include source citations")
@click.option("--explain", is_flag=True, help="Show reasoning process")
@click.option("--session-id", help="Session ID for conversation continuity")
@click.pass_context
def ask(ctx: click.Context, question: str, format: str, strategy: Optional[str], 
        max_results: Optional[int], confidence: Optional[float], sources: bool, explain: bool, session_id: Optional[str]):
    """Ask a question about NVMe specifications."""
    
    try:
        # Initialize pipeline
        pipeline = _initialize_pipeline()
        
        # Configure retrieval strategy
        retrieval_config = None
        if strategy or max_results:
            retrieval_config = RetrievalConfig(
                strategy=strategy or "hybrid",
                top_k=max_results or 5
            )
        
        # Configure answer generation
        answer_config = AnswerGenerationConfig(
            include_sources=sources,
            include_reasoning=explain,
            min_confidence=confidence or 0.0
        )
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing question...", total=None)
            
            # Ask the question
            qa_result = pipeline.ask_question(
                question=question,
                session_id=session_id,
                answer_config=answer_config
            )
            
            progress.update(task, completed=True)
        
        # Display results based on format
        if format == "json":
            result_dict = pipeline.export_qa_result(qa_result)
            console.print(json.dumps(result_dict, indent=2))
        elif format == "markdown":
            _display_markdown_answer(qa_result)
        else:
            _display_text_answer(qa_result, sources, explain)
            
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        console.print(f"[red]Failed to answer question: {e}[/red]")
        raise click.Abort()


@query_group.command()
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Number of results")
@click.option("--threshold", type=float, help="Similarity threshold")
@click.option("--format", type=click.Choice(["table", "json", "text"]), default="table", help="Output format")
@click.option("--metadata", is_flag=True, help="Include chunk metadata")
def search(query: str, limit: int, threshold: Optional[float], format: str, metadata: bool):
    """Search for relevant document chunks."""
    
    # TODO: Implement vector search
    console.print(f"[bold blue]Searching for:[/bold blue] {query}")
    console.print("[yellow]⚠ This command is not yet implemented[/yellow]")
    
    # Mock results for demonstration
    table = Table()
    table.add_column("Score", style="cyan")
    table.add_column("Document", style="green")
    table.add_column("Section")
    table.add_column("Content")
    
    table.add_row(
        "0.94",
        "NVMe Base Spec",
        "1.1 Introduction",
        "NVMe (Non-Volatile Memory Express) is a scalable host controller interface..."
    )
    table.add_row(
        "0.87", 
        "NVMe Base Spec",
        "2.1 Architecture",
        "The NVMe architecture defines a register interface and command set..."
    )
    
    console.print(table)


@query_group.command()
@click.option("--session", type=click.Path(), help="Load/save session file")
@click.option("--model", help="Use specific model")
@click.option("--temperature", type=float, help="Set generation temperature")
@click.pass_context
def chat(ctx: click.Context, session: Optional[str], model: Optional[str], temperature: Optional[float]):
    """Start interactive chat mode."""
    
    # TODO: Implement interactive chat
    console.print(Panel(
        "Interactive NVMe RAG Chat\nType 'help' for commands, 'exit' to quit",
        title="[bold blue]Chat Mode[/bold blue]",
        expand=False
    ))
    
    console.print("[yellow]⚠ This command is not yet implemented[/yellow]")
    console.print("\nComing soon:")
    console.print("• Interactive question-answering")
    console.print("• Session persistence")
    console.print("• Conversation history")
    console.print("• Command shortcuts")
    
    # Mock interactive loop
    try:
        while True:
            user_input = input("\n🤖 nvme-rag> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                console.print("Available commands:")
                console.print("  help - Show this help")
                console.print("  exit - Exit chat mode")
                console.print("  <question> - Ask about NVMe")
            elif user_input:
                console.print(f"[yellow]You asked: {user_input}[/yellow]")
                console.print("[dim]Chat functionality coming soon...[/dim]")
                
    except KeyboardInterrupt:
        console.print("\nGoodbye!")


# Add the ask command directly to the main query group for convenience
@click.command()
@click.argument("question")
@click.option("--format", type=click.Choice(["text", "json", "markdown"]), default="text", help="Output format")
@click.option("--strategy", type=click.Choice(["semantic", "hybrid", "filtered", "reranked"]), help="Retrieval strategy")
@click.option("--max-results", type=int, help="Maximum results to return")
@click.option("--confidence", type=float, help="Minimum confidence threshold")
@click.option("--sources", is_flag=True, help="Include source citations")
@click.option("--explain", is_flag=True, help="Show reasoning process")
@click.pass_context
def ask_direct(ctx: click.Context, question: str, format: str, strategy: Optional[str], 
               max_results: Optional[int], confidence: Optional[float], sources: bool, explain: bool):
    """Ask a question about NVMe specifications (direct command)."""
    # Forward to the ask command in the query group
    ask.invoke(ctx, question, format, strategy, max_results, confidence, sources, explain)