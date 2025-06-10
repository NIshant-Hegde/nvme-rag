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
@click.option("--filters", help="JSON string of metadata filters")
def search(query: str, limit: int, threshold: Optional[float], format: str, metadata: bool, filters: Optional[str]):
    """Search for relevant document chunks."""
    
    try:
        # Initialize pipeline
        pipeline = _initialize_pipeline()
        
        # Parse filters if provided
        search_filters = None
        if filters:
            try:
                search_filters = json.loads(filters)
            except json.JSONDecodeError:
                console.print(f"[red]Invalid JSON in filters: {filters}[/red]")
                raise click.Abort()
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Searching documents...", total=None)
            
            # Perform search
            search_result = pipeline.search_and_retrieve(
                query=query,
                filters=search_filters
            )
            
            progress.update(task, completed=True)
        
        if not search_result.get("success", False):
            console.print(f"[red]Search failed: {search_result.get('error', 'Unknown error')}[/red]")
            raise click.Abort()
        
        # Filter results by threshold if specified
        chunks = search_result.get("chunks", [])
        if threshold:
            chunks = [chunk for chunk in chunks if chunk.get("score", 0) >= threshold]
        
        # Limit results
        chunks = chunks[:limit]
        
        # Display results based on format
        if format == "json":
            console.print(json.dumps({
                "query": search_result.get("query", {}),
                "chunks": chunks,
                "stats": search_result.get("stats", {})
            }, indent=2))
        elif format == "text":
            _display_text_search_results(chunks, query)
        else:
            _display_table_search_results(chunks, query, metadata)
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        console.print(f"[red]Search failed: {e}[/red]")
        raise click.Abort()


@query_group.command()
@click.option("--session", help="Session ID for conversation continuity")
@click.option("--model", help="Use specific model")
@click.option("--temperature", type=float, help="Set generation temperature")
@click.option("--sources", is_flag=True, help="Always include sources")
@click.pass_context
def chat(ctx: click.Context, session: Optional[str], model: Optional[str], temperature: Optional[float], sources: bool):
    """Start interactive chat mode."""
    
    try:
        # Initialize pipeline
        pipeline = _initialize_pipeline()
        
        # Generate session ID if not provided
        session_id = session or str(uuid.uuid4())[:8]
        
        console.print(Panel(
            f"Interactive NVMe RAG Chat\n" +
            f"Session ID: {session_id}\n" +
            f"Type 'help' for commands, 'exit' to quit",
            title="[bold blue]Chat Mode[/bold blue]",
            expand=False
        ))
        
        # Configure answer generation
        answer_config = AnswerGenerationConfig(
            include_sources=sources,
            temperature=temperature or 0.7
        )
        
        # Interactive loop
        conversation_count = 0
        
        while True:
            try:
                user_input = input(f"\n🤖 nvme-rag[{conversation_count}]> ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    # Show conversation summary
                    summary = pipeline.get_conversation_summary(session_id)
                    if summary.get('total_exchanges', 0) > 0:
                        console.print("\n[bold cyan]Conversation Summary:[/bold cyan]")
                        console.print(f"Total exchanges: {summary.get('total_exchanges', 0)}")
                        console.print(f"Session ID: {session_id}")
                    console.print("[green]Goodbye![/green]")
                    break
                    
                elif user_input.lower() == 'help':
                    _show_chat_help()
                    
                elif user_input.lower() == 'status':
                    status = pipeline.get_pipeline_status()
                    console.print(f"[blue]System Status: {status.get('overall_status', 'Unknown')}[/blue]")
                    
                elif user_input.lower() == 'summary':
                    summary = pipeline.get_conversation_summary(session_id)
                    _display_conversation_summary(summary)
                    
                elif user_input.startswith('/search '):
                    # Quick search command
                    search_query = user_input[8:].strip()
                    if search_query:
                        _quick_search(pipeline, search_query)
                    
                elif user_input:
                    # Ask question
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        
                        task = progress.add_task("Thinking...", total=None)
                        
                        qa_result = pipeline.ask_question(
                            question=user_input,
                            session_id=session_id,
                            answer_config=answer_config
                        )
                        
                        progress.update(task, completed=True)
                    
                    _display_chat_answer(qa_result)
                    conversation_count += 1
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit properly[/yellow]")
                
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        console.print(f"[red]Chat failed: {e}[/red]")
        raise click.Abort()


# Add the ask command directly to the main query group for convenience
@click.command()
@click.argument("question")
@click.option("--format", type=click.Choice(["text", "json", "markdown"]), default="text", help="Output format")
@click.option("--strategy", type=click.Choice(["semantic", "hybrid", "filtered", "reranked"]), help="Retrieval strategy")
@click.option("--max-results", type=int, help="Maximum results to return")
@click.option("--confidence", type=float, help="Minimum confidence threshold")
@click.option("--sources", is_flag=True, help="Include source citations")
@click.option("--explain", is_flag=True, help="Show reasoning process")
@click.option("--session-id", help="Session ID for conversation continuity")
@click.pass_context
def ask_direct(ctx: click.Context, question: str, format: str, strategy: Optional[str], 
               max_results: Optional[int], confidence: Optional[float], sources: bool, explain: bool, session_id: Optional[str]):
    """Ask a question about NVMe specifications (direct command)."""
    # Forward to the ask command in the query group
    ask.invoke(ctx, question, format, strategy, max_results, confidence, sources, explain, session_id)


def _initialize_pipeline() -> RAGPipelineIntegration:
    """Initialize RAG pipeline with current configuration."""
    config_manager = ConfigManager()
    config = config_manager.config
    
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
    
    return RAGPipelineIntegration(
        vector_store_path=config.vector_store.persist_directory,
        embedding_config=embedding_config,
        ollama_config=ollama_config
    )


def _display_text_answer(qa_result, sources: bool, explain: bool):
    """Display QA result in text format."""
    # Question
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {qa_result.question}",
        title="[bold blue]NVMe RAG Query[/bold blue]",
        expand=False
    ))
    
    # Answer
    console.print(f"\n[bold green]Answer:[/bold green]")
    console.print(qa_result.answer)
    
    # Sources
    if sources and qa_result.context_used:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        sources_table = Table()
        sources_table.add_column("Chunk", style="cyan")
        sources_table.add_column("Section", style="green")
        sources_table.add_column("Score", style="yellow")
        sources_table.add_column("Content Preview")
        
        for i, chunk in enumerate(qa_result.context_used[:5], 1):
            content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            sources_table.add_row(
                str(i),
                chunk.section_header or "Unknown",
                f"{getattr(chunk, 'score', 0.0):.3f}",
                content_preview
            )
        
        console.print(sources_table)
    
    # Metadata
    metadata_parts = []
    if hasattr(qa_result, 'confidence_score'):
        metadata_parts.append(f"Confidence: {qa_result.confidence_score:.1%}")
    if hasattr(qa_result, 'processing_time_seconds'):
        metadata_parts.append(f"Processing Time: {qa_result.processing_time_seconds:.2f}s")
    if hasattr(qa_result, 'model_used'):
        metadata_parts.append(f"Model: {qa_result.model_used}")
    
    if metadata_parts:
        console.print(f"\n[dim]{' | '.join(metadata_parts)}[/dim]")


def _display_markdown_answer(qa_result):
    """Display QA result in markdown format."""
    markdown_content = f"""# NVMe RAG Query Result

## Question
{qa_result.question}

## Answer
{qa_result.answer}
"""
    
    if qa_result.context_used:
        markdown_content += "\n## Sources\n"
        for i, chunk in enumerate(qa_result.context_used[:5], 1):
            markdown_content += f"\n{i}. **{chunk.section_header or 'Unknown Section'}**\n"
            markdown_content += f"   Score: {getattr(chunk, 'score', 0.0):.3f}\n"
            content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            markdown_content += f"   {content_preview}\n"
    
    console.print(Markdown(markdown_content))


def _display_table_search_results(chunks, query: str, include_metadata: bool):
    """Display search results in table format."""
    console.print(f"[bold blue]Search Results for:[/bold blue] {query}")
    
    if not chunks:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table()
    table.add_column("#", style="cyan", width=3)
    table.add_column("Score", style="yellow", width=6)
    table.add_column("Section", style="green")
    table.add_column("Content", style="white")
    
    if include_metadata:
        table.add_column("Doc ID", style="blue")
    
    for i, chunk in enumerate(chunks, 1):
        content_preview = chunk.get('content', '')[:150] + "..." if len(chunk.get('content', '')) > 150 else chunk.get('content', '')
        
        row_data = [
            str(i),
            f"{chunk.get('score', 0.0):.3f}",
            chunk.get('metadata', {}).get('section_header', 'Unknown'),
            content_preview
        ]
        
        if include_metadata:
            row_data.append(chunk.get('parent_doc_id', 'Unknown')[:8])
        
        table.add_row(*row_data)
    
    console.print(table)
    console.print(f"\n[dim]Found {len(chunks)} results[/dim]")


def _display_text_search_results(chunks, query: str):
    """Display search results in text format."""
    console.print(f"[bold blue]Search Results for:[/bold blue] {query}\n")
    
    if not chunks:
        console.print("[yellow]No results found[/yellow]")
        return
    
    for i, chunk in enumerate(chunks, 1):
        console.print(f"[bold cyan]Result {i}[/bold cyan] (Score: {chunk.get('score', 0.0):.3f})")
        console.print(f"[green]Section:[/green] {chunk.get('metadata', {}).get('section_header', 'Unknown')}")
        console.print(f"[white]{chunk.get('content', '')}[/white]")
        console.print()


def _display_chat_answer(qa_result):
    """Display QA result in chat format."""
    console.print(f"\n[bold green]🤖 Assistant:[/bold green]")
    console.print(qa_result.answer)
    
    if qa_result.context_used:
        console.print(f"\n[dim]📚 Used {len(qa_result.context_used)} sources[/dim]")


def _show_chat_help():
    """Show chat help commands."""
    console.print("\n[bold cyan]Chat Commands:[/bold cyan]")
    console.print("  help     - Show this help")
    console.print("  status   - Show system status")
    console.print("  summary  - Show conversation summary")
    console.print("  /search <query> - Quick search")
    console.print("  exit     - Exit chat mode")
    console.print("  <question> - Ask about NVMe specifications")


def _display_conversation_summary(summary: Dict[str, Any]):
    """Display conversation summary."""
    console.print(f"\n[bold cyan]Conversation Summary:[/bold cyan]")
    console.print(f"Total exchanges: {summary.get('total_exchanges', 0)}")
    console.print(f"Session started: {summary.get('session_start', 'Unknown')}")
    
    recent_topics = summary.get('recent_topics', [])
    if recent_topics:
        console.print(f"Recent topics: {', '.join(recent_topics[:3])}")


def _quick_search(pipeline: RAGPipelineIntegration, query: str):
    """Perform quick search in chat mode."""
    try:
        search_result = pipeline.search_and_retrieve(query=query)
        
        if search_result.get('success', False):
            chunks = search_result.get('chunks', [])[:3]  # Show top 3 results
            console.print(f"\n[bold blue]🔍 Quick Search Results:[/bold blue]")
            
            for i, chunk in enumerate(chunks, 1):
                content_preview = chunk.get('content', '')[:100] + "..." if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
                console.print(f"[cyan]{i}.[/cyan] {content_preview}")
        else:
            console.print(f"[red]Search failed: {search_result.get('error', 'Unknown error')}[/red]")
    except Exception as e:
        console.print(f"[red]Search error: {e}[/red]")