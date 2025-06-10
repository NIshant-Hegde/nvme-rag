#!/usr/bin/env python3
"""
Main CLI entry point for NVMe RAG tool.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

from nvme_rag import __version__
from nvme_rag.config import ConfigManager
from nvme_rag.cli.commands import setup, document, query, system, config

# Initialize console for rich output
console = Console()

# Global options
@click.group()
@click.option(
    "--verbose", "-v", 
    is_flag=True, 
    help="Enable verbose output"
)
@click.option(
    "--quiet", "-q", 
    is_flag=True, 
    help="Suppress non-essential output"
)
@click.option(
    "--config-file", 
    type=click.Path(exists=True), 
    help="Use specific configuration file"
)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config_file: Optional[str]):
    """
    NVMe RAG - Professional Retrieval-Augmented Generation for NVMe specifications.
    
    A comprehensive CLI tool for processing NVMe specification documents,
    building vector stores, and answering technical questions with high accuracy.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    
    # Store global options in context
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config_file"] = config_file
    
    # Initialize configuration manager
    try:
        ctx.obj["config_manager"] = ConfigManager(config_file)
    except Exception as e:
        if not quiet:
            console.print(f"[red]Warning: Failed to load configuration: {e}[/red]")
        ctx.obj["config_manager"] = None


# Add command groups
cli.add_command(setup.setup_group)
cli.add_command(document.document_group)
cli.add_command(query.query_group)
cli.add_command(system.system_group)
cli.add_command(config.config_group)

# Add direct commands for convenience
from nvme_rag.cli.commands.query import ask
cli.add_command(ask, name="ask")


@cli.command()
@click.pass_context
def version(ctx: click.Context):
    """Show version information."""
    console.print(f"[bold blue]NVMe RAG[/bold blue] version [green]{__version__}[/green]")
    
    # Show additional info if verbose
    if ctx.obj.get("verbose"):
        console.print(f"Python: {sys.version}")
        console.print(f"Platform: {sys.platform}")


@cli.command()
@click.pass_context
def info(ctx: click.Context):
    """Show system information and status."""
    console.print("[bold blue]NVMe RAG System Information[/bold blue]")
    console.print(f"Version: {__version__}")
    
    # Check configuration
    config_manager = ctx.obj.get("config_manager")
    if config_manager:
        console.print(f"Configuration: [green]✓[/green] Loaded")
        console.print(f"Config file: {config_manager.config_file}")
    else:
        console.print("Configuration: [red]✗[/red] Not loaded")
    
    # Show data directory
    data_dir = Path.home() / ".nvme-rag"
    if data_dir.exists():
        console.print(f"Data directory: [green]✓[/green] {data_dir}")
    else:
        console.print(f"Data directory: [yellow]![/yellow] Not found ({data_dir})")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()