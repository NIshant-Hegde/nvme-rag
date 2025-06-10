"""
Configuration management commands for NVMe RAG CLI.
"""

import json
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from nvme_rag.config import ConfigManager

console = Console()


@click.group(name="configure")
def config_group():
    """Configuration management commands."""
    pass


@config_group.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--edit", is_flag=True, help="Edit configuration interactively")
@click.option("--reset", is_flag=True, help="Reset to defaults")
@click.option("--export", type=click.Path(), help="Export configuration to file")
@click.option("--import", "import_file", type=click.Path(exists=True), help="Import configuration from file")
@click.pass_context
def configure(ctx: click.Context, show: bool, edit: bool, reset: bool, export: Optional[str], import_file: Optional[str]):
    """Manage system configuration."""
    
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    
    if show:
        _show_configuration(config_manager)
    elif edit:
        _edit_configuration(config_manager)
    elif reset:
        _reset_configuration(config_manager)
    elif export:
        _export_configuration(config_manager, Path(export))
    elif import_file:
        _import_configuration(config_manager, Path(import_file))
    else:
        # Default: show current configuration
        _show_configuration(config_manager)


@config_group.command()
@click.argument("section")
@click.argument("key")
@click.argument("value")
@click.pass_context
def set(ctx: click.Context, section: str, key: str, value: str):
    """Set a configuration value.
    
    Examples:
        nvme-rag configure set ollama model llama3.2:3b
        nvme-rag configure set system log_level DEBUG
    """
    
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    
    try:
        # Convert value to appropriate type
        converted_value = _convert_value(value)
        
        # Update configuration
        config_manager.update_section(section, {key: converted_value})
        
        console.print(f"[green]✓[/green] Set {section}.{key} = {converted_value}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to set configuration: {e}")


@config_group.command()
@click.argument("section")
@click.argument("key")
@click.pass_context
def get(ctx: click.Context, section: str, key: str):
    """Get a configuration value.
    
    Examples:
        nvme-rag configure get ollama model
        nvme-rag configure get system data_dir
    """
    
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    
    try:
        section_obj = config_manager.get_section(section)
        value = getattr(section_obj, key)
        console.print(f"{section}.{key} = {value}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get configuration: {e}")


@config_group.command()
@click.pass_context
def validate(ctx: click.Context):
    """Validate current configuration."""
    
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    
    validation = config_manager.validate()
    
    if validation["valid"]:
        console.print("[green]✓[/green] Configuration is valid")
    else:
        console.print("[red]✗[/red] Configuration has issues:")
        for issue in validation["issues"]:
            console.print(f"  • {issue}")
    
    if validation["warnings"]:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in validation["warnings"]:
            console.print(f"  • {warning}")


def _show_configuration(config_manager: ConfigManager) -> None:
    """Display current configuration in a nice format."""
    
    console.print(Panel(
        "Current NVMe RAG Configuration",
        title="[bold blue]Configuration[/bold blue]",
        expand=False
    ))
    
    config = config_manager.config
    
    # System configuration
    console.print("\n[bold cyan]System Configuration[/bold cyan]")
    table = Table()
    table.add_column("Setting", style="green")
    table.add_column("Value")
    
    table.add_row("Data Directory", str(config.system.data_dir))
    table.add_row("Log Level", config.system.log_level)
    table.add_row("Max Parallel Jobs", str(config.system.max_parallel_jobs))
    console.print(table)
    
    # Ollama configuration
    console.print("\n[bold cyan]Ollama Configuration[/bold cyan]")
    table = Table()
    table.add_column("Setting", style="green")
    table.add_column("Value")
    
    table.add_row("Base URL", config.ollama.base_url)
    table.add_row("Model", config.ollama.model)
    table.add_row("Temperature", str(config.ollama.temperature))
    table.add_row("Max Tokens", str(config.ollama.max_tokens))
    table.add_row("Timeout", f"{config.ollama.timeout}s")
    console.print(table)
    
    # Embedding configuration
    console.print("\n[bold cyan]Embedding Configuration[/bold cyan]")
    table = Table()
    table.add_column("Setting", style="green")
    table.add_column("Value")
    
    table.add_row("Model Name", config.embedding.model_name)
    table.add_row("Device", config.embedding.device)
    table.add_row("Cache Embeddings", str(config.embedding.cache_embeddings))
    table.add_row("Batch Size", str(config.embedding.batch_size))
    console.print(table)
    
    # Retrieval configuration
    console.print("\n[bold cyan]Retrieval Configuration[/bold cyan]")
    table = Table()
    table.add_column("Setting", style="green")
    table.add_column("Value")
    
    table.add_row("Strategy", config.retrieval.strategy)
    table.add_row("Max Results", str(config.retrieval.max_results))
    table.add_row("Confidence Threshold", str(config.retrieval.confidence_threshold))
    table.add_row("Rerank Results", str(config.retrieval.rerank_results))
    console.print(table)
    
    # Processing configuration
    console.print("\n[bold cyan]Processing Configuration[/bold cyan]")
    table = Table()
    table.add_column("Setting", style="green")
    table.add_column("Value")
    
    table.add_row("Chunk Size", str(config.processing.chunk_size))
    table.add_row("Chunk Overlap", str(config.processing.chunk_overlap))
    table.add_row("Min Chunk Size", str(config.processing.min_chunk_size))
    table.add_row("Max Chunk Size", str(config.processing.max_chunk_size))
    console.print(table)
    
    # Configuration file location
    console.print(f"\n[dim]Configuration file: {config_manager.config_file}[/dim]")


def _edit_configuration(config_manager: ConfigManager) -> None:
    """Edit configuration interactively."""
    
    console.print("[bold]Interactive Configuration Editor[/bold]\n")
    
    config = config_manager.config
    
    # System settings
    if Confirm.ask("Edit system settings?"):
        console.print("\n[cyan]System Settings[/cyan]")
        
        data_dir = Prompt.ask(
            "Data directory",
            default=str(config.system.data_dir)
        )
        config.system.data_dir = Path(data_dir)
        
        log_level = Prompt.ask(
            "Log level",
            default=config.system.log_level,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"]
        )
        config.system.log_level = log_level
        
        max_jobs = Prompt.ask(
            "Max parallel jobs",
            default=str(config.system.max_parallel_jobs)
        )
        config.system.max_parallel_jobs = int(max_jobs)
    
    # Ollama settings
    if Confirm.ask("Edit Ollama settings?"):
        console.print("\n[cyan]Ollama Settings[/cyan]")
        
        base_url = Prompt.ask(
            "Base URL",
            default=config.ollama.base_url
        )
        config.ollama.base_url = base_url
        
        model = Prompt.ask(
            "Model",
            default=config.ollama.model
        )
        config.ollama.model = model
        
        temperature = Prompt.ask(
            "Temperature",
            default=str(config.ollama.temperature)
        )
        config.ollama.temperature = float(temperature)
        
        max_tokens = Prompt.ask(
            "Max tokens",
            default=str(config.ollama.max_tokens)
        )
        config.ollama.max_tokens = int(max_tokens)
    
    # Embedding settings
    if Confirm.ask("Edit embedding settings?"):
        console.print("\n[cyan]Embedding Settings[/cyan]")
        
        model_name = Prompt.ask(
            "Model name",
            default=config.embedding.model_name
        )
        config.embedding.model_name = model_name
        
        device = Prompt.ask(
            "Device",
            default=config.embedding.device,
            choices=["cpu", "cuda", "mps"]
        )
        config.embedding.device = device
        
        cache_embeddings = Confirm.ask(
            "Cache embeddings?",
            default=config.embedding.cache_embeddings
        )
        config.embedding.cache_embeddings = cache_embeddings
    
    # Retrieval settings
    if Confirm.ask("Edit retrieval settings?"):
        console.print("\n[cyan]Retrieval Settings[/cyan]")
        
        strategy = Prompt.ask(
            "Strategy",
            default=config.retrieval.strategy,
            choices=["semantic", "hybrid", "filtered", "reranked"]
        )
        config.retrieval.strategy = strategy
        
        max_results = Prompt.ask(
            "Max results",
            default=str(config.retrieval.max_results)
        )
        config.retrieval.max_results = int(max_results)
        
        confidence_threshold = Prompt.ask(
            "Confidence threshold",
            default=str(config.retrieval.confidence_threshold)
        )
        config.retrieval.confidence_threshold = float(confidence_threshold)
    
    # Processing settings
    if Confirm.ask("Edit processing settings?"):
        console.print("\n[cyan]Processing Settings[/cyan]")
        
        chunk_size = Prompt.ask(
            "Chunk size",
            default=str(config.processing.chunk_size)
        )
        config.processing.chunk_size = int(chunk_size)
        
        chunk_overlap = Prompt.ask(
            "Chunk overlap",
            default=str(config.processing.chunk_overlap)
        )
        config.processing.chunk_overlap = int(chunk_overlap)
    
    # Save configuration
    try:
        config_manager.save()
        console.print("\n[green]✓[/green] Configuration saved successfully")
    except Exception as e:
        console.print(f"\n[red]✗[/red] Failed to save configuration: {e}")


def _reset_configuration(config_manager: ConfigManager) -> None:
    """Reset configuration to defaults."""
    
    if Confirm.ask("[red]Reset configuration to defaults? This cannot be undone.[/red]"):
        try:
            config_manager.reset_to_defaults()
            console.print("[green]✓[/green] Configuration reset to defaults")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to reset configuration: {e}")
    else:
        console.print("Reset cancelled")


def _export_configuration(config_manager: ConfigManager, output_file: Path) -> None:
    """Export configuration to file."""
    
    try:
        config_manager.export_config(output_file)
        console.print(f"[green]✓[/green] Configuration exported to {output_file}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to export configuration: {e}")


def _import_configuration(config_manager: ConfigManager, input_file: Path) -> None:
    """Import configuration from file."""
    
    try:
        config_manager.import_config(input_file)
        console.print(f"[green]✓[/green] Configuration imported from {input_file}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to import configuration: {e}")


def _convert_value(value: str):
    """Convert string value to appropriate type."""
    
    # Try boolean
    if value.lower() in ["true", "yes", "1", "on"]:
        return True
    elif value.lower() in ["false", "no", "0", "off"]:
        return False
    
    # Try integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value