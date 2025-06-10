"""
System management commands for NVMe RAG CLI.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from nvme_rag.config import ConfigManager

console = Console()


@click.group(name="system")
def system_group():
    """System management and status commands."""
    pass


@system_group.command()
@click.option("--detailed", is_flag=True, help="Show detailed status information")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--health-check", is_flag=True, help="Run comprehensive health check")
@click.pass_context
def status(ctx: click.Context, detailed: bool, output_json: bool, health_check: bool):
    """Check system status and health."""
    
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    status_info = _get_system_status(config_manager, detailed, health_check)
    
    if output_json:
        console.print(json.dumps(status_info, indent=2, default=str))
    else:
        _display_status(status_info, detailed)


@system_group.command()
@click.option("--wait", is_flag=True, help="Wait for server to be ready")
@click.option("--timeout", default=30, help="Server startup timeout in seconds")
def start_server(wait: bool, timeout: int):
    """Start Ollama server."""
    
    console.print("Starting Ollama server...")
    
    try:
        # Check if already running
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            console.print("[green]✓[/green] Ollama server is already running")
            return
    except:
        pass
    
    try:
        # Start server
        process = subprocess.Popen(["ollama", "serve"], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
        
        if wait:
            import time
            for i in range(timeout):
                time.sleep(1)
                try:
                    result = subprocess.run(["ollama", "list"], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        console.print("[green]✓[/green] Ollama server started successfully")
                        return
                except:
                    continue
            
            console.print("[red]✗[/red] Server startup timed out")
        else:
            console.print("[green]✓[/green] Ollama server start command issued")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start Ollama server: {e}")


@system_group.command()
@click.option("--force", is_flag=True, help="Force stop server")
def stop_server(force: bool):
    """Stop Ollama server."""
    
    console.print("Stopping Ollama server...")
    
    try:
        if force:
            # Kill the process
            subprocess.run(["pkill", "-f", "ollama"], timeout=10)
        else:
            # Try graceful shutdown (if such command exists)
            # For now, we'll use the same approach
            subprocess.run(["pkill", "-f", "ollama"], timeout=10)
        
        console.print("[green]✓[/green] Ollama server stopped")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to stop Ollama server: {e}")


@system_group.command()
@click.option("--keep-docs", is_flag=True, help="Keep document data")
@click.option("--keep-config", is_flag=True, help="Keep configuration")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def reset(ctx: click.Context, keep_docs: bool, keep_config: bool, confirm: bool):
    """Reset system to clean state."""
    
    if not confirm:
        from rich.prompt import Confirm
        if not Confirm.ask("[red]This will reset the NVMe RAG system. Continue?[/red]"):
            console.print("Reset cancelled.")
            return
    
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    
    console.print("Resetting system...")
    
    try:
        paths = config_manager.config.get_data_paths()
        
        # Remove vector store
        if not keep_docs and paths["vector_store"].exists():
            import shutil
            shutil.rmtree(paths["vector_store"])
            console.print("[green]✓[/green] Vector store cleared")
        
        # Remove processed documents
        if not keep_docs and paths["processed"].exists():
            import shutil
            shutil.rmtree(paths["processed"])
            console.print("[green]✓[/green] Processed documents cleared")
        
        # Remove embeddings cache
        if not keep_docs and paths["embeddings_cache"].exists():
            import shutil
            shutil.rmtree(paths["embeddings_cache"])
            console.print("[green]✓[/green] Embeddings cache cleared")
        
        # Reset configuration
        if not keep_config:
            config_manager.reset_to_defaults()
            console.print("[green]✓[/green] Configuration reset to defaults")
        
        # Recreate directories
        config_manager.config.create_directories()
        console.print("[green]✓[/green] Directory structure recreated")
        
        console.print("\n[green]System reset completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Reset failed: {e}")


def _get_system_status(config_manager: ConfigManager, detailed: bool, health_check: bool) -> Dict[str, Any]:
    """Get comprehensive system status."""
    
    status = {
        "system_health": "unknown",
        "ollama": {"status": "unknown"},
        "vector_store": {"status": "unknown"},
        "models": {},
        "storage": {},
        "configuration": {"status": "unknown"}
    }
    
    # Check configuration
    try:
        validation = config_manager.validate()
        status["configuration"] = {
            "status": "healthy" if validation["valid"] else "issues",
            "config_file": str(config_manager.config_file),
            "issues": validation.get("issues", []),
            "warnings": validation.get("warnings", [])
        }
    except Exception as e:
        status["configuration"] = {"status": "error", "error": str(e)}
    
    # Check Ollama server
    try:
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            status["ollama"]["status"] = "running"
            status["ollama"]["url"] = config_manager.config.ollama.base_url
            status["ollama"]["models"] = _parse_ollama_models(result.stdout)
        else:
            status["ollama"]["status"] = "not_responding"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        status["ollama"]["status"] = "not_installed"
    except Exception as e:
        status["ollama"]["status"] = "error"
        status["ollama"]["error"] = str(e)
    
    # Check vector store
    try:
        paths = config_manager.config.get_data_paths()
        vector_store_path = paths["vector_store"]
        
        if vector_store_path.exists():
            # Try to get vector store info
            status["vector_store"]["status"] = "ready"
            status["vector_store"]["path"] = str(vector_store_path)
            
            # Count documents/chunks if possible
            if detailed:
                status["vector_store"]["size"] = _get_directory_size(vector_store_path)
        else:
            status["vector_store"]["status"] = "not_initialized"
            
    except Exception as e:
        status["vector_store"]["status"] = "error"
        status["vector_store"]["error"] = str(e)
    
    # Check storage usage
    try:
        paths = config_manager.config.get_data_paths()
        status["storage"] = {
            "data_dir": str(paths["data_dir"]),
            "total_size": _get_directory_size(paths["data_dir"]) if paths["data_dir"].exists() else 0
        }
        
        if detailed:
            for name, path in paths.items():
                if path.exists() and not str(path).endswith(('.yml', '.yaml', '.json')):
                    status["storage"][f"{name}_size"] = _get_directory_size(path)
                    
    except Exception as e:
        status["storage"]["error"] = str(e)
    
    # Overall health assessment
    issues = []
    if status["configuration"]["status"] != "healthy":
        issues.append("configuration")
    if status["ollama"]["status"] != "running":
        issues.append("ollama")
    if status["vector_store"]["status"] not in ["ready", "not_initialized"]:
        issues.append("vector_store")
    
    if not issues:
        status["system_health"] = "healthy"
    elif len(issues) == 1 and "not_initialized" in str(status):
        status["system_health"] = "needs_setup"
    else:
        status["system_health"] = "unhealthy"
    
    status["issues"] = issues
    
    return status


def _parse_ollama_models(output: str) -> list:
    """Parse ollama list output to extract model information."""
    models = []
    lines = output.strip().split('\n')[1:]  # Skip header
    
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                models.append({
                    "name": parts[0],
                    "id": parts[1] if len(parts) > 1 else "",
                    "size": parts[2] if len(parts) > 2 else "",
                    "modified": " ".join(parts[3:]) if len(parts) > 3 else ""
                })
    
    return models


def _get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except:
        pass
    return total


def _format_size(bytes_size: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def _display_status(status_info: Dict[str, Any], detailed: bool) -> None:
    """Display status information in a nice format."""
    
    # System health header
    health = status_info["system_health"]
    health_color = {
        "healthy": "green",
        "needs_setup": "yellow", 
        "unhealthy": "red",
        "unknown": "yellow"
    }.get(health, "yellow")
    
    console.print(Panel(
        f"[bold {health_color}]{health.upper().replace('_', ' ')}[/bold {health_color}]",
        title="[bold blue]NVMe RAG System Status[/bold blue]",
        expand=False
    ))
    
    # Create status table
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details")
    
    # Configuration status
    config_status = status_info["configuration"]["status"]
    config_color = "green" if config_status == "healthy" else "red"
    table.add_row(
        "Configuration",
        f"[{config_color}]{config_status}[/{config_color}]",
        status_info["configuration"].get("config_file", "")
    )
    
    # Ollama status
    ollama_status = status_info["ollama"]["status"]
    ollama_color = "green" if ollama_status == "running" else "red"
    ollama_details = ""
    if ollama_status == "running":
        url = status_info["ollama"].get("url", "")
        models = status_info["ollama"].get("models", [])
        ollama_details = f"{url} ({len(models)} models)"
    
    table.add_row(
        "Ollama Server",
        f"[{ollama_color}]{ollama_status.replace('_', ' ')}[/{ollama_color}]",
        ollama_details
    )
    
    # Vector store status
    vs_status = status_info["vector_store"]["status"]
    vs_color = "green" if vs_status == "ready" else "yellow" if vs_status == "not_initialized" else "red"
    vs_details = ""
    if vs_status == "ready" and "size" in status_info["vector_store"]:
        size = status_info["vector_store"]["size"]
        vs_details = _format_size(size)
    
    table.add_row(
        "Vector Store",
        f"[{vs_color}]{vs_status.replace('_', ' ')}[/{vs_color}]",
        vs_details
    )
    
    console.print(table)
    
    # Storage information
    if detailed and "storage" in status_info:
        storage = status_info["storage"]
        if "total_size" in storage:
            console.print(f"\n[bold]Storage Usage[/bold]")
            console.print(f"Data Directory: {storage['data_dir']}")
            console.print(f"Total Size: {_format_size(storage['total_size'])}")
    
    # Show issues if any
    if status_info.get("issues"):
        console.print(f"\n[red]Issues found:[/red]")
        for issue in status_info["issues"]:
            console.print(f"  • {issue}")
    
    # Show configuration issues
    config_issues = status_info["configuration"].get("issues", [])
    if config_issues:
        console.print(f"\n[red]Configuration issues:[/red]")
        for issue in config_issues:
            console.print(f"  • {issue}")
    
    # Show recommendations
    if health != "healthy":
        console.print(f"\n[yellow]Recommendations:[/yellow]")
        if "ollama" in status_info.get("issues", []):
            console.print("  • Run: nvme-rag start-server")
        if status_info["vector_store"]["status"] == "not_initialized":
            console.print("  • Run: nvme-rag setup")
            console.print("  • Add documents: nvme-rag add-document <path>")
        if config_issues:
            console.print("  • Run: nvme-rag configure")