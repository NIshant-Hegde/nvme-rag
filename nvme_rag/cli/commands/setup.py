"""
Setup and installation commands for NVMe RAG CLI.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from nvme_rag.config import ConfigManager

console = Console()


@click.group(name="setup")
def setup_group():
    """Setup and installation commands."""
    pass


@setup_group.command()
@click.option("--force", is_flag=True, help="Force reinstallation")
@click.option("--models", help="Comma-separated list of models to install")
@click.option("--skip-ollama", is_flag=True, help="Skip Ollama installation")
@click.option("--config-only", is_flag=True, help="Only create configuration")
@click.pass_context
def setup(ctx: click.Context, force: bool, models: Optional[str], skip_ollama: bool, config_only: bool):
    """Interactive setup wizard for NVMe RAG system."""
    
    console.print("[bold blue]NVMe RAG Setup Wizard[/bold blue]")
    console.print("This wizard will guide you through setting up the NVMe RAG system.\n")
    
    # Get configuration manager
    config_manager = ctx.obj.get("config_manager") or ConfigManager()
    
    # Step 1: Configuration setup
    console.print("[bold]Step 1: Configuration[/bold]")
    if config_only or _setup_configuration(config_manager, force):
        if config_only:
            console.print("[green]✓[/green] Configuration created successfully!")
            return
    
    # Step 2: System dependencies
    console.print("\n[bold]Step 2: System Dependencies[/bold]")
    if not _check_system_dependencies():
        console.print("[red]✗[/red] System dependency check failed!")
        return
    
    # Step 3: Python dependencies
    console.print("\n[bold]Step 3: Python Dependencies[/bold]")
    if not _install_python_dependencies(force):
        console.print("[red]✗[/red] Python dependency installation failed!")
        return
    
    # Step 4: Ollama setup
    if not skip_ollama:
        console.print("\n[bold]Step 4: Ollama Setup[/bold]")
        if not _setup_ollama(config_manager.config.ollama.model):
            console.print("[red]✗[/red] Ollama setup failed!")
            return
    
    # Step 5: Model installation
    console.print("\n[bold]Step 5: Model Installation[/bold]")
    model_list = models.split(",") if models else None
    if not _install_models(model_list, config_manager):
        console.print("[yellow]![/yellow] Model installation had issues")
    
    # Step 6: Directory structure
    console.print("\n[bold]Step 6: Directory Structure[/bold]")
    _setup_directories(config_manager)
    
    # Step 7: Health check
    console.print("\n[bold]Step 7: System Health Check[/bold]")
    if _run_health_check(config_manager):
        console.print("\n[green]✓ Setup completed successfully![/green]")
        console.print("\nNext steps:")
        console.print("1. Add a document: [cyan]nvme-rag add-document /path/to/nvme-spec.pdf[/cyan]")
        console.print("2. Ask a question: [cyan]nvme-rag ask \"What is NVMe?\"[/cyan]")
        console.print("3. Start interactive chat: [cyan]nvme-rag chat[/cyan]")
    else:
        console.print("\n[yellow]⚠ Setup completed with warnings. Run 'nvme-rag status' for details.[/yellow]")


def _setup_configuration(config_manager: ConfigManager, force: bool) -> bool:
    """Setup configuration interactively."""
    
    config_file = config_manager.config_file
    
    if config_file.exists() and not force:
        if not Confirm.ask(f"Configuration file exists at {config_file}. Overwrite?"):
            console.print("[green]✓[/green] Using existing configuration")
            return True
    
    console.print("Setting up configuration...")
    
    # Get user preferences
    data_dir = Prompt.ask(
        "Data directory", 
        default=str(Path.home() / ".nvme-rag" / "data")
    )
    
    ollama_url = Prompt.ask(
        "Ollama server URL", 
        default="http://localhost:11434"
    )
    
    ollama_model = Prompt.ask(
        "Default Ollama model", 
        default="gemma3:12b-it-qat"
    )
    
    # Update configuration
    config = config_manager.config
    config.system.data_dir = Path(data_dir)
    config.ollama.base_url = ollama_url
    config.ollama.model = ollama_model
    
    # Save configuration
    try:
        config_manager.save()
        console.print(f"[green]✓[/green] Configuration saved to {config_file}")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to save configuration: {e}")
        return False


def _check_system_dependencies() -> bool:
    """Check system dependencies."""
    
    dependencies = [
        ("python", "Python 3.8+"),
        ("pip", "Python package installer"),
    ]
    
    all_good = True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for cmd, description in dependencies:
            task = progress.add_task(f"Checking {description}...", total=None)
            
            try:
                result = subprocess.run([cmd, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    console.print(f"[green]✓[/green] {description}: Found")
                else:
                    console.print(f"[red]✗[/red] {description}: Not found")
                    all_good = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                console.print(f"[red]✗[/red] {description}: Not found")
                all_good = False
            
            progress.remove_task(task)
    
    return all_good


def _install_python_dependencies(force: bool) -> bool:
    """Install Python dependencies."""
    
    console.print("Installing Python dependencies...")
    
    # Check if requirements file exists
    req_file = Path("requirements.txt")
    if not req_file.exists():
        console.print("[red]✗[/red] requirements.txt not found!")
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if force:
            cmd.append("--force-reinstall")
        cmd.extend(["-r", str(req_file)])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("Installing packages..."),
            console=console
        ) as progress:
            task = progress.add_task("Installing...", total=None)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            progress.remove_task(task)
        
        if result.returncode == 0:
            console.print("[green]✓[/green] Python dependencies installed successfully")
            return True
        else:
            console.print(f"[red]✗[/red] Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        console.print(f"[red]✗[/red] Installation failed: {e}")
        return False


def _setup_ollama(default_model: str) -> bool:
    """Setup Ollama server."""
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            console.print("[green]✓[/green] Ollama is installed")
        else:
            console.print("[yellow]![/yellow] Ollama not found")
            if not _install_ollama():
                return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("[yellow]![/yellow] Ollama not found")
        if not _install_ollama():
            return False
    
    # Check if server is running
    try:
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            console.print("[green]✓[/green] Ollama server is accessible")
            return True
        else:
            console.print("[yellow]![/yellow] Ollama server not responding")
            return _start_ollama_server()
    except Exception:
        console.print("[yellow]![/yellow] Ollama server not responding")
        return _start_ollama_server()


def _install_ollama() -> bool:
    """Install Ollama if not present."""
    
    if not Confirm.ask("Ollama not found. Install it automatically?"):
        console.print("Please install Ollama manually from https://ollama.ai")
        return False
    
    console.print("Installing Ollama...")
    
    try:
        # Use the official Ollama installation script
        result = subprocess.run([
            "curl", "-fsSL", "https://ollama.ai/install.sh"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Run the installation script
            result = subprocess.run([
                "sh", "-c", result.stdout
            ], timeout=120)
            
            if result.returncode == 0:
                console.print("[green]✓[/green] Ollama installed successfully")
                return True
        
        console.print("[red]✗[/red] Ollama installation failed")
        return False
        
    except Exception as e:
        console.print(f"[red]✗[/red] Ollama installation failed: {e}")
        return False


def _start_ollama_server() -> bool:
    """Start Ollama server."""
    
    console.print("Starting Ollama server...")
    
    try:
        # Start server in background
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        # Check if server is responding
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            console.print("[green]✓[/green] Ollama server started successfully")
            return True
        else:
            console.print("[red]✗[/red] Failed to start Ollama server")
            return False
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start Ollama server: {e}")
        return False


def _install_models(model_list: Optional[List[str]], config_manager: ConfigManager) -> bool:
    """Install required models."""
    
    if not model_list:
        # Use default models
        model_list = [
            config_manager.config.ollama.model,
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ]
    
    success = True
    
    for model in model_list:
        console.print(f"Installing model: {model}")
        
        if model.startswith("sentence-transformers/"):
            # This is a Hugging Face model - will be downloaded on first use
            console.print(f"[green]✓[/green] {model} will be downloaded on first use")
        else:
            # This is an Ollama model
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Downloading {model}..."),
                    console=console
                ) as progress:
                    task = progress.add_task("Downloading...", total=None)
                    
                    result = subprocess.run(["ollama", "pull", model], 
                                          capture_output=True, text=True, timeout=300)
                    progress.remove_task(task)
                
                if result.returncode == 0:
                    console.print(f"[green]✓[/green] {model} installed successfully")
                else:
                    console.print(f"[red]✗[/red] Failed to install {model}: {result.stderr}")
                    success = False
                    
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to install {model}: {e}")
                success = False
    
    return success


def _setup_directories(config_manager: ConfigManager) -> None:
    """Setup directory structure."""
    
    console.print("Creating directory structure...")
    
    try:
        config_manager.config.create_directories()
        console.print("[green]✓[/green] Directory structure created")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create directories: {e}")


def _run_health_check(config_manager: ConfigManager) -> bool:
    """Run system health check."""
    
    console.print("Running health check...")
    
    # This would normally import and run actual health checks
    # For now, just simulate the check
    
    issues = []
    
    # Check configuration
    validation = config_manager.validate()
    if not validation["valid"]:
        issues.extend(validation["issues"])
    
    # Check directories
    paths = config_manager.config.get_data_paths()
    for name, path in paths.items():
        if not path.exists() and not str(path).endswith(('.yml', '.yaml', '.json')):
            issues.append(f"Directory missing: {path}")
    
    if issues:
        console.print("[yellow]⚠ Health check found issues:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
        return False
    else:
        console.print("[green]✓[/green] Health check passed")
        return True


@setup_group.command()
@click.option("--list", "list_models", is_flag=True, help="List available models")
@click.option("--model", help="Install specific model")
@click.option("--all", "install_all", is_flag=True, help="Install all recommended models")
def install_models(list_models: bool, model: Optional[str], install_all: bool):
    """Install and manage AI models."""
    
    if list_models:
        _list_available_models()
    elif model:
        _install_single_model(model)
    elif install_all:
        _install_all_models()
    else:
        console.print("Please specify --list, --model, or --all")


def _list_available_models() -> None:
    """List available models."""
    
    console.print("[bold]Available Models[/bold]\n")
    
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description")
    table.add_column("Size")
    
    models = [
        ("gemma3:12b-it-qat", "LLM", "Quantized Gemma 3 12B for chat", "~7GB"),
        ("llama3.2:3b", "LLM", "Compact Llama 3.2 3B", "~2GB"),
        ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "Embedding", "Multi-QA MiniLM for embeddings", "~90MB"),
        ("sentence-transformers/all-mpnet-base-v2", "Embedding", "All-purpose MPNet embeddings", "~420MB"),
    ]
    
    for model_name, model_type, description, size in models:
        table.add_row(model_name, model_type, description, size)
    
    console.print(table)


def _install_single_model(model: str) -> None:
    """Install a single model."""
    console.print(f"Installing model: {model}")
    _install_models([model], ConfigManager())


def _install_all_models() -> None:
    """Install all recommended models."""
    console.print("Installing all recommended models...")
    _install_models(None, ConfigManager())