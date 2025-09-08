"""
Command Line Interface for TRIDENT-Net.

Author: Yağızhan Keskin
"""

import logging
from pathlib import Path
from typing import Optional, List

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.logging import RichHandler
    TYPER_AVAILABLE = True
except ImportError:
    # Simple fallback for typer/rich
    class typer:
        class Typer:
            def __init__(self, **kwargs):
                pass
            def command(self):
                def decorator(func):
                    return func
                return decorator
        @staticmethod
        def Option(default, *args, **kwargs):
            return default
        @staticmethod
        def echo(msg):
            print(msg)
    
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    
    class Table:
        def __init__(self, **kwargs):
            pass
        def add_column(self, *args, **kwargs):
            pass
        def add_row(self, *args, **kwargs):
            pass
    
    class RichHandler:
        def __init__(self, **kwargs):
            pass
    
    TYPER_AVAILABLE = False

import torch

from trident.runtime.config import load_config, ConfigLoader
from trident.runtime.trainer import Trainer
from trident.runtime.evaluator import Evaluator
from trident.runtime.server import TridentServer
from trident.runtime.graph import create_inference_graph
from trident.data.dataset import create_data_loaders
from trident.data.synthetic import create_synthetic_dataset

# CLI app
app = typer.Typer(name="trident", help="TRIDENT-Net: Modular Multimodal Fusion System")
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def run_task(
    name: str = typer.Argument(..., help="Task name from configuration"),
    config: str = typer.Option("tasks.yml", "--config", "-c", help="Configuration file path"),
    data_root: Optional[str] = typer.Option(None, "--data-root", help="Override data root path"),
    device: Optional[str] = typer.Option(None, "--device", help="Device to use (cuda/cpu)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    synthetic_data: bool = typer.Option(False, "--synthetic", help="Use synthetic data"),
):
    """Run a training or evaluation task."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            raise typer.Exit(1)
        
        trident_config = load_config(config_path)
        config_loader = ConfigLoader()
        config_loader.config = trident_config
        
        # Override settings
        if data_root:
            trident_config.paths.data_root = data_root
        
        # Setup device
        if device:
            target_device = torch.device(device)
        else:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Running task: {name}")
        logger.info(f"Using device: {target_device}")
        
        # Get task configuration
        task_config = config_loader.get_task_config(name)
        
        # Create data loaders
        if synthetic_data:
            logger.info("Using synthetic data")
            from trident.data.synthetic import SyntheticDataset
            
            # Create synthetic datasets
            train_dataset = SyntheticDataset(n_samples=1000)
            val_dataset = SyntheticDataset(n_samples=200)
            test_dataset = SyntheticDataset(n_samples=200)
            
            batch_size_val = batch_size or task_config.batch_size
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size_val, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size_val, shuffle=False
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size_val, shuffle=False
            )
            
            data_loaders = (train_loader, val_loader, test_loader)
        else:
            data_loaders = create_data_loaders(
                data_root=trident_config.paths.data_root,
                batch_size=batch_size or task_config.batch_size,
            )
        
        # Run task based on type
        if task_config.run == "train":
            trainer = Trainer(config_loader, device=target_device)
            results = trainer.train_single_component(name, data_loaders)
            
        elif task_config.run == "train_multi":
            trainer = Trainer(config_loader, device=target_device)
            results = trainer.train_multi_component(name, data_loaders)
            
        elif task_config.run == "train_fusion":
            trainer = Trainer(config_loader, device=target_device)
            
            # Load frozen component checkpoints
            frozen_checkpoints = {}
            for comp_name in task_config.freeze or []:
                ckpt_path = f"{trident_config.paths.ckpt_root}/{comp_name}.pt"
                if Path(ckpt_path).exists():
                    frozen_checkpoints[comp_name] = ckpt_path
                else:
                    logger.warning(f"Checkpoint not found for {comp_name}: {ckpt_path}")
            
            results = trainer.train_fusion(name, data_loaders, frozen_checkpoints)
            
        elif task_config.run == "evaluate":
            evaluator = Evaluator(config_loader, device=target_device)
            
            # Load component checkpoints
            checkpoint_map = task_config.checkpoint_map or {}
            results = evaluator.evaluate_system(task_config.components, data_loaders[2], checkpoint_map)
            
        elif task_config.run == "fit_classical":
            # Handle classical ML training (SVM)
            logger.info("Classical ML training not fully implemented in CLI")
            results = {"status": "not_implemented"}
            
        elif task_config.run == "serve":
            # Handle serving
            logger.info("Serving mode - use 'trident serve' command instead")
            results = {"status": "use_serve_command"}
            
        else:
            logger.error(f"Unknown task run type: {task_config.run}")
            raise typer.Exit(1)
        
        # Display results
        console.print("[green]Task completed successfully![/green]")
        console.print(f"Results: {results}")
        
        # Save results if specified
        if hasattr(task_config, 'report_to') and task_config.report_to:
            import json
            with open(task_config.report_to, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {task_config.report_to}")
        
    except Exception as e:
        logger.error(f"Task failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def serve(
    config: str = typer.Option("tasks.yml", "--config", "-c", help="Configuration file path"),
    task: str = typer.Option("infer_realtime", "--task", help="Serving task name"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Start TRIDENT-Net inference server."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        trident_config = load_config(config)
        config_loader = ConfigLoader()
        config_loader.config = trident_config
        
        task_config = config_loader.get_task_config(task)
        
        # Create inference graph
        components = task_config.graph.get("order", [])
        checkpoint_map = task_config.checkpoint_map or {}
        
        graph = create_inference_graph(
            trident_config,
            components,
            checkpoint_map,
            frozen_components=components,  # All components frozen for inference
        )
        
        # Start server
        server = TridentServer(graph, host=host, port=port)
        
        console.print(f"[green]Starting TRIDENT-Net server on {host}:{port}[/green]")
        server.start()
        
    except Exception as e:
        logger.error(f"Server failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def eval(
    config: str = typer.Option("tasks.yml", "--config", "-c", help="Configuration file path"),
    components: List[str] = typer.Option([], "--component", help="Components to evaluate"),
    checkpoint_dir: str = typer.Option("./checkpoints", "--checkpoint-dir", help="Checkpoint directory"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    synthetic_data: bool = typer.Option(False, "--synthetic", help="Use synthetic data"),
):
    """Evaluate TRIDENT-Net components."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        trident_config = load_config(config)
        config_loader = ConfigLoader()
        config_loader.config = trident_config
        
        # Use all components if none specified
        if not components:
            components = list(trident_config.components.keys())
        
        # Build checkpoint map
        checkpoint_map = {}
        checkpoint_dir = Path(checkpoint_dir)
        
        for comp_name in components:
            ckpt_path = checkpoint_dir / f"{comp_name}.pt"
            if ckpt_path.exists():
                checkpoint_map[comp_name] = str(ckpt_path)
            else:
                logger.warning(f"Checkpoint not found for {comp_name}: {ckpt_path}")
        
        # Create data loaders
        if synthetic_data:
            from trident.data.synthetic import SyntheticDataset
            test_dataset = SyntheticDataset(n_samples=500)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        else:
            _, _, test_loader = create_data_loaders(trident_config.paths.data_root)
        
        # Evaluate
        evaluator = Evaluator(config_loader)
        results = evaluator.evaluate_system(components, test_loader, checkpoint_map)
        
        # Display results
        console.print("[green]Evaluation completed![/green]")
        
        # Create results table
        table = Table(title="TRIDENT-Net Evaluation Results")
        table.add_column("Component", style="cyan")
        table.add_column("AUROC", justify="right")
        table.add_column("F1", justify="right") 
        table.add_column("Brier", justify="right")
        
        for comp_name in components:
            comp_results = results.get(f"{comp_name}_metrics", {})
            table.add_row(
                comp_name,
                f"{comp_results.get('auroc', 0.0):.3f}",
                f"{comp_results.get('f1', 0.0):.3f}",
                f"{comp_results.get('brier', 0.0):.3f}",
            )
        
        console.print(table)
        
        # Save results
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def list_components(
    config: str = typer.Option("tasks.yml", "--config", "-c", help="Configuration file path"),
):
    """List available components from configuration."""
    try:
        trident_config = load_config(config)
        
        # Create components table
        table = Table(title="TRIDENT-Net Components")
        table.add_column("Name", style="cyan")
        table.add_column("Class", style="magenta")
        table.add_column("Kind", style="yellow")
        table.add_column("Inputs", style="green")
        table.add_column("Outputs", style="blue")
        
        for comp_name, comp_config in trident_config.components.items():
            table.add_row(
                comp_name,
                comp_config.class_path,
                comp_config.kind,
                ", ".join(comp_config.inputs),
                ", ".join(comp_config.outputs),
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tasks(
    config: str = typer.Option("tasks.yml", "--config", "-c", help="Configuration file path"),
):
    """List available tasks from configuration."""
    try:
        trident_config = load_config(config)
        
        # Create tasks table
        table = Table(title="TRIDENT-Net Tasks")
        table.add_column("Name", style="cyan")
        table.add_column("Run Type", style="magenta")
        table.add_column("Component(s)", style="yellow")
        table.add_column("Epochs", justify="right")
        table.add_column("Output", style="green")
        
        for task_name, task_config in trident_config.tasks.items():
            components = task_config.component or ", ".join(task_config.components or [])
            
            table.add_row(
                task_name,
                task_config.run,
                components,
                str(task_config.epochs),
                task_config.save_to,
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_synthetic_data(
    output_dir: str = typer.Argument(..., help="Output directory for synthetic data"),
    n_samples: int = typer.Option(1000, "--samples", "-n", help="Number of samples to generate"),
    splits: str = typer.Option("0.7,0.2,0.1", "--splits", help="Train/val/test split ratios"),
):
    """Create synthetic dataset for testing."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Parse splits
        split_ratios = [float(x) for x in splits.split(",")]
        if len(split_ratios) != 3 or sum(split_ratios) != 1.0:
            console.print("[red]Split ratios must sum to 1.0 and have 3 values[/red]")
            raise typer.Exit(1)
        
        console.print(f"Creating {n_samples} synthetic samples in {output_path}")
        
        # Create synthetic dataset
        dataset = create_synthetic_dataset(
            n_samples=n_samples,
            split_ratios=tuple(split_ratios),
            save_path=str(output_path),
        )
        
        console.print("[green]Synthetic dataset created successfully![/green]")
        
        # Show statistics
        for split_name, samples in dataset.items():
            console.print(f"{split_name}: {len(samples)} samples")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show TRIDENT-Net version."""
    from trident import __version__, __author__
    
    console.print(f"TRIDENT-Net version {__version__}")
    console.print(f"Author: {__author__}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()