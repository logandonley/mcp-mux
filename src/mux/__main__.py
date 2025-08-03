"""Main entry point for the MCP Mux CLI."""

import sys

import click
from rich.console import Console

from . import __version__
from .config import get_config_path, init_config

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version=__version__, prog_name="mux")
def cli(ctx: click.Context) -> None:
    """MCP Mux - Model Context Protocol router with semantic search."""
    if ctx.invoked_subcommand is None:
        import asyncio
        import warnings

        # Suppress warnings about unclosed async generators during shutdown
        warnings.filterwarnings("ignore", message=".*asynchronous generator.*")

        import contextlib

        from .server import run_mcp_server

        with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
            run_mcp_server()


@cli.command()
def init() -> None:
    """Initialize MCP Mux configuration."""
    config_path = get_config_path()

    if config_path.exists() and not click.confirm(
        f"Config file already exists at {config_path}. Overwrite?"
    ):
        console.print("[yellow]Initialization cancelled.[/yellow]")
        return

    try:
        init_config()
        console.print(f"[green]✓[/green] Configuration initialized at {config_path}")
        console.print("\nYou can now:")
        console.print("  • Edit the config file to add MCP servers")
        console.print("  • Run 'mux' to start the MCP server")
        console.print("  • Run 'mux ui' to launch the configuration UI")
    except OSError as e:
        console.print(f"[red]Error writing configuration file: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error initializing configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the UI server to")
@click.option("--port", default=8080, help="Port to bind the UI server to")
@click.option("--dev", is_flag=True, help="Run in development mode")
def ui(host: str, port: int, dev: bool) -> None:  # noqa: ARG001
    """Launch the configuration UI server."""
    console.print("[yellow]UI command not yet implemented[/yellow]")
    console.print("The UI will be available in a future update.")
    # TODO: Implement FastAPI server launch


@cli.command()
@click.argument("model_name", required=False)
@click.option("--list", "-l", is_flag=True, help="List available embedding models")
def model(model_name: str | None, list: bool) -> None:
    """Manage embedding models for semantic search."""
    from .config import load_config, save_config
    from .search import RECOMMENDED_MODELS

    if list:
        console.print("\n[bold]Available Embedding Models:[/bold]\n")

        config = load_config()
        current_model = config.search.model

        for model_info in RECOMMENDED_MODELS:
            name = model_info["name"]
            if name == current_model:
                console.print(f"  • {name} [green](current)[/green]")
            else:
                console.print(f"  • {name}")
            console.print(f"    {model_info['description']}")
            dims = model_info["dimensions"]
            speed = model_info["speed"]
            console.print(f"    Dimensions: {dims}, Speed: {speed}\n")

        return

    if model_name is None:
        config = load_config()
        console.print(f"Current model: [green]{config.search.model}[/green]")
        console.print("\nUse 'mux model --list' to see available models")
        console.print("Use 'mux model <model-name>' to switch models")
        return

    # Switch to new model
    valid_models = [m["name"] for m in RECOMMENDED_MODELS]
    if model_name not in valid_models:
        console.print(f"[red]Error: Unknown model '{model_name}'[/red]")
        console.print(f"\nAvailable models: {', '.join(valid_models)}")
        sys.exit(1)

    config = load_config()
    old_model = config.search.model

    if old_model == model_name:
        console.print(f"[yellow]Already using model '{model_name}'[/yellow]")
        return

    config.search.model = model_name
    save_config(config)

    console.print(f"[green]✓[/green] Switched from '{old_model}' to '{model_name}'")
    console.print("\nNote: The new model will be used the next time you start mux.")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
