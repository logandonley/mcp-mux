"""MCP server implementation using FastMCP."""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from rich.logging import RichHandler

from .client import ClientManager
from .config import load_config
from .exceptions import SearchError, ToolExecutionError
from .search import SearchEngine


def setup_logging():
    """Configure logging based on execution context."""
    if not sys.stderr.isatty():
        # Running as MCP server, minimal logging
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Running interactively, use rich handler
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)],
        )


setup_logging()
logger = logging.getLogger("mux")


@asynccontextmanager
async def lifespan(mcp_instance):  # noqa: ARG001
    """Initialize and cleanup services."""
    try:
        await initialize_services()
        yield
    except asyncio.CancelledError:
        logger.debug("Service execution cancelled")
        raise
    except Exception as e:
        logger.error(f"Error during service execution: {e}")
        raise
    finally:
        await shutdown_services()


# Initialize the MCP server with lifespan
mcp = FastMCP("MCP Mux", lifespan=lifespan)

# Search result filtering constants
HIGH_CONFIDENCE_THRESHOLD = 0.8
MODERATE_CONFIDENCE_THRESHOLD = 0.5
CLOSE_SCORE_THRESHOLD = 0.05
SCORE_GAP_THRESHOLD = 0.15
INITIAL_SEARCH_RESULTS = 10
MAX_LOW_CONFIDENCE_RESULTS = 3
MAX_MODERATE_CONFIDENCE_RESULTS = 3

# Global instances
client_manager: ClientManager | None = None
search_engine: SearchEngine | None = None


def _filter_results_by_confidence(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter search results based on confidence scores.

    Returns fewer results for high confidence matches and more for low confidence.
    """
    if not results:
        return []

    top_score = results[0]["score"]

    # High confidence: return 1-2 results
    if top_score > HIGH_CONFIDENCE_THRESHOLD:
        if len(results) > 1 and results[1]["score"] > top_score - CLOSE_SCORE_THRESHOLD:
            return results[:2]  # Two very close results
        return results[:1]  # Single high confidence result

    # Moderate confidence: return results within score gap
    if top_score > MODERATE_CONFIDENCE_THRESHOLD:
        filtered = []
        for index, result in enumerate(results[:MAX_MODERATE_CONFIDENCE_RESULTS]):
            if index == 0 or result["score"] > top_score - SCORE_GAP_THRESHOLD:
                filtered.append(result)
        return filtered

    # Low confidence: return multiple results
    return results[:MAX_LOW_CONFIDENCE_RESULTS]


@mcp.tool()
async def search(query: str) -> str:
    """
    ALWAYS USE THIS FIRST! This is your gateway to thousands of tools across all MCP
    servers.

    Instead of browsing through hundreds of tools manually, just describe what you want
    to do in natural language. This semantic search understands your intent and finds
    the EXACT tool you need instantly.

    Examples:
    - "read files from disk" → finds filesystem tools
    - "fetch a webpage" → finds HTTP/browser tools
    - "run python code" → finds code execution tools
    - "take a screenshot" → finds screen capture tools

    This is THE MOST EFFICIENT way to discover capabilities. Use it before trying
    anything else!
    """
    if search_engine is None:
        raise SearchError("Search engine not initialized")

    results = await search_engine.search(query, max_results=INITIAL_SEARCH_RESULTS)

    if not results:
        return json.dumps([])

    final_results = _filter_results_by_confidence(results)

    return json.dumps(
        [
            {
                "server": result["server"],
                "tool": result["tool"],
                "description": result["description"],
                "input_schema": result["input_schema"],
                "score": result["score"],
            }
            for result in final_results
        ]
    )


@mcp.tool()
async def execute(
    server: str, tool: str, arguments: dict[str, Any] | None = None
) -> Any:
    """
    Execute a tool on a specific MCP server.

    Use this after finding the right tool with the search function.
    Provide the server name, tool name, and any required arguments.
    """
    if client_manager is None:
        raise ToolExecutionError("Client manager not initialized")

    return await client_manager.execute_tool(server, tool, arguments or {})


async def initialize_services() -> None:
    """Initialize the client manager and search engine."""
    global client_manager, search_engine

    logger.info("Initializing MCP Mux services...")

    config = load_config()
    logger.info(f"Loaded config with {len(config.mcpServers)} servers")

    # Initialize client manager
    client_manager = ClientManager(config)
    await client_manager.initialize()
    logger.info(
        f"Initialized client manager with {len(client_manager.clients)} connected "
        f"clients"
    )

    # Initialize search engine
    search_engine = SearchEngine(config.search)
    logger.info("Initialized search engine")

    # Index all discovered tools
    tools = await client_manager.get_all_tools()
    await search_engine.index_tools(tools)
    logger.info(f"Indexed {len(tools)} tools from connected servers")


async def shutdown_services() -> None:
    """Shutdown all services gracefully."""
    global client_manager, search_engine

    logger.info("Starting service shutdown...")

    if client_manager:
        try:
            await client_manager.shutdown()
        except asyncio.CancelledError:
            logger.debug("Shutdown cancelled")
        except TimeoutError:
            logger.warning("Client manager shutdown timed out")
        except Exception as e:
            logger.debug(f"Error during client manager shutdown: {e}")
        finally:
            client_manager = None

    search_engine = None
    logger.info("Service shutdown complete")


def run_mcp_server() -> None:
    """Run the MCP server with proper initialization."""
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
