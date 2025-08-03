# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

mcp-mux is a Model Context Protocol (MCP) multiplexer that acts as an intelligent router for multiple MCP servers. It exposes just two tools (`search` and `execute`) instead of hundreds, using semantic search to help LLMs find the right tool from many available options.

## Key Commands

### Development Setup
```bash
# Install dependencies using uv
uv sync --all-extras --dev

# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run pyright

# Run tests
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests only
uv run pytest --cov=mux               # All tests with coverage
uv run pytest tests/unit/test_server.py::test_search_tools  # Run specific test

# Run pre-commit hooks
pre-commit run --all-files
```

### Running the Server
```bash
# Run as MCP server (production mode)
uvx mcp-mux

# Run with dev dependencies
uv run mcp-mux

# Initialize configuration
uvx mcp-mux init

# Manage embedding models
uvx mcp-mux model --list
uvx mcp-mux model BAAI/bge-small-en-v1.5
```

## Architecture

### Core Components

1. **MCP Server (`src/mux/server.py`)**
   - FastMCP-based server implementation
   - Exposes `search` and `execute` tools
   - Manages lifespan and service initialization

2. **Client Manager (`src/mux/client.py`)**
   - Manages connections to multiple MCP servers
   - Handles server lifecycle (spawn, connect, disconnect)
   - Maintains server registry and tool information

3. **Search Engine (`src/mux/search.py`)**
   - Semantic search using FastEmbed
   - Indexes tools from all connected servers
   - Supports multiple embedding models

4. **Configuration (`src/mux/config.py`)**
   - Pydantic-based configuration management
   - Supports both file-based and environment variable config
   - Default config location: `~/.mux/config.json`

### Key Design Patterns

- **Async-first**: All server operations use asyncio
- **Type-safe**: Extensive use of Pydantic models and type hints
- **Modular**: Clear separation between server, client, search, and config
- **Error handling**: Custom exceptions in `exceptions.py`

### Tool Flow

1. Client requests `search` → SearchEngine finds matching tools across all servers
2. Client requests `execute` → ClientManager routes to appropriate server
3. Results returned through unified interface

## Testing Strategy

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test real MCP server interactions
- **E2E tests**: Test full workflow from search to execution
- **Platform testing**: CI runs on Ubuntu, macOS, and Windows

## Version Management

- Uses `setuptools_scm` for git-based versioning
- Version stored in `src/mux/_version.py` (auto-generated)
- Tagged releases trigger automatic PyPI deployment

## Important Patterns

- Always use `uv` for dependency management and virtual environments
- Follow existing code style (enforced by Ruff)
- Maintain type safety (enforced by Pyright)
- Keep logging minimal in production mode (when not running interactively)
