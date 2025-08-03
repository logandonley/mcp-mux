# mcp-mux

[![CI](https://github.com/logandonley/mcp-mux/workflows/CI/badge.svg)](https://github.com/logandonley/mcp-mux/actions)
[![Python Version](https://img.shields.io/pypi/pyversions/mcp-mux)](https://pypi.org/project/mcp-mux/)
[![PyPI](https://img.shields.io/pypi/v/mcp-mux)](https://pypi.org/project/mcp-mux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) multiplexer that intelligently routes to the right tool from hundreds of available options using semantic search.

## Why mcp-mux?

**Problem**: Managing multiple MCP servers means dealing with hundreds of tools, making it hard for LLMs to find the right one.

**Solution**: mcp-mux acts as an intelligent router that exposes just two tools:
- `search` - Find tools using natural language
- `execute` - Run the tool you found

This dramatically simplifies tool discovery and improves LLM performance with MCP servers.

## Requirements

- Python 3.10+
- MCP-compatible client (e.g., Claude Desktop)

## Installation

```bash
# Using uvx (recommended)
uvx mcp-mux init

# Or install globally
pip install mcp-mux
```

## Quick Start

1. Initialize configuration:

```bash
uvx mcp-mux init
```

2. Configure your MCP servers in `~/.mux/config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/Users/me/projects"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

3. Add mcp-mux to your MCP client configuration:

```json
{
  "mcpServers": {
    "mux": {
      "command": "uvx",
      "args": ["mcp-mux"]
    }
  }
}
```

## How It Works

Instead of exposing hundreds of tools from multiple servers, mcp-mux exposes just two:

1. **`search`** - Find tools using natural language
2. **`execute`** - Run a specific tool

## Configuration

```bash
# List available embedding models
uvx mcp-mux model --list

# Switch to a different model
uvx mcp-mux model BAAI/bge-small-en-v1.5
```

## License

MIT

## Development Status

This project is in early development. Features may change.
