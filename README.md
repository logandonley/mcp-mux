# mcp-mux

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) multiplexer that simplifies tool access through semantic search.

## Overview

mcp-mux acts as a router between MCP clients and servers, providing a simplified interface that exposes only two tools (`search` and `execute`) instead of potentially hundreds. It uses semantic search to dynamically find and execute the right tools based on natural language queries.

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
