"""Test MCP server for integration testing."""

import asyncio
import json
import logging
import sys
from typing import Any

from mcp import Resource, Tool
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from pydantic import AnyUrl

logger = logging.getLogger("test_mcp_server")

# Create test server
server = Server("test-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available test tools."""
    return [
        Tool(
            name="echo",
            description="Echo a message back to the user",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"},
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        ),
        Tool(
            name="list_files",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path",
                        "default": ".",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    """Execute a tool."""
    if not arguments:
        arguments = {}

    if name == "echo":
        message = arguments.get("message", "")
        return [TextContent(type="text", text=f"Echo: {message}")]

    if name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        result = a + b
        return [TextContent(type="text", text=f"Result: {result}")]

    if name == "get_weather":
        location = arguments.get("location", "Unknown")
        # Mock weather response
        return [
            TextContent(type="text", text=f"Weather in {location}: Sunny, 72°F (22°C)")
        ]

    if name == "list_files":
        path = arguments.get("path", ".")
        # Mock file listing
        files = ["file1.txt", "file2.py", "data.json", "README.md"]
        return [TextContent(type="text", text=f"Files in {path}: {', '.join(files)}")]

    raise ValueError(f"Unknown tool: {name}")


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri=AnyUrl("file:///test/data.json"),
            name="Test Data",
            description="Sample test data",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read a resource."""
    if str(uri) == "file:///test/data.json":
        return json.dumps({"test": "data", "items": [1, 2, 3]})
    raise ValueError(f"Unknown resource: {uri}")


async def run_server():
    """Run the test MCP server."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="test-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    # Handle keyboard interrupt gracefully
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        sys.exit(0)
