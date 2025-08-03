"""Shared fixtures and utilities for tests."""
# pyright: reportCallIssue=false

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from mux.config import Config, MCPServerConfig, SearchConfig

# Sample tool definitions for testing
SAMPLE_TOOLS = [
    {
        "server": "test-server",
        "tool": "echo",
        "description": "Echo a message back to the user",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo",
                }
            },
            "required": ["message"],
        },
    },
    {
        "server": "test-server",
        "tool": "add",
        "description": "Add two numbers together",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "server": "filesystem-server",
        "tool": "list_files",
        "description": "List files in a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path",
                }
            },
            "required": ["path"],
        },
    },
    {
        "server": "filesystem-server",
        "tool": "read_file",
        "description": "Read contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read",
                }
            },
            "required": ["path"],
        },
    },
    {
        "server": "web-server",
        "tool": "fetch_url",
        "description": "Fetch content from a URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional headers",
                },
            },
            "required": ["url"],
        },
    },
]


@pytest.fixture
def tmp_config_path(tmp_path):
    """Provide a temporary config file path."""
    return tmp_path / "test_config.json"


@pytest.fixture
def sample_config():
    """Provide a sample configuration."""
    return Config(
        mcpServers={
            "test-server": MCPServerConfig(
                command="test-cmd",
                args=["--test"],
                enabled=True,
            ),
            "filesystem-server": MCPServerConfig(
                command="fs-server",
                args=[],
                enabled=True,
            ),
            "web-server": MCPServerConfig(
                url="http://localhost:8080",
                enabled=True,
            ),
            "disabled-server": MCPServerConfig(
                command="disabled",
                enabled=False,
            ),
        },
        search=SearchConfig(
            method="local",
            model="BAAI/bge-small-en-v1.5",
            cache_embeddings=True,
            max_results=5,
        ),
    )


@pytest.fixture
def sample_tools():
    """Provide sample tool definitions."""
    return SAMPLE_TOOLS.copy()


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model that returns predictable embeddings."""

    class MockEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts: list[str]):
            """Generate predictable embeddings based on text content."""
            embeddings = []
            for text in texts:
                # Create a simple embedding based on text characteristics
                # This makes tests predictable
                text_lower = text.lower()
                embedding = [0.0] * 384  # Standard embedding size

                # Set different values based on keywords
                if "echo" in text_lower:
                    embedding[0] = 0.9
                elif "add" in text_lower:
                    embedding[1] = 0.9
                elif "list" in text_lower or "files" in text_lower:
                    embedding[2] = 0.9
                elif "read" in text_lower or "file" in text_lower:
                    embedding[3] = 0.9
                elif "fetch" in text_lower or "url" in text_lower:
                    embedding[4] = 0.9

                # Add some variation based on server name
                if "test-server" in text_lower:
                    embedding[10] = 0.5
                elif "filesystem" in text_lower:
                    embedding[11] = 0.5
                elif "web" in text_lower:
                    embedding[12] = 0.5

                embeddings.append(embedding)

            return iter(embeddings)

    return MockEmbedding


@pytest.fixture
def mock_mcp_session():
    """Mock MCP session for testing."""
    session = AsyncMock()

    # Mock tool listing
    tools_response = MagicMock()
    tools_response.tools = [
        MagicMock(
            name=tool["tool"],
            description=tool["description"],
            inputSchema=tool["input_schema"],
        )
        for tool in SAMPLE_TOOLS[:2]  # Only first 2 tools for this mock
    ]
    session.list_tools.return_value = tools_response

    # Mock tool execution
    async def mock_call_tool(tool_name: str, arguments: dict[str, Any]):
        result = MagicMock()
        result.content = []

        if tool_name == "echo":
            content = MagicMock()
            content.text = arguments.get("message", "")
            result.content = [content]
        elif tool_name == "add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            content = MagicMock()
            content.text = str(a + b)
            result.content = [content]

        return result

    session.call_tool = mock_call_tool
    return session


@pytest.fixture
def mock_stdio_client():
    """Mock stdio client context manager."""

    class MockStdioClient:
        def __init__(self, server_params):
            self.server_params = server_params
            self.read = AsyncMock()
            self.write = AsyncMock()

        async def __aenter__(self):
            return self.read, self.write

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockStdioClient


@pytest.fixture
def mock_http_client():
    """Mock HTTP client context manager."""

    class MockHttpClient:
        def __init__(self, url):
            self.url = url
            self.read = AsyncMock()
            self.write = AsyncMock()
            self.get_session_id = MagicMock(return_value="test-session-id")

        async def __aenter__(self):
            return self.read, self.write, self.get_session_id

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockHttpClient


@pytest_asyncio.fixture
async def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_client_session(mock_mcp_session):
    """Mock ClientSession class."""

    class MockClientSession:
        def __init__(self, read, write):
            self.read = read
            self.write = write
            self.session = mock_mcp_session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockClientSession


def create_test_config(
    tmp_path: Path, servers: dict[str, MCPServerConfig] | None = None
) -> Path:
    """Create a test configuration file.

    Args:
        tmp_path: Temporary directory path
        servers: Optional dict of MCP servers to include

    Returns:
        Path to the created config file
    """
    config_path = tmp_path / "test_config.json"

    if servers is None:
        servers = {
            "test-server": MCPServerConfig(
                command="echo",
                args=["test"],
                enabled=True,
            )
        }

    config = Config(mcpServers=servers)

    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f)

    return config_path


def assert_tool_in_results(
    results: list[dict[str, Any]], tool_name: str, server: str | None = None
):
    """Assert that a specific tool is in the search results.

    Args:
        results: Search results
        tool_name: Name of the tool to find
        server: Optional server name to match
    """
    found = False
    for result in results:
        if result["tool"] == tool_name and (
            server is None or result["server"] == server
        ):
            found = True
            break

    assert found, f"Tool '{tool_name}' not found in results"


def assert_tools_ordered(results: list[dict[str, Any]], expected_order: list[str]):
    """Assert that tools appear in the expected order.

    Args:
        results: Search results
        expected_order: List of tool names in expected order
    """
    actual_order = [r["tool"] for r in results[: len(expected_order)]]
    assert actual_order == expected_order, (
        f"Expected order {expected_order}, got {actual_order}"
    )
