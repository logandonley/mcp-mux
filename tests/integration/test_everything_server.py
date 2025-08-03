"""Integration tests using the MCP 'everything' test server."""

import asyncio
import json

import pytest
import pytest_asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ImageContent, TextContent


class TestEverythingServer:
    """Test mux with the comprehensive 'everything' MCP test server."""

    @pytest_asyncio.fixture
    async def everything_config_file(self, tmp_path):
        """Create config using the everything server."""
        config_path = tmp_path / "everything_config.json"
        config = {
            "mcpServers": {
                "everything": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-everything"],
                    "enabled": True,
                }
            },
            "search": {
                "method": "local",
                "model": "BAAI/bge-small-en-v1.5",
                "cache_embeddings": True,
                "max_results": 10,
            },
        }
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_everything_server_tools(self, everything_config_file):
        """Test mux can discover and use all tools from everything server."""
        server_params = StdioServerParameters(
            command="mcp-mux",
            args=[],
            env={"MUX_CONFIG_PATH": str(everything_config_file)},
        )

        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Test search finds echo tool
                result = await session.call_tool(
                    "search", arguments={"query": "echo message"}
                )
                content = result.content[0]
                assert isinstance(content, TextContent)
                tools = json.loads(content.text)
                assert any(t["tool"] == "echo" for t in tools)

                # Test echo tool execution
                result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "everything",
                        "tool": "echo",
                        "arguments": {"message": "Hello from everything server!"},
                    },
                )
                assert result.content
                assert "Hello from everything server!" in str(result.content[0])

                # Test add tool
                result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "everything",
                        "tool": "add",
                        "arguments": {"a": 100, "b": 23},
                    },
                )
                assert result.content
                assert "123" in str(result.content[0])

                # Test getTinyImage tool
                result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "everything",
                        "tool": "getTinyImage",
                        "arguments": {},
                    },
                )
                # Should return image content
                assert result.content
                content = result.content[0]
                # The everything server returns base64 encoded image data
                assert isinstance(content, TextContent | ImageContent)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_everything_server_search_quality(self, everything_config_file):
        """Test search quality with various queries."""
        server_params = StdioServerParameters(
            command="mcp-mux",
            args=[],
            env={"MUX_CONFIG_PATH": str(everything_config_file)},
        )

        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()

                test_cases = [
                    ("print environment variables", ["printEnv"]),
                    ("long running task with progress", ["longRunningOperation"]),
                    ("get image", ["getTinyImage"]),
                    ("sample from llm", ["sampleLLM"]),
                    ("structured content", ["structuredContent"]),
                ]

                for query, expected_tools in test_cases:
                    result = await session.call_tool(
                        "search", arguments={"query": query}
                    )
                    content = result.content[0]
                    assert isinstance(content, TextContent)
                    tools = json.loads(content.text)
                    found_tools = [t["tool"] for t in tools]

                    # Check that at least one expected tool is in top results
                    assert any(
                        expected in found_tools for expected in expected_tools
                    ), (
                        f"Query '{query}' did not find expected tools "
                        f"{expected_tools}, got {found_tools}"
                    )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_everything_server_error_handling(self, everything_config_file):
        """Test error handling with everything server."""
        server_params = StdioServerParameters(
            command="mcp-mux",
            args=[],
            env={"MUX_CONFIG_PATH": str(everything_config_file)},
        )

        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Test invalid tool name
                result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "everything",
                        "tool": "nonexistent_tool",
                        "arguments": {},
                    },
                )
                assert result.content
                error_text = str(result.content[0])
                assert (
                    "error" in error_text.lower() or "not found" in error_text.lower()
                )

                # Test invalid arguments
                result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "everything",
                        "tool": "add",
                        "arguments": {"x": 1, "y": 2},  # Wrong param names
                    },
                )
                # Should handle gracefully
                assert result.content

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_tool_execution(self, everything_config_file):
        """Test concurrent execution of multiple tools."""
        server_params = StdioServerParameters(
            command="mcp-mux",
            args=[],
            env={"MUX_CONFIG_PATH": str(everything_config_file)},
        )

        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Execute multiple tools concurrently
                tasks = [
                    session.call_tool(
                        "execute",
                        arguments={
                            "server": "everything",
                            "tool": "echo",
                            "arguments": {"message": f"Concurrent {i}"},
                        },
                    )
                    for i in range(5)
                ]

                results = await asyncio.gather(*tasks)

                # All should succeed
                assert len(results) == 5
                for i, result in enumerate(results):
                    assert result.content
                    assert f"Concurrent {i}" in str(result.content[0])
