"""Integration tests for stdio MCP server connections."""
# pyright: reportCallIssue=false

import asyncio
import sys
from pathlib import Path

import pytest

from mux.client import ClientManager
from mux.config import Config, MCPServerConfig, SearchConfig
from mux.search import SearchEngine

# Get path to test server
TEST_SERVER_PATH = Path(__file__).parent / "test_mcp_server.py"


class TestStdioIntegration:
    """Test real stdio MCP server integration."""

    @pytest.fixture
    def test_config(self, tmp_path):  # noqa: ARG002
        """Create test configuration with stdio server."""
        return Config(
            mcpServers={
                "test-server": MCPServerConfig(
                    command=sys.executable,
                    args=[str(TEST_SERVER_PATH)],
                    enabled=True,
                )
            },
            search=SearchConfig(
                method="local",
                model="BAAI/bge-small-en-v1.5",
                cache_embeddings=False,
            ),
        )

    @pytest.mark.asyncio
    async def test_full_stdio_flow(self, test_config):
        """Test complete flow: connect → discover → search → execute."""
        # Initialize client manager
        manager = ClientManager(test_config)
        search_engine = SearchEngine(test_config.search)

        try:
            # Connect to server
            await manager.initialize()
            assert len(manager.clients) == 1
            assert "test-server" in manager.clients

            # Get all tools
            tools = await manager.get_all_tools()
            assert len(tools) == 4  # echo, add, get_weather, list_files

            # Index tools for search
            await search_engine.index_tools(tools)

            # Search for echo tool
            results = await search_engine.search("echo message")
            assert len(results) > 0
            assert results[0]["tool"] == "echo"

            # Execute echo tool
            result = await manager.execute_tool(
                "test-server", "echo", {"message": "Hello, MCP!"}
            )
            assert result == "Echo: Hello, MCP!"

            # Search for math tool
            results = await search_engine.search("add numbers")
            assert any(r["tool"] == "add" for r in results)

            # Execute add tool
            result = await manager.execute_tool("test-server", "add", {"a": 5, "b": 3})
            assert result == "Result: 8"

            # Search for weather tool
            results = await search_engine.search("weather forecast")
            assert any(r["tool"] == "get_weather" for r in results)

            # Execute weather tool
            result = await manager.execute_tool(
                "test-server", "get_weather", {"location": "San Francisco"}
            )
            assert "Weather in San Francisco" in result
            assert "Sunny" in result

        finally:
            # Clean up
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling(self, test_config):
        """Test error handling with stdio servers."""
        manager = ClientManager(test_config)

        try:
            await manager.initialize()

            # Try to execute unknown tool - our test server still executes it
            # but returns an error in the result
            result = await manager.execute_tool(
                "test-server", "unknown_tool", {"arg": "value"}
            )
            # The test server raises ValueError for unknown tools
            assert "Unknown tool" in str(result) or result is None

            # Try to execute on unknown server
            from mux.exceptions import ConnectionError as MuxConnectionError

            with pytest.raises(MuxConnectionError) as exc_info:
                await manager.execute_tool(
                    "unknown-server", "echo", {"message": "test"}
                )
            assert "not found" in str(exc_info.value)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_connection_failure(self, tmp_path):  # noqa: ARG002
        """Test handling of connection failures."""
        # Create config with non-existent command
        bad_config = Config(
            mcpServers={
                "bad-server": MCPServerConfig(
                    command="/non/existent/command",
                    args=["--test"],
                    enabled=True,
                )
            }
        )

        manager = ClientManager(bad_config)

        # Should handle connection failure gracefully
        await manager.initialize()
        assert len(manager.clients) == 0  # No successful connections

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, test_config):
        """Test concurrent tool executions."""
        manager = ClientManager(test_config)

        try:
            await manager.initialize()

            # Execute multiple tools concurrently
            tasks = [
                manager.execute_tool("test-server", "echo", {"message": f"Test {i}"})
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # Verify all results
            for i, result in enumerate(results):
                assert result == f"Echo: Test {i}"

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, test_config):
        """Test proper cleanup on shutdown."""
        manager = ClientManager(test_config)

        await manager.initialize()
        assert len(manager.clients) == 1

        # Get client reference
        client = manager.clients["test-server"]
        assert client.session is not None

        # Shutdown
        await manager.shutdown()

        # Verify cleanup
        assert len(manager.clients) == 0
        assert client.session is None

    @pytest.mark.asyncio
    async def test_tool_discovery_updates(self, test_config):
        """Test that tool discovery reflects server state."""
        manager = ClientManager(test_config)
        search_engine = SearchEngine(test_config.search)

        try:
            await manager.initialize()

            # Initial tool discovery
            tools1 = await manager.get_all_tools()
            await search_engine.index_tools(tools1)

            # Search should find all tools
            echo_results = await search_engine.search("echo")
            assert any(r["tool"] == "echo" for r in echo_results)

            weather_results = await search_engine.search("weather")
            assert any(r["tool"] == "get_weather" for r in weather_results)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_long_running_operation(self, test_config):
        """Test handling of operations with timeouts."""
        manager = ClientManager(test_config)

        try:
            await manager.initialize()

            # Execute a normal operation (should complete quickly)
            result = await asyncio.wait_for(
                manager.execute_tool("test-server", "echo", {"message": "Quick test"}),
                timeout=5.0,
            )
            assert "Quick test" in result

        finally:
            await manager.shutdown()
