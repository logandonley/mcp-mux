"""End-to-end integration tests for the complete MCP Mux system."""
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mux.client import ClientManager
from mux.config import Config, MCPServerConfig, SearchConfig
from mux.search import SearchEngine
from mux.server import execute, initialize_services, search, shutdown_services

# Get path to test server
TEST_SERVER_PATH = Path(__file__).parent / "test_mcp_server.py"


class TestEndToEnd:
    """Test complete MCP Mux system with multiple servers."""

    @pytest.fixture
    def multi_server_config(self, tmp_path):  # noqa: ARG002
        """Create configuration with multiple servers."""
        return Config(
            mcpServers={
                "test-stdio": MCPServerConfig(
                    command=sys.executable,
                    args=[str(TEST_SERVER_PATH)],
                    enabled=True,
                ),
                "test-http": MCPServerConfig(
                    url="http://localhost:8081/mcp",
                    enabled=True,
                ),
                "disabled-server": MCPServerConfig(
                    command="disabled",
                    enabled=False,  # Should be ignored
                ),
            },
            search=SearchConfig(
                method="local",
                model="BAAI/bge-small-en-v1.5",
                cache_embeddings=True,
                max_results=10,
            ),
        )

    @pytest.fixture
    def mock_http_setup(self):
        """Set up mocks for HTTP server."""
        with (
            patch("mux.client.streamablehttp_client") as mock_http,
            patch("mux.client.ClientSession") as mock_session,
            patch("mux.client.stdio_client") as mock_stdio,
        ):
            # Configure HTTP client mock
            http_context = AsyncMock()
            http_context.__aenter__.return_value = (
                AsyncMock(),
                AsyncMock(),
                MagicMock(return_value="session-123"),
            )
            http_context.__aexit__.return_value = None
            mock_http.return_value = http_context

            # Configure session mock
            session_context = AsyncMock()
            session_instance = AsyncMock()

            # Mock HTTP server tools
            tools_response = MagicMock()
            http_tool = MagicMock()
            http_tool.name = "fetch_api"
            http_tool.description = "Fetch data from API endpoint"
            http_tool.inputSchema = {"type": "object", "properties": {}}
            tools_response.tools = [http_tool]

            session_instance.initialize = AsyncMock()
            session_instance.list_tools.return_value = tools_response

            # Mock tool execution
            async def mock_call_tool(name, arguments):  # noqa: ARG001
                from mcp.types import TextContent

                result = MagicMock()
                content = TextContent(type="text", text=f"HTTP Result: {name}")
                result.content = [content]
                return result

            session_instance.call_tool = mock_call_tool

            session_context.__aenter__.return_value = session_instance
            session_context.__aexit__.return_value = None
            mock_session.return_value = session_context

            # Configure stdio client mock
            stdio_context = AsyncMock()
            stdio_context.__aenter__.return_value = (AsyncMock(), AsyncMock())
            stdio_context.__aexit__.return_value = None

            # For stdio connections, return a different mock session
            stdio_session_context = AsyncMock()
            stdio_session_instance = AsyncMock()

            # Mock stdio server tools
            stdio_tools_response = MagicMock()
            tools = []
            for tool_name, desc in [
                ("echo", "Echo a message"),
                ("add", "Add two numbers"),
                ("get_weather", "Get weather"),
                ("list_files", "List files"),
            ]:
                tool = MagicMock()
                tool.name = tool_name
                tool.description = desc
                tool.inputSchema = {"type": "object", "properties": {}}
                tools.append(tool)
            stdio_tools_response.tools = tools

            stdio_session_instance.initialize = AsyncMock()
            stdio_session_instance.list_tools.return_value = stdio_tools_response

            # Mock stdio tool execution
            async def mock_stdio_call_tool(name, arguments):
                from mcp.types import TextContent

                result = MagicMock()
                if name == "echo":
                    text = f"Echo: {arguments.get('message', '')}"
                elif name == "add":
                    a = arguments.get("a", 0)
                    b = arguments.get("b", 0)
                    text = f"Result: {a + b}"
                elif name == "get_weather":
                    location = arguments.get("location", "Unknown")
                    text = f"Weather in {location}: Sunny, 72°F"
                else:
                    text = f"Result from {name}"
                content = TextContent(type="text", text=text)
                result.content = [content]
                return result

            stdio_session_instance.call_tool = mock_stdio_call_tool

            stdio_session_context.__aenter__.return_value = stdio_session_instance
            stdio_session_context.__aexit__.return_value = None

            # Return different sessions based on context
            call_count = 0

            def session_side_effect(*args, **kwargs):  # noqa: ARG001
                nonlocal call_count
                # Check if this is for HTTP or stdio based on the context
                # For simplicity, we'll alternate or use some heuristic
                call_count += 1

                # First call is stdio, second is HTTP
                if call_count % 2 == 1:
                    return stdio_session_context
                return session_context

            mock_session.side_effect = session_side_effect
            mock_stdio.return_value = stdio_context

            yield mock_http, mock_session

    @pytest.mark.asyncio
    async def test_multi_server_initialization(
        self,
        multi_server_config,
        mock_http_setup,  # noqa: ARG002
    ):
        """Test initialization with multiple servers."""
        manager = ClientManager(multi_server_config)
        search_engine = SearchEngine(multi_server_config.search)

        try:
            await manager.initialize()

            # Should have 2 active clients (disabled one ignored)
            assert len(manager.clients) == 2
            assert "test-stdio" in manager.clients
            assert "test-http" in manager.clients
            assert "disabled-server" not in manager.clients

            # Get all tools from both servers
            tools = await manager.get_all_tools()

            # Should have tools from both servers
            stdio_tools = [t for t in tools if t["server"] == "test-stdio"]
            http_tools = [t for t in tools if t["server"] == "test-http"]

            assert len(stdio_tools) == 4  # echo, add, get_weather, list_files
            assert len(http_tools) == 1  # fetch_api

            # Index all tools
            await search_engine.index_tools(tools)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_cross_server_search(self, multi_server_config, mock_http_setup):  # noqa: ARG002
        """Test searching across multiple servers."""
        manager = ClientManager(multi_server_config)
        search_engine = SearchEngine(multi_server_config.search)

        try:
            await manager.initialize()
            tools = await manager.get_all_tools()
            await search_engine.index_tools(tools)

            # Search for "fetch" should find both fetch_api and possibly get_weather
            results = await search_engine.search("fetch data")
            assert len(results) > 0

            # Should prioritize exact match
            fetch_results = [r for r in results if "fetch" in r["tool"]]
            assert len(fetch_results) > 0
            assert fetch_results[0]["tool"] == "fetch_api"

            # Search for "add" should find add tool from stdio server
            results = await search_engine.search("add numbers")
            assert any(
                r["tool"] == "add" and r["server"] == "test-stdio" for r in results
            )

            # Search for generic "list" should find list_files
            results = await search_engine.search("list directory")
            assert any(r["tool"] == "list_files" for r in results)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_cross_server_execution(self, multi_server_config, mock_http_setup):  # noqa: ARG002
        """Test executing tools on different servers."""
        manager = ClientManager(multi_server_config)

        try:
            await manager.initialize()

            # Execute tool on stdio server
            result = await manager.execute_tool(
                "test-stdio", "echo", {"message": "Hello from stdio"}
            )
            assert result == "Echo: Hello from stdio"

            # Execute tool on HTTP server
            result = await manager.execute_tool("test-http", "fetch_api", {})
            assert result == "HTTP Result: fetch_api"

            # Execute another tool on stdio server
            result = await manager.execute_tool("test-stdio", "add", {"a": 10, "b": 20})
            assert result == "Result: 30"

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_server_level_operations(self, multi_server_config, mock_http_setup):  # noqa: ARG002
        """Test server-level operations like those in mux.server."""
        # Mock the globals in mux.server
        import mux.server

        # Save originals
        orig_manager = mux.server.client_manager
        orig_engine = mux.server.search_engine
        orig_config = getattr(mux.server, "config", None)

        try:
            # Initialize services
            with patch("mux.server.load_config") as mock_load_config:
                mock_load_config.return_value = multi_server_config
                await initialize_services()

            # Test search tool
            search_results = await search("echo message")
            results = json.loads(search_results)
            assert len(results) > 0
            assert any(r["tool"] == "echo" for r in results)

            # Test execute tool
            exec_result = await execute(
                "test-stdio", "echo", {"message": "Server test"}
            )
            assert exec_result == "Echo: Server test"

            # Test searching across servers
            search_results = await search("fetch api data")
            results = json.loads(search_results)
            assert any(r["tool"] == "fetch_api" for r in results)

            # Execute on HTTP server through server interface
            exec_result = await execute("test-http", "fetch_api", {})
            assert exec_result == "HTTP Result: fetch_api"

        finally:
            # Shutdown services
            await shutdown_services()

            # Restore originals
            mux.server.client_manager = orig_manager
            mux.server.search_engine = orig_engine
            if orig_config is not None:
                mux.server.config = orig_config

    @pytest.mark.asyncio
    async def test_partial_server_failure(self, multi_server_config, mock_http_setup):
        """Test system continues working when some servers fail."""
        # Make HTTP server fail to connect
        mock_http, _ = mock_http_setup
        mock_http.side_effect = Exception("HTTP connection failed")

        manager = ClientManager(multi_server_config)

        try:
            await manager.initialize()

            # Should have only stdio server
            assert len(manager.clients) == 1
            assert "test-stdio" in manager.clients
            assert "test-http" not in manager.clients

            # System should still work with remaining servers
            result = await manager.execute_tool(
                "test-stdio", "echo", {"message": "Still working"}
            )
            assert result == "Echo: Still working"

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_multi_server_operations(
        self,
        multi_server_config,
        mock_http_setup,  # noqa: ARG002
    ):
        """Test concurrent operations across multiple servers."""
        manager = ClientManager(multi_server_config)

        try:
            await manager.initialize()

            # Create mixed operations across servers
            tasks = [
                manager.execute_tool("test-stdio", "echo", {"message": "Test 1"}),
                manager.execute_tool("test-http", "fetch_api", {}),
                manager.execute_tool("test-stdio", "add", {"a": 1, "b": 2}),
                manager.execute_tool("test-stdio", "get_weather", {"location": "NYC"}),
                manager.execute_tool("test-http", "fetch_api", {}),
            ]

            # Execute concurrently
            results = await asyncio.gather(*tasks)

            # Verify results
            assert results[0] == "Echo: Test 1"
            assert results[1] == "HTTP Result: fetch_api"
            assert results[2] == "Result: 3"
            assert "Weather in NYC" in results[3]
            assert results[4] == "HTTP Result: fetch_api"

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_multi_server(
        self,
        multi_server_config,
        mock_http_setup,  # noqa: ARG002
    ):
        """Test graceful shutdown with multiple servers."""
        manager = ClientManager(multi_server_config)

        await manager.initialize()
        assert len(manager.clients) == 2

        # Get client references
        clients = list(manager.clients.values())
        for client in clients:
            assert client.session is not None

        # Shutdown
        await manager.shutdown()

        # Verify all cleaned up
        assert len(manager.clients) == 0
        for client in clients:
            assert client.session is None
