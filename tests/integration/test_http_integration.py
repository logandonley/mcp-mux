"""Integration tests for HTTP/SSE MCP server connections."""
# pyright: reportCallIssue=false

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mux.client import ClientManager
from mux.config import Config, MCPServerConfig, SearchConfig
from mux.search import SearchEngine


class TestHTTPIntegration:
    """Test HTTP/SSE MCP server integration."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration with HTTP server."""
        return Config(
            mcpServers={
                "http-server": MCPServerConfig(
                    url="http://localhost:8080/mcp",
                    enabled=True,
                )
            },
            search=SearchConfig(
                method="local",
                model="BAAI/bge-small-en-v1.5",
                cache_embeddings=False,
            ),
        )

    @pytest.fixture
    def mock_http_client(self):
        """Mock streamablehttp_client for HTTP connections."""
        with patch("mux.client.streamablehttp_client") as mock:
            # Create mock context manager
            context = AsyncMock()

            # Mock session components
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_session_id = MagicMock(return_value="test-session-123")

            # Configure context manager
            context.__aenter__.return_value = (mock_read, mock_write, mock_session_id)
            context.__aexit__.return_value = None

            mock.return_value = context
            yield mock, mock_read, mock_write

    @pytest.fixture
    def mock_client_session(self):
        """Mock ClientSession for HTTP connections."""
        with patch("mux.client.ClientSession") as mock:
            session_context = AsyncMock()
            session_instance = AsyncMock()

            # Configure session behavior
            session_instance.initialize = AsyncMock()

            # Mock tool listing
            tools_response = MagicMock()
            tool1 = MagicMock()
            tool1.name = "fetch_data"
            tool1.description = "Fetch data from API"
            tool1.inputSchema = {
                "type": "object",
                "properties": {"endpoint": {"type": "string"}},
                "required": ["endpoint"],
            }

            tool2 = MagicMock()
            tool2.name = "post_data"
            tool2.description = "Post data to API"
            tool2.inputSchema = {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["endpoint", "data"],
            }

            tools_response.tools = [tool1, tool2]
            session_instance.list_tools.return_value = tools_response

            # Mock tool execution
            async def mock_call_tool(name, arguments):  # noqa: ARG001
                from mcp.types import TextContent

                result = MagicMock()

                if name == "fetch_data":
                    text = json.dumps({"status": "success", "data": {"value": 42}})
                elif name == "post_data":
                    text = json.dumps({"status": "created", "id": 123})
                else:
                    raise ValueError(f"Unknown tool: {name}")

                content = TextContent(type="text", text=text)
                result.content = [content]
                return result

            session_instance.call_tool = mock_call_tool

            # Configure context manager
            session_context.__aenter__.return_value = session_instance
            session_context.__aexit__.return_value = None

            mock.return_value = session_context
            yield mock, session_instance

    @pytest.mark.asyncio
    async def test_http_connection_and_discovery(
        self,
        test_config,
        mock_http_client,
        mock_client_session,  # noqa: ARG002
    ):
        """Test HTTP server connection and tool discovery."""
        manager = ClientManager(test_config)

        try:
            await manager.initialize()

            # Verify connection
            assert len(manager.clients) == 1
            assert "http-server" in manager.clients

            # Verify HTTP client was called correctly
            mock_http, _, _ = mock_http_client
            mock_http.assert_called_once_with("http://localhost:8080/mcp")

            # Get tools
            tools = await manager.get_all_tools()
            assert len(tools) == 2
            assert any(t["tool"] == "fetch_data" for t in tools)
            assert any(t["tool"] == "post_data" for t in tools)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_http_tool_execution(
        self,
        test_config,
        mock_http_client,  # noqa: ARG002
        mock_client_session,  # noqa: ARG002
    ):
        """Test executing tools over HTTP."""
        manager = ClientManager(test_config)

        try:
            await manager.initialize()

            # Execute fetch_data tool
            result = await manager.execute_tool(
                "http-server", "fetch_data", {"endpoint": "/api/test"}
            )
            data = json.loads(result)
            assert data["status"] == "success"
            assert data["data"]["value"] == 42

            # Execute post_data tool
            result = await manager.execute_tool(
                "http-server",
                "post_data",
                {"endpoint": "/api/items", "data": {"name": "Test Item"}},
            )
            data = json.loads(result)
            assert data["status"] == "created"
            assert data["id"] == 123

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_http_search_integration(
        self,
        test_config,
        mock_http_client,  # noqa: ARG002
        mock_client_session,  # noqa: ARG002
    ):
        """Test search functionality with HTTP servers."""
        manager = ClientManager(test_config)
        search_engine = SearchEngine(test_config.search)

        try:
            await manager.initialize()

            # Index tools
            tools = await manager.get_all_tools()
            await search_engine.index_tools(tools)

            # Search for fetch tool
            results = await search_engine.search("fetch api data")
            assert len(results) > 0
            assert any(r["tool"] == "fetch_data" for r in results)

            # Search for post tool
            results = await search_engine.search("post create data")
            assert any(r["tool"] == "post_data" for r in results)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_http_connection_failure(self, test_config):
        """Test handling of HTTP connection failures."""
        # Mock connection failure
        with patch("mux.client.streamablehttp_client") as mock_http:
            mock_http.side_effect = Exception("Connection refused")

            manager = ClientManager(test_config)
            await manager.initialize()

            # Should handle failure gracefully
            assert len(manager.clients) == 0

    @pytest.mark.asyncio
    async def test_http_timeout_handling(
        self,
        test_config,
        mock_http_client,  # noqa: ARG002
        mock_client_session,
    ):
        """Test timeout handling for HTTP operations."""
        manager = ClientManager(test_config)
        _, session_instance = mock_client_session

        # Make call_tool hang
        async def slow_call_tool(name, arguments):  # noqa: ARG001
            await asyncio.sleep(10)  # Simulate slow response

        session_instance.call_tool = slow_call_tool

        try:
            await manager.initialize()

            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    manager.execute_tool(
                        "http-server", "fetch_data", {"endpoint": "/slow"}
                    ),
                    timeout=1.0,
                )

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_http_concurrent_requests(
        self,
        test_config,
        mock_http_client,  # noqa: ARG002
        mock_client_session,  # noqa: ARG002
    ):
        """Test concurrent HTTP requests."""
        manager = ClientManager(test_config)

        try:
            await manager.initialize()

            # Execute multiple requests concurrently
            tasks = [
                manager.execute_tool(
                    "http-server", "fetch_data", {"endpoint": f"/api/item/{i}"}
                )
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 5
            for result in results:
                data = json.loads(result)
                assert data["status"] == "success"

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_http_error_responses(
        self,
        test_config,
        mock_http_client,  # noqa: ARG002
        mock_client_session,
    ):
        """Test handling of error responses from HTTP server."""
        manager = ClientManager(test_config)
        _, session_instance = mock_client_session

        # Mock error response
        async def mock_error_tool(name, arguments):
            if name == "error_tool":
                raise Exception("Server error: Invalid request")
            return await session_instance.call_tool(name, arguments)

        session_instance.call_tool = mock_error_tool

        try:
            await manager.initialize()

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await manager.execute_tool(
                    "http-server", "error_tool", {"bad": "request"}
                )
            assert "Server error" in str(exc_info.value)

        finally:
            await manager.shutdown()
