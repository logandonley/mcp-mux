"""Unit tests for MCP client management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mux.client import ClientManager, MCPClient
from mux.config import Config, MCPServerConfig


class TestMCPClient:
    """Test MCPClient class."""

    @pytest.fixture
    def stdio_config(self):
        """Stdio server configuration."""
        return MCPServerConfig(
            command="test-server",
            args=["--test"],
            env={"TEST": "value"},
            url=None,
            enabled=True,
        )

    @pytest.fixture
    def http_config(self):
        """HTTP server configuration."""
        return MCPServerConfig(command=None, url="http://localhost:8080", enabled=True)

    @pytest.fixture
    def mock_stdio_client(self):
        """Mock stdio_client context manager."""
        with patch("mux.client.stdio_client") as mock:
            context = AsyncMock()
            context.__aenter__.return_value = (AsyncMock(), AsyncMock())
            context.__aexit__.return_value = None
            mock.return_value = context
            yield mock

    @pytest.fixture
    def mock_http_client(self):
        """Mock streamablehttp_client context manager."""
        with patch("mux.client.streamablehttp_client") as mock:
            context = AsyncMock()
            context.__aenter__.return_value = (
                AsyncMock(),
                AsyncMock(),
                Mock(return_value="session-id"),
            )
            context.__aexit__.return_value = None
            mock.return_value = context
            yield mock

    @pytest.fixture
    def mock_client_session(self):
        """Mock ClientSession class."""
        with patch("mux.client.ClientSession") as mock:
            session_context = AsyncMock()
            session_instance = AsyncMock()
            session_context.__aenter__.return_value = session_instance
            session_context.__aexit__.return_value = None
            mock.return_value = session_context
            yield mock, session_instance

    def test_initialization_stdio(self, stdio_config):
        """Test MCPClient initialization with stdio config."""
        client = MCPClient("test-client", stdio_config)
        assert client.name == "test-client"
        assert client.config == stdio_config
        assert client.session is None
        assert client._transport_context is None
        assert client._session_context is None
        assert client.tools == []

    def test_initialization_http(self, http_config):
        """Test MCPClient initialization with HTTP config."""
        client = MCPClient("http-client", http_config)
        assert client.name == "http-client"
        assert client.config == http_config
        assert client.session is None

    @pytest.mark.asyncio
    async def test_connect_stdio(
        self,
        stdio_config,
        mock_stdio_client,  # noqa: ARG002
        mock_client_session,
    ):
        """Test connecting to stdio server."""
        client = MCPClient("test-client", stdio_config)
        _, session_instance = mock_client_session

        # Mock tool discovery
        tools_response = MagicMock()
        tool1 = MagicMock()
        tool1.name = "echo"
        tool1.description = "Echo tool"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "add"
        tool2.description = "Add tool"
        tool2.inputSchema = {}

        tools_response.tools = [tool1, tool2]
        session_instance.list_tools.return_value = tools_response
        session_instance.initialize = AsyncMock()

        await client.connect()

        # Verify connection
        assert client.session is not None
        assert client._transport_context is not None
        assert client._session_context is not None

        # Verify initialization
        session_instance.initialize.assert_called_once()

        # Verify tools discovered
        assert len(client.tools) == 2
        assert client.tools[0]["tool"] == "echo"
        assert client.tools[0]["server"] == "test-client"
        assert client.tools[1]["tool"] == "add"

    @pytest.mark.asyncio
    async def test_connect_http(
        self, http_config, mock_http_client, mock_client_session
    ):
        """Test connecting to HTTP server."""
        client = MCPClient("http-client", http_config)
        _, session_instance = mock_client_session

        # Mock tool discovery
        tools_response = MagicMock()
        tools_response.tools = []
        session_instance.list_tools.return_value = tools_response
        session_instance.initialize = AsyncMock()

        await client.connect()

        # Verify connection
        assert client.session is not None
        mock_http_client.assert_called_once_with("http://localhost:8080")

    @pytest.mark.asyncio
    async def test_connect_already_connected(
        self,
        stdio_config,
        mock_stdio_client,
        mock_client_session,  # noqa: ARG002
    ):
        """Test connecting when already connected does nothing."""
        client = MCPClient("test-client", stdio_config)
        client.session = MagicMock()  # Simulate already connected

        await client.connect()

        # Should not attempt new connection
        mock_stdio_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_failure(self, stdio_config, mock_stdio_client):
        """Test connection failure handling."""
        client = MCPClient("test-client", stdio_config)
        mock_stdio_client.side_effect = Exception("Connection failed")

        from mux.exceptions import ConnectionError as MuxConnectionError

        with pytest.raises(MuxConnectionError) as exc_info:
            await client.connect()
        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_tool(self, stdio_config, mock_client_session):
        """Test executing a tool."""
        client = MCPClient("test-client", stdio_config)
        _, session_instance = mock_client_session
        client.session = session_instance

        # Mock tool execution result
        from mcp.types import TextContent

        result = MagicMock()
        content = TextContent(type="text", text="Hello, World!")
        result.content = [content]
        session_instance.call_tool.return_value = result

        response = await client.execute_tool("echo", {"message": "Hello"})

        assert response == "Hello, World!"
        session_instance.call_tool.assert_called_once_with(
            "echo", arguments={"message": "Hello"}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_no_content(self, stdio_config, mock_client_session):
        """Test executing tool with no content."""
        client = MCPClient("test-client", stdio_config)
        _, session_instance = mock_client_session
        client.session = session_instance

        # Mock empty result
        result = MagicMock()
        result.content = []
        session_instance.call_tool.return_value = result

        response = await client.execute_tool("test", {})
        assert response is None

    @pytest.mark.asyncio
    async def test_execute_tool_not_connected(self, stdio_config):
        """Test executing tool when not connected."""
        client = MCPClient("test-client", stdio_config)

        from mux.exceptions import ConnectionError as MuxConnectionError

        with pytest.raises(MuxConnectionError) as exc_info:
            await client.execute_tool("test", {})
        assert "Client test-client is not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_disconnect(
        self, stdio_config, mock_stdio_client, mock_client_session
    ):
        """Test disconnecting from server."""
        client = MCPClient("test-client", stdio_config)

        # Set up mock connections
        transport_context = mock_stdio_client.return_value
        session_context, _ = mock_client_session

        client._transport_context = transport_context
        client._session_context = session_context
        client.session = MagicMock()

        await client.disconnect()

        # Verify cleanup
        assert client.session is None
        assert client._session_context is None
        assert client._transport_context is None

        # Verify contexts were exited
        session_context.__aexit__.assert_called_once()
        transport_context.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_with_errors(
        self, stdio_config, mock_stdio_client, mock_client_session
    ):
        """Test disconnect handles errors gracefully."""
        client = MCPClient("test-client", stdio_config)

        # Set up mock connections
        transport_context = mock_stdio_client.return_value
        session_context, _ = mock_client_session

        # Make __aexit__ raise exceptions
        session_context.__aexit__.side_effect = OSError("Session error")
        transport_context.__aexit__.side_effect = OSError("Transport error")

        client._transport_context = transport_context
        client._session_context = session_context
        client.session = MagicMock()

        # Should not raise
        await client.disconnect()

        # Verify cleanup still happened
        assert client.session is None
        assert client._session_context is None
        assert client._transport_context is None


class TestClientManager:
    """Test ClientManager class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration with multiple servers."""
        return Config(
            mcpServers={
                "server1": MCPServerConfig(command="server1", url=None, enabled=True),
                "server2": MCPServerConfig(
                    command=None, url="http://server2", enabled=True
                ),
                "server3": MCPServerConfig(command="server3", url=None, enabled=False),
            }
        )

    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCPClient class."""
        with patch("mux.client.MCPClient") as mock:
            # Create mock instances for each server
            instances = {}

            def create_instance(name, config):
                instance = AsyncMock()
                instance.name = name
                instance.config = config
                instance.tools = [
                    {
                        "server": name,
                        "tool": f"tool_{name}",
                        "description": f"Tool for {name}",
                    }
                ]
                instance.connect = AsyncMock()
                instance.disconnect = AsyncMock()
                instance.execute_tool = AsyncMock(return_value=f"Result from {name}")
                instances[name] = instance
                return instance

            mock.side_effect = create_instance
            mock.instances = instances
            yield mock

    def test_initialization(self, sample_config):
        """Test ClientManager initialization."""
        manager = ClientManager(sample_config)
        assert manager.config == sample_config
        assert manager.clients == {}

    @pytest.mark.asyncio
    async def test_initialize(self, sample_config, mock_mcp_client):  # noqa: ARG002
        """Test initializing all clients."""
        manager = ClientManager(sample_config)
        await manager.initialize()

        # Should create clients for enabled servers only
        assert len(manager.clients) == 2
        assert "server1" in manager.clients
        assert "server2" in manager.clients
        assert "server3" not in manager.clients  # Disabled

        # Verify connections were attempted
        for client in manager.clients.values():
            client.connect.assert_called_once()  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_initialize_with_failures(self, sample_config, mock_mcp_client):
        """Test initialization handles connection failures."""
        manager = ClientManager(sample_config)

        # Initialize will create clients, we need to set up the mock before that
        # Store original side_effect
        original_side_effect = mock_mcp_client.side_effect

        def create_failing_instance(name, config):
            instance = original_side_effect(name, config)
            if name == "server1":
                instance.connect.side_effect = Exception("Connection failed")
            return instance

        mock_mcp_client.side_effect = create_failing_instance

        await manager.initialize()

        # Should only have server2
        assert len(manager.clients) == 1
        assert "server2" in manager.clients
        assert "server1" not in manager.clients

    @pytest.mark.asyncio
    async def test_get_all_tools(self, sample_config, mock_mcp_client):  # noqa: ARG002
        """Test getting all tools from all clients."""
        manager = ClientManager(sample_config)
        await manager.initialize()

        tools = await manager.get_all_tools()

        assert len(tools) == 2
        tool_names = [t["tool"] for t in tools]
        assert "tool_server1" in tool_names
        assert "tool_server2" in tool_names

    @pytest.mark.asyncio
    async def test_execute_tool(self, sample_config, mock_mcp_client):  # noqa: ARG002
        """Test executing a tool on a specific server."""
        manager = ClientManager(sample_config)
        await manager.initialize()

        result = await manager.execute_tool("server1", "echo", {"message": "test"})

        assert result == "Result from server1"
        manager.clients["server1"].execute_tool.assert_called_once_with(  # type: ignore[attr-defined]
            "echo", {"message": "test"}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_server_not_found(self, sample_config, mock_mcp_client):  # noqa: ARG002
        """Test executing tool on non-existent server."""
        manager = ClientManager(sample_config)
        await manager.initialize()

        from mux.exceptions import ConnectionError as MuxConnectionError

        with pytest.raises(MuxConnectionError) as exc_info:
            await manager.execute_tool("unknown", "tool", {})
        assert "Server 'unknown' not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_shutdown(self, sample_config, mock_mcp_client):  # noqa: ARG002
        """Test shutting down all clients."""
        manager = ClientManager(sample_config)
        await manager.initialize()

        # Store client references
        clients = list(manager.clients.values())

        await manager.shutdown()

        # Verify all clients disconnected
        for client in clients:
            client.disconnect.assert_called_once()  # type: ignore[attr-defined]

        # Verify clients cleared
        assert len(manager.clients) == 0

    @pytest.mark.asyncio
    async def test_shutdown_empty(self, sample_config):
        """Test shutdown with no clients."""
        manager = ClientManager(sample_config)

        # Should not raise
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_with_timeout(self, sample_config, mock_mcp_client):  # noqa: ARG002
        """Test shutdown handles timeouts."""
        manager = ClientManager(sample_config)
        await manager.initialize()

        # Make disconnect hang
        async def slow_disconnect():
            await asyncio.sleep(5)  # Longer than timeout

        for client in manager.clients.values():
            client.disconnect.side_effect = slow_disconnect  # type: ignore[attr-defined]

        # Should complete within timeout
        await manager.shutdown()

        # Clients should still be cleared
        assert len(manager.clients) == 0
