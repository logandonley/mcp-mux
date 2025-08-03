"""MCP client connection management."""

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from .config import Config, MCPServerConfig
from .exceptions import ConnectionError

logger = logging.getLogger("mux.client")

# Client configuration constants
CLIENT_SHUTDOWN_TIMEOUT = 2.0


def _extract_text_content(result: Any) -> Any:
    """Extract text content from a tool call result."""
    if not result.content or len(result.content) == 0:
        return None

    content = result.content[0]
    if isinstance(content, TextContent):
        return content.text
    return content


class MCPClient:
    """Wrapper for an MCP client connection."""

    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self.session: ClientSession | None = None
        self._transport_context = None
        self._session_context = None
        self._read_stream = None
        self._write_stream = None
        self.tools: list[dict[str, Any]] = []

    async def _connect_stdio_server(self) -> None:
        """Connect to a stdio-based MCP server."""
        if not self.config.command:
            raise ValueError("No command configured for stdio server")
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        command_string = f"{self.config.command} {' '.join(self.config.args)}"
        logger.debug(f"Server command: {command_string}")
        self._transport_context = stdio_client(server_params)
        (
            self._read_stream,
            self._write_stream,
        ) = await self._transport_context.__aenter__()

    async def _connect_http_server(self) -> None:
        """Connect to an HTTP-based MCP server."""
        if not self.config.url:
            raise ValueError("No URL configured for HTTP server")
        logger.debug(f"Server URL: {self.config.url}")
        self._transport_context = streamablehttp_client(self.config.url)
        read_stream, write_stream, _ = await self._transport_context.__aenter__()
        self._read_stream = read_stream
        self._write_stream = write_stream

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self.session is not None:
            return

        logger.info(f"Connecting to MCP server '{self.name}'...")

        try:
            # Establish transport connection
            if self.config.command:
                await self._connect_stdio_server()
            elif self.config.url:
                await self._connect_http_server()
            else:
                raise ValueError("No command or URL configured")

            # Create and initialize session
            if not self._read_stream or not self._write_stream:
                raise ConnectionError("Transport streams not established")
            self._session_context = ClientSession(self._read_stream, self._write_stream)
            self.session = await self._session_context.__aenter__()
            await self.session.initialize()

            logger.info(f"Successfully connected to '{self.name}'")
            await self._discover_tools()
        except TimeoutError as e:
            logger.error(f"Connection timeout for '{self.name}': {e}")
            await self.disconnect()
            raise ConnectionError(f"Connection timeout for '{self.name}': {e}") from e
        except OSError as e:
            logger.error(f"Network error connecting to '{self.name}': {e}")
            await self.disconnect()
            raise ConnectionError(
                f"Network error connecting to '{self.name}': {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to connect to '{self.name}': {e}")
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to '{self.name}': {e}") from e

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        if self.session is None:
            return

        tools_response = await self.session.list_tools()
        self.tools = [
            {
                "server": self.name,
                "tool": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response.tools
        ]
        logger.info(f"Discovered {len(self.tools)} tools from '{self.name}'")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool and return the result."""
        if self.session is None:
            raise ConnectionError(f"Client {self.name} is not connected")

        result = await self.session.call_tool(tool_name, arguments=arguments)
        return _extract_text_content(result)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        logger.debug(f"Disconnecting from '{self.name}'...")

        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except asyncio.CancelledError:
                pass
            except OSError as e:
                logger.debug(f"Error closing session for '{self.name}': {e}")
            finally:
                self._session_context = None
                self.session = None

        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except asyncio.CancelledError:
                pass
            except OSError as e:
                logger.debug(f"Error closing transport for '{self.name}': {e}")
            finally:
                self._transport_context = None
                self._read_stream = None
                self._write_stream = None


class ClientManager:
    """Manages multiple MCP client connections."""

    def __init__(self, config: Config):
        self.config = config
        self.clients: dict[str, MCPClient] = {}

    async def initialize(self) -> None:
        """Initialize all configured MCP clients."""
        tasks = []
        client_names = []

        for name, server_config in self.config.mcpServers.items():
            if not server_config.enabled:
                continue

            client = MCPClient(name, server_config)
            self.clients[name] = client
            tasks.append(self._connect_client(client))
            client_names.append(name)

        # Connect to all servers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_clients = []
        for name, result in zip(client_names, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(f"Failed to connect to {name}: {result}")
                failed_clients.append(name)

        # Remove failed clients after iteration
        for name in failed_clients:
            del self.clients[name]

    async def _connect_client(self, client: MCPClient) -> None:
        """Connect a single client with error handling."""
        try:
            await client.connect()
        except ConnectionError:
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {client.name}: {e}") from e

    async def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools from all connected clients."""
        all_tools = []
        for client in self.clients.values():
            all_tools.extend(client.tools)
        return all_tools

    async def execute_tool(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> Any:
        """Execute a tool on a specific server."""
        if server not in self.clients:
            raise ConnectionError(f"Server '{server}' not found or not connected")

        client = self.clients[server]
        return await client.execute_tool(tool, arguments)

    async def shutdown(self) -> None:
        """Shutdown all client connections gracefully."""
        if not self.clients:
            return

        logger.info(f"Shutting down {len(self.clients)} client connections...")

        # Disconnect all clients concurrently
        tasks = [client.disconnect() for client in self.clients.values()]

        # Wait for all disconnections with a shorter timeout
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=CLIENT_SHUTDOWN_TIMEOUT,
                )
            except TimeoutError:
                logger.debug("Some clients did not disconnect within timeout")
            except asyncio.CancelledError:
                pass
            except OSError as e:
                logger.debug(f"Error during shutdown: {e}")

        self.clients.clear()
        logger.info("All clients disconnected")
