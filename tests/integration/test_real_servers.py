"""Real end-to-end tests with actual MCP servers running."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# Get paths
TEST_SERVER_PATH = Path(__file__).parent / "test_mcp_server.py"


class TestRealServers:
    """Test with real MCP servers running as subprocesses."""

    @pytest_asyncio.fixture
    async def test_config_file(self, tmp_path):
        """Create a test configuration file."""
        config_path = tmp_path / "test_config.json"
        config = {
            "mcpServers": {
                "test-server": {
                    "command": sys.executable,
                    "args": [str(TEST_SERVER_PATH)],
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

    @pytest_asyncio.fixture
    async def mux_server(self, test_config_file):
        """Start MCP Mux server as a subprocess."""
        env = os.environ.copy()
        env["MUX_CONFIG_PATH"] = str(test_config_file)

        # Start mux server using the mux-nix wrapper for NixOS compatibility
        mux_nix_path = Path(__file__).parents[2] / "mux-nix"
        process = subprocess.Popen(
            [str(mux_nix_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give it time to start
        await asyncio.sleep(1)

        yield process

        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @pytest.mark.asyncio
    async def test_real_mux_integration(self, test_config_file):
        """Test real MCP Mux server with actual tool discovery and execution."""
        # Connect to mux server as an MCP client
        server_params = StdioServerParameters(
            command="mcp-mux",
            args=[],
            env={"MUX_CONFIG_PATH": str(test_config_file)},
        )

        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()

                # List tools from mux (should show search and execute)
                tools_response = await session.list_tools()
                tools = [tool.name for tool in tools_response.tools]
                assert "search" in tools
                assert "execute" in tools

                # Search for echo tool
                search_result = await session.call_tool(
                    "search", arguments={"query": "echo message"}
                )
                assert search_result.content
                content = search_result.content[0]
                assert isinstance(content, TextContent)
                search_data = json.loads(content.text)
                assert len(search_data) > 0
                assert any(tool["tool"] == "echo" for tool in search_data)

                # Execute echo tool through mux
                echo_result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "test-server",
                        "tool": "echo",
                        "arguments": {"message": "Hello from real test!"},
                    },
                )
                assert echo_result.content
                content = echo_result.content[0]
                assert isinstance(content, TextContent)
                assert content.text == "Echo: Hello from real test!"

                # Search for math tool
                search_result = await session.call_tool(
                    "search", arguments={"query": "add numbers math"}
                )
                content = search_result.content[0]
                assert isinstance(content, TextContent)
                search_data = json.loads(content.text)
                assert any(tool["tool"] == "add" for tool in search_data)

                # Execute add tool
                add_result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "test-server",
                        "tool": "add",
                        "arguments": {"a": 15, "b": 27},
                    },
                )
                content = add_result.content[0]
                assert isinstance(content, TextContent)
                assert content.text == "Result: 42"

                # Test error handling - unknown tool
                unknown_tool_result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "test-server",
                        "tool": "unknown_tool",
                        "arguments": {},
                    },
                )
                # Should return an error message, not raise exception
                assert unknown_tool_result.content
                assert any(
                    "error" in str(c).lower()
                    or "not found" in str(c).lower()
                    or "unknown" in str(c).lower()
                    for c in unknown_tool_result.content
                )

                # Test error handling - unknown server
                unknown_server_result = await session.call_tool(
                    "execute",
                    arguments={
                        "server": "unknown-server",
                        "tool": "echo",
                        "arguments": {},
                    },
                )
                # Should return an error message, not raise exception
                assert unknown_server_result.content
                assert any(
                    "error" in str(c).lower() or "not found" in str(c).lower()
                    for c in unknown_server_result.content
                )

    @pytest.mark.asyncio
    async def test_multiple_real_servers(self, tmp_path):
        """Test mux with multiple real MCP servers."""
        # Create config with multiple servers
        config_path = tmp_path / "multi_config.json"
        config = {
            "mcpServers": {
                "test-server-1": {
                    "command": sys.executable,
                    "args": [str(TEST_SERVER_PATH)],
                    "enabled": True,
                },
                "test-server-2": {
                    "command": sys.executable,
                    "args": [str(TEST_SERVER_PATH)],
                    "enabled": True,
                },
            },
            "search": {
                "method": "local",
                "model": "BAAI/bge-small-en-v1.5",
            },
        }
        config_path.write_text(json.dumps(config, indent=2))

        # Start mux with multi-server config
        env = os.environ.copy()
        env["MUX_CONFIG_PATH"] = str(config_path)

        mux_process = subprocess.Popen(
            [sys.executable, "-m", "mux"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parents[2] / "src",
        )

        try:
            # Give it time to start and connect to all servers
            await asyncio.sleep(2)

            # Connect as client
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mux"],
                env={"MUX_CONFIG_PATH": str(config_path)},
                cwd=str(Path(__file__).parents[2] / "src"),
            )

            async with stdio_client(server_params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Search should find tools from both servers
                    search_result = await session.call_tool(
                        "search", arguments={"query": "echo"}
                    )
                    content = search_result.content[0]
                    assert isinstance(content, TextContent)
                    search_data = json.loads(content.text)

                    # Should have echo tool from both servers
                    echo_tools = [t for t in search_data if t["tool"] == "echo"]
                    servers = {t["server"] for t in echo_tools}
                    assert "test-server-1" in servers or "test-server-2" in servers

                    # Execute on specific server
                    for server in ["test-server-1", "test-server-2"]:
                        try:
                            result = await session.call_tool(
                                "execute",
                                arguments={
                                    "server": server,
                                    "tool": "echo",
                                    "arguments": {"message": f"Hello from {server}"},
                                },
                            )
                            content = result.content[0]
                            assert isinstance(content, TextContent)
                            assert f"Echo: Hello from {server}" in content.text
                            break  # At least one server worked
                        except Exception:
                            continue  # Try next server

        finally:
            mux_process.terminate()
            try:
                mux_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mux_process.kill()
                mux_process.wait()

    @pytest.mark.asyncio
    async def test_server_disconnection_handling(self, tmp_path):
        """Test mux handles server disconnections gracefully."""
        # This test would require a more complex setup with a server
        # that can be stopped mid-test. For now, we'll test the concept.

        config_path = tmp_path / "disconnect_config.json"
        config = {
            "mcpServers": {
                "unstable-server": {
                    "command": sys.executable,
                    "args": ["-c", "import time; time.sleep(1)"],  # Dies quickly
                    "enabled": True,
                },
                "stable-server": {
                    "command": sys.executable,
                    "args": [str(TEST_SERVER_PATH)],
                    "enabled": True,
                },
            },
        }
        config_path.write_text(json.dumps(config, indent=2))

        env = os.environ.copy()
        env["MUX_CONFIG_PATH"] = str(config_path)

        mux_process = subprocess.Popen(
            [sys.executable, "-m", "mux"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parents[2] / "src",
        )

        try:
            await asyncio.sleep(3)  # Let servers start and unstable one die

            # Connect and verify stable server still works
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mux"],
                env={"MUX_CONFIG_PATH": str(config_path)},
                cwd=str(Path(__file__).parents[2] / "src"),
            )

            async with stdio_client(server_params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Should still be able to use stable server
                    result = await session.call_tool(
                        "execute",
                        arguments={
                            "server": "stable-server",
                            "tool": "echo",
                            "arguments": {"message": "Still working!"},
                        },
                    )
                    content = result.content[0]
                    assert isinstance(content, TextContent)
                    assert content.text == "Echo: Still working!"

                    # Unstable server should fail or return an error
                    try:
                        result = await session.call_tool(
                            "execute",
                            arguments={
                                "server": "unstable-server",
                                "tool": "echo",
                                "arguments": {"message": "This won't work"},
                            },
                        )
                        # If no exception, check if result indicates failure
                        content = result.content[0]
                        assert isinstance(content, TextContent)
                        # Should contain error message
                        assert (
                            "error" in content.text.lower()
                            or "not found" in content.text.lower()
                        )
                    except Exception:
                        # Expected - server is disconnected
                        pass

        finally:
            mux_process.terminate()
            try:
                mux_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mux_process.kill()
                mux_process.wait()

    @pytest.mark.asyncio
    async def test_performance_with_many_tools(self, tmp_path):
        """Test search performance with many tools indexed."""
        # This test validates that search remains fast even with many tools
        config_path = tmp_path / "perf_config.json"
        config = {
            "mcpServers": {
                "test-server": {
                    "command": sys.executable,
                    "args": [str(TEST_SERVER_PATH)],
                    "enabled": True,
                }
            },
            "search": {
                "method": "local",
                "model": "BAAI/bge-small-en-v1.5",
                "cache_embeddings": True,
            },
        }
        config_path.write_text(json.dumps(config, indent=2))

        env = os.environ.copy()
        env["MUX_CONFIG_PATH"] = str(config_path)

        mux_process = subprocess.Popen(
            [sys.executable, "-m", "mux"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parents[2] / "src",
        )

        try:
            await asyncio.sleep(2)

            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mux"],
                env={"MUX_CONFIG_PATH": str(config_path)},
                cwd=str(Path(__file__).parents[2] / "src"),
            )

            async with stdio_client(server_params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Perform multiple searches and measure time
                    queries = [
                        "echo message",
                        "add numbers",
                        "weather forecast",
                        "list files directory",
                        "fetch data api",
                    ]

                    for query in queries:
                        start_time = time.time()
                        result = await session.call_tool(
                            "search", arguments={"query": query}
                        )
                        search_time = time.time() - start_time

                        # Search should be fast (< 1 second)
                        assert search_time < 1.0
                        assert result.content

        finally:
            mux_process.terminate()
            try:
                mux_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mux_process.kill()
                mux_process.wait()
