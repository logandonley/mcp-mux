"""Unit tests for MCP server implementation."""
# pyright: reportAttributeAccessIssue=false

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mux.server import (
    execute,
    initialize_services,
    search,
    shutdown_services,
)


class TestServerTools:
    """Test server tool functions."""

    @pytest.mark.asyncio
    async def test_search_tool_no_engine(self):
        """Test search tool when engine not initialized."""
        # Temporarily set search_engine to None
        import mux.server

        original = mux.server.search_engine
        mux.server.search_engine = None

        try:
            from mux.exceptions import SearchError

            with pytest.raises(SearchError) as exc_info:
                await search("test query")
            assert "Search engine not initialized" in str(exc_info.value)
        finally:
            mux.server.search_engine = original

    @pytest.mark.asyncio
    async def test_search_tool_empty_results(self):
        """Test search tool with no results."""
        import mux.server

        # Mock search engine
        mock_engine = AsyncMock()
        mock_engine.search.return_value = []

        original = mux.server.search_engine
        mux.server.search_engine = mock_engine

        try:
            result = await search("no matches")
            assert result == "[]"
            mock_engine.search.assert_called_once_with("no matches", max_results=10)
        finally:
            mux.server.search_engine = original

    @pytest.mark.asyncio
    async def test_search_tool_high_confidence(self):
        """Test search tool with high confidence results."""
        import mux.server

        # Mock search engine with high confidence results
        mock_engine = AsyncMock()
        mock_engine.search.return_value = [
            {
                "server": "test",
                "tool": "tool1",
                "description": "Test 1",
                "input_schema": {},
                "score": 0.95,
            },
            {
                "server": "test",
                "tool": "tool2",
                "description": "Test 2",
                "input_schema": {},
                "score": 0.85,
            },
            {
                "server": "test",
                "tool": "tool3",
                "description": "Test 3",
                "input_schema": {},
                "score": 0.75,
            },
        ]

        original = mux.server.search_engine
        mux.server.search_engine = mock_engine

        try:
            result = await search("test query")
            parsed = json.loads(result)

            # Should return only top result for very high confidence
            assert len(parsed) == 1
            assert parsed[0]["tool"] == "tool1"
            assert parsed[0]["score"] == 0.95
        finally:
            mux.server.search_engine = original

    @pytest.mark.asyncio
    async def test_search_tool_medium_confidence(self):
        """Test search tool with medium confidence results."""
        import mux.server

        # Mock search engine with medium confidence results
        mock_engine = AsyncMock()
        mock_engine.search.return_value = [
            {
                "server": "test",
                "tool": "tool1",
                "description": "Test 1",
                "input_schema": {},
                "score": 0.65,
            },
            {
                "server": "test",
                "tool": "tool2",
                "description": "Test 2",
                "input_schema": {},
                "score": 0.60,
            },
            {
                "server": "test",
                "tool": "tool3",
                "description": "Test 3",
                "input_schema": {},
                "score": 0.40,
            },
        ]

        original = mux.server.search_engine
        mux.server.search_engine = mock_engine

        try:
            result = await search("test query")
            parsed = json.loads(result)

            # Should return top 2 with close scores
            assert len(parsed) == 2
            assert parsed[0]["tool"] == "tool1"
            assert parsed[1]["tool"] == "tool2"
        finally:
            mux.server.search_engine = original

    @pytest.mark.asyncio
    async def test_search_tool_low_confidence(self):
        """Test search tool with low confidence results."""
        import mux.server

        # Mock search engine with low confidence results
        mock_engine = AsyncMock()
        mock_engine.search.return_value = [
            {
                "server": "test",
                "tool": "tool1",
                "description": "Test 1",
                "input_schema": {},
                "score": 0.45,
            },
            {
                "server": "test",
                "tool": "tool2",
                "description": "Test 2",
                "input_schema": {},
                "score": 0.40,
            },
            {
                "server": "test",
                "tool": "tool3",
                "description": "Test 3",
                "input_schema": {},
                "score": 0.35,
            },
        ]

        original = mux.server.search_engine
        mux.server.search_engine = mock_engine

        try:
            result = await search("test query")
            parsed = json.loads(result)

            # Should return up to 3 results for low confidence
            assert len(parsed) == 3
        finally:
            mux.server.search_engine = original

    @pytest.mark.asyncio
    async def test_execute_tool_no_manager(self):
        """Test execute tool when manager not initialized."""
        import mux.server

        original = mux.server.client_manager
        mux.server.client_manager = None

        try:
            from mux.exceptions import ToolExecutionError

            with pytest.raises(ToolExecutionError) as exc_info:
                await execute("server", "tool", {})
            assert "Client manager not initialized" in str(exc_info.value)
        finally:
            mux.server.client_manager = original

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        import mux.server

        # Mock client manager
        mock_manager = AsyncMock()
        mock_manager.execute_tool.return_value = "Success!"

        original = mux.server.client_manager
        mux.server.client_manager = mock_manager

        try:
            result = await execute("test-server", "echo", {"message": "hello"})
            assert result == "Success!"
            mock_manager.execute_tool.assert_called_once_with(
                "test-server", "echo", {"message": "hello"}
            )
        finally:
            mux.server.client_manager = original

    @pytest.mark.asyncio
    async def test_execute_tool_no_arguments(self):
        """Test execute tool with no arguments."""
        import mux.server

        # Mock client manager
        mock_manager = AsyncMock()
        mock_manager.execute_tool.return_value = "No args"

        original = mux.server.client_manager
        mux.server.client_manager = mock_manager

        try:
            result = await execute("test-server", "ping", None)
            assert result == "No args"
            mock_manager.execute_tool.assert_called_once_with("test-server", "ping", {})
        finally:
            mux.server.client_manager = original


class TestServiceLifecycle:
    """Test service initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_services(self):
        """Test service initialization."""
        with (
            patch("mux.server.load_config") as mock_load_config,
            patch("mux.server.ClientManager") as mock_client_manager,
            patch("mux.server.SearchEngine") as mock_search_engine,
        ):
            # Mock config
            mock_config = MagicMock()
            mock_config.mcpServers = {"test": MagicMock()}
            mock_config.search = MagicMock()
            mock_load_config.return_value = mock_config

            # Mock client manager
            mock_manager_instance = AsyncMock()
            mock_manager_instance.initialize = AsyncMock()
            mock_manager_instance.get_all_tools = AsyncMock(
                return_value=[
                    {"tool": "test1", "server": "server1"},
                    {"tool": "test2", "server": "server2"},
                ]
            )
            mock_manager_instance.clients = {"test": MagicMock()}
            mock_client_manager.return_value = mock_manager_instance

            # Mock search engine
            mock_engine_instance = AsyncMock()
            mock_engine_instance.index_tools = AsyncMock()
            mock_search_engine.return_value = mock_engine_instance

            await initialize_services()

            # Verify initialization
            mock_load_config.assert_called_once()
            mock_client_manager.assert_called_once_with(mock_config)
            mock_manager_instance.initialize.assert_called_once()
            mock_search_engine.assert_called_once_with(mock_config.search)
            mock_engine_instance.index_tools.assert_called_once_with(
                [
                    {"tool": "test1", "server": "server1"},
                    {"tool": "test2", "server": "server2"},
                ]
            )

    @pytest.mark.asyncio
    async def test_shutdown_services(self):
        """Test service shutdown."""
        import mux.server

        # Mock client manager
        mock_manager = AsyncMock()
        mock_manager.shutdown = AsyncMock()

        # Set up globals
        original_manager = mux.server.client_manager
        original_engine = mux.server.search_engine

        mux.server.client_manager = mock_manager
        mux.server.search_engine = MagicMock()

        try:
            await shutdown_services()

            # Verify shutdown
            mock_manager.shutdown.assert_called_once()
            assert mux.server.client_manager is None
            assert mux.server.search_engine is None
        finally:
            # Restore originals
            mux.server.client_manager = original_manager
            mux.server.search_engine = original_engine

    @pytest.mark.asyncio
    async def test_shutdown_services_with_error(self):
        """Test service shutdown handles errors."""
        import mux.server

        # Mock client manager that throws error
        mock_manager = AsyncMock()
        mock_manager.shutdown.side_effect = Exception("Shutdown error")

        # Set up globals
        original_manager = mux.server.client_manager
        original_engine = mux.server.search_engine

        mux.server.client_manager = mock_manager
        mux.server.search_engine = MagicMock()

        try:
            # Should not raise
            await shutdown_services()

            # Should still clean up
            assert mux.server.client_manager is None
            assert mux.server.search_engine is None
        finally:
            # Restore originals
            mux.server.client_manager = original_manager
            mux.server.search_engine = original_engine

    @pytest.mark.asyncio
    async def test_shutdown_services_none(self):
        """Test shutdown when services are None."""
        import mux.server

        # Save originals
        original_manager = mux.server.client_manager
        original_engine = mux.server.search_engine

        mux.server.client_manager = None
        mux.server.search_engine = None

        try:
            # Should not raise
            await shutdown_services()
        finally:
            # Restore originals
            mux.server.client_manager = original_manager
            mux.server.search_engine = original_engine


class TestLifespan:
    """Test lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan(self):
        """Test lifespan initialization and cleanup."""
        with (
            patch("mux.server.initialize_services") as mock_init,
            patch("mux.server.shutdown_services") as mock_shutdown,
        ):
            mock_init.return_value = None
            mock_shutdown.return_value = None

            # Import lifespan after patching
            from mux.server import lifespan

            # Test lifespan
            async with lifespan(None):
                mock_init.assert_called_once()
                mock_shutdown.assert_not_called()

            # After exiting, shutdown should be called
            mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_with_exception(self):
        """Test lifespan cleanup on exception."""
        with (
            patch("mux.server.initialize_services") as mock_init,
            patch("mux.server.shutdown_services") as mock_shutdown,
        ):
            mock_init.return_value = None
            mock_shutdown.return_value = None

            from mux.server import lifespan

            # Test lifespan with exception
            with pytest.raises(ValueError):
                async with lifespan(None):
                    mock_init.assert_called_once()
                    raise ValueError("Test error")

            # Shutdown should still be called
            mock_shutdown.assert_called_once()


class TestRunServer:
    """Test server running."""

    def test_run_mcp_server(self):
        """Test run_mcp_server function."""
        with patch("mux.server.mcp") as mock_mcp:
            from mux.server import run_mcp_server

            run_mcp_server()
            mock_mcp.run.assert_called_once()

    def test_run_mcp_server_keyboard_interrupt(self):
        """Test run_mcp_server handles KeyboardInterrupt."""
        with patch("mux.server.mcp") as mock_mcp:
            mock_mcp.run.side_effect = KeyboardInterrupt()

            from mux.server import run_mcp_server

            # Should not raise
            run_mcp_server()

    def test_run_mcp_server_unexpected_error(self):
        """Test run_mcp_server handles unexpected errors."""
        with patch("mux.server.mcp") as mock_mcp:
            mock_mcp.run.side_effect = Exception("Unexpected error")

            from mux.server import run_mcp_server

            with pytest.raises(Exception) as exc_info:
                run_mcp_server()
            assert "Unexpected error" in str(exc_info.value)
