"""Custom exceptions for MCP Mux."""


class MuxError(Exception):
    """Base exception for all MCP Mux errors."""


class ConnectionError(MuxError):
    """Raised when connection to MCP server fails."""


class ConfigurationError(MuxError):
    """Raised when configuration is invalid or missing."""


class ToolExecutionError(MuxError):
    """Raised when tool execution fails."""


class SearchError(MuxError):
    """Raised when search operation fails."""
