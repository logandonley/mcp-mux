"""Utility functions for MCP Mux."""

import logging
from typing import Any

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging with rich handler."""
    logging.basicConfig(
        level=level, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger("mux")


def format_tool_info(tool: dict[str, Any]) -> str:
    """Format tool information for display."""
    lines = [
        f"Tool: {tool['tool']}",
        f"Server: {tool['server']}",
        f"Description: {tool['description']}",
    ]

    if "input_schema" in tool:
        schema = tool["input_schema"]
        if schema and "properties" in schema:
            lines.append("Parameters:")
            for param, details in schema["properties"].items():
                param_type = details.get("type", "any")
                description = details.get("description", "")
                required = param in schema.get("required", [])

                param_line = f"  - {param} ({param_type})"
                if required:
                    param_line += " [required]"
                if description:
                    param_line += f": {description}"
                lines.append(param_line)

    return "\n".join(lines)


def validate_tool_arguments(
    tool_schema: dict[str, Any], arguments: dict[str, Any]
) -> tuple[bool, str | None]:
    """
    Validate tool arguments against the schema.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not tool_schema:
        return True, None

    required_params = tool_schema.get("required", [])
    properties = tool_schema.get("properties", {})

    for param in required_params:
        if param not in arguments:
            return False, f"Missing required parameter: {param}"

    for param, value in arguments.items():
        if param not in properties:
            # Allow extra parameters unless additionalProperties is false
            if tool_schema.get("additionalProperties") is False:
                return False, f"Unknown parameter: {param}"
            continue

        param_schema = properties[param]
        param_type = param_schema.get("type")

        if param_type and not validate_type(value, param_type):
            return False, f"Parameter '{param}' should be of type {param_type}"

    return True, None


def validate_type(value: Any, expected_type: str) -> bool:
    """Validate that a value matches the expected JSON schema type."""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    if expected_type not in type_map:
        return True  # Unknown type, allow it

    expected = type_map[expected_type]
    return isinstance(value, expected)
