"""Unit tests for utility functions."""

from mux.utils import format_tool_info, validate_tool_arguments, validate_type


class TestFormatToolInfo:
    """Test format_tool_info function."""

    def test_basic_tool(self):
        """Test formatting basic tool info."""
        tool = {
            "tool": "echo",
            "server": "test-server",
            "description": "Echo a message",
        }

        result = format_tool_info(tool)
        assert "Tool: echo" in result
        assert "Server: test-server" in result
        assert "Description: Echo a message" in result

    def test_tool_with_parameters(self):
        """Test formatting tool with parameters."""
        tool = {
            "tool": "greet",
            "server": "test-server",
            "description": "Greet someone",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to greet",
                    },
                    "formal": {
                        "type": "boolean",
                        "description": "Use formal greeting",
                    },
                },
                "required": ["name"],
            },
        }

        result = format_tool_info(tool)
        assert "Tool: greet" in result
        assert "Parameters:" in result
        assert "- name (string) [required]: Name to greet" in result
        assert "- formal (boolean): Use formal greeting" in result

    def test_tool_without_schema(self):
        """Test formatting tool without input schema."""
        tool = {
            "tool": "ping",
            "server": "test-server",
            "description": "Simple ping",
            "input_schema": None,
        }

        result = format_tool_info(tool)
        assert "Tool: ping" in result
        assert "Parameters:" not in result

    def test_tool_with_empty_schema(self):
        """Test formatting tool with empty schema."""
        tool = {
            "tool": "status",
            "server": "test-server",
            "description": "Get status",
            "input_schema": {},
        }

        result = format_tool_info(tool)
        assert "Tool: status" in result
        assert "Parameters:" not in result

    def test_complex_parameters(self):
        """Test formatting tool with complex parameter types."""
        tool = {
            "tool": "search",
            "server": "test-server",
            "description": "Search for items",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "filters": {
                        "type": "array",
                        "description": "Filter criteria",
                    },
                    "options": {
                        "type": "object",
                        "description": "Search options",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
        }

        result = format_tool_info(tool)
        lines = result.split("\n")

        # Check all parameters are present
        param_lines = [line for line in lines if line.strip().startswith("- ")]
        assert len(param_lines) == 4
        assert any("query (string) [required]" in line for line in param_lines)
        assert any("filters (array)" in line for line in param_lines)
        assert any("options (object)" in line for line in param_lines)
        assert any("max_results (integer)" in line for line in param_lines)


class TestValidateToolArguments:
    """Test validate_tool_arguments function."""

    def test_no_schema(self):
        """Test validation with no schema (always valid)."""
        valid, error = validate_tool_arguments(None, {"any": "args"})  # type: ignore
        assert valid is True
        assert error is None

    def test_valid_arguments(self):
        """Test validation with valid arguments."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        valid, error = validate_tool_arguments(schema, {"name": "Alice", "age": 30})
        assert valid is True
        assert error is None

    def test_missing_required(self):
        """Test validation with missing required parameter."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }

        valid, error = validate_tool_arguments(schema, {"name": "Bob"})
        assert valid is False
        assert error is not None
        assert "Missing required parameter: email" in error

    def test_wrong_type(self):
        """Test validation with wrong parameter type."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }

        valid, error = validate_tool_arguments(schema, {"count": "not a number"})
        assert valid is False
        assert error is not None
        assert "Parameter 'count' should be of type integer" in error

    def test_additional_properties_allowed(self):
        """Test validation with additional properties (default allowed)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        valid, error = validate_tool_arguments(
            schema, {"name": "Charlie", "extra": "value"}
        )
        assert valid is True
        assert error is None

    def test_additional_properties_forbidden(self):
        """Test validation with additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        }

        valid, error = validate_tool_arguments(
            schema, {"name": "David", "extra": "value"}
        )
        assert valid is False
        assert error is not None
        assert "Unknown parameter: extra" in error

    def test_empty_schema(self):
        """Test validation with empty schema."""
        schema = {}
        valid, error = validate_tool_arguments(schema, {"any": "args"})
        assert valid is True
        assert error is None


class TestValidateType:
    """Test validate_type function."""

    def test_string_type(self):
        """Test string type validation."""
        assert validate_type("hello", "string") is True
        assert validate_type(123, "string") is False
        assert validate_type(None, "string") is False

    def test_number_type(self):
        """Test number type validation."""
        assert validate_type(123, "number") is True
        assert validate_type(45.6, "number") is True
        assert validate_type("123", "number") is False

    def test_integer_type(self):
        """Test integer type validation."""
        assert validate_type(123, "integer") is True
        assert validate_type(45.6, "integer") is False
        assert validate_type("123", "integer") is False

    def test_boolean_type(self):
        """Test boolean type validation."""
        assert validate_type(True, "boolean") is True
        assert validate_type(False, "boolean") is True
        assert validate_type(1, "boolean") is False
        assert validate_type("true", "boolean") is False

    def test_array_type(self):
        """Test array type validation."""
        assert validate_type([], "array") is True
        assert validate_type([1, 2, 3], "array") is True
        assert validate_type({"not": "array"}, "array") is False
        assert validate_type("[]", "array") is False

    def test_object_type(self):
        """Test object type validation."""
        assert validate_type({}, "object") is True
        assert validate_type({"key": "value"}, "object") is True
        assert validate_type([], "object") is False
        assert validate_type("object", "object") is False

    def test_null_type(self):
        """Test null type validation."""
        assert validate_type(None, "null") is True
        assert validate_type("null", "null") is False
        assert validate_type(0, "null") is False
        assert validate_type(False, "null") is False

    def test_unknown_type(self):
        """Test unknown type (always valid)."""
        assert validate_type("anything", "custom-type") is True
        assert validate_type(123, "unknown") is True
        assert validate_type(None, "weird-type") is True


class TestComplexScenarios:
    """Test complex validation scenarios."""

    def test_nested_validation(self):
        """Test validation with nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name"],
                },
                "settings": {
                    "type": "object",
                },
            },
            "required": ["user"],
        }

        # Valid nested object
        valid, error = validate_tool_arguments(
            schema,
            {
                "user": {"name": "Eve", "email": "eve@example.com"},
                "settings": {"theme": "dark"},
            },
        )
        assert valid is True
        assert error is None

        # Invalid - wrong type for nested object
        valid, error = validate_tool_arguments(
            schema,
            {"user": "not an object", "settings": {}},
        )
        assert valid is False
        assert error is not None
        assert "Parameter 'user' should be of type object" in error

    def test_array_validation(self):
        """Test validation with arrays."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array"},
                "numbers": {"type": "array"},
            },
        }

        # Valid arrays
        valid, error = validate_tool_arguments(
            schema,
            {"tags": ["python", "mcp"], "numbers": [1, 2, 3]},
        )
        assert valid is True
        assert error is None

        # Invalid - not an array
        valid, error = validate_tool_arguments(
            schema,
            {"tags": "not,an,array"},
        )
        assert valid is False
        assert error is not None
        assert "Parameter 'tags' should be of type array" in error
