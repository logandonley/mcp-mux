"""Unit tests for configuration management."""
# pyright: reportCallIssue=false

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mux.config import (
    Config,
    MCPServerConfig,
    SearchConfig,
    get_config_path,
    init_config,
    load_config,
    save_config,
)


class TestSearchConfig:
    """Test SearchConfig model."""

    def test_default_values(self):
        """Test default values for SearchConfig."""
        config = SearchConfig()
        assert config.method == "local"
        assert config.model == "BAAI/bge-small-en-v1.5"
        assert config.cache_embeddings is True
        assert config.max_results == 5

    def test_custom_values(self):
        """Test custom values for SearchConfig."""
        config = SearchConfig(
            method="remote",
            model="custom-model",
            cache_embeddings=False,
            max_results=10,
        )
        assert config.method == "remote"
        assert config.model == "custom-model"
        assert config.cache_embeddings is False
        assert config.max_results == 10


class TestMCPServerConfig:
    """Test MCPServerConfig model."""

    def test_stdio_server(self):
        """Test stdio server configuration."""
        config = MCPServerConfig(
            command="python",
            args=["-m", "my_server"],
            env={"KEY": "value"},
            enabled=True,
        )
        assert config.command == "python"
        assert config.args == ["-m", "my_server"]
        assert config.env == {"KEY": "value"}
        assert config.url is None
        assert config.enabled is True

    def test_http_server(self):
        """Test HTTP server configuration."""
        config = MCPServerConfig(url="http://localhost:8080", enabled=False)
        assert config.url == "http://localhost:8080"
        assert config.command is None
        assert config.args == []
        assert config.env == {}
        assert config.enabled is False

    def test_validation_both_command_and_url(self):
        """Test that both command and URL cannot be specified."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerConfig(command="python", url="http://localhost:8080")
        assert "Cannot specify both command and url" in str(exc_info.value)

    def test_validation_neither_command_nor_url(self):
        """Test that either command or URL must be specified."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerConfig()
        assert "Must specify either command or url" in str(exc_info.value)


class TestConfig:
    """Test main Config model."""

    def test_default_values(self):
        """Test default values for Config."""
        config = Config()
        assert config.mcpServers == {}
        assert isinstance(config.search, SearchConfig)
        assert config.search.method == "local"

    def test_with_servers(self):
        """Test Config with MCP servers."""
        config = Config(
            mcpServers={
                "test-server": MCPServerConfig(command="test", enabled=True),
                "http-server": MCPServerConfig(url="http://test.com"),
            }
        )
        assert len(config.mcpServers) == 2
        assert "test-server" in config.mcpServers
        assert "http-server" in config.mcpServers

    def test_environment_variables(self):
        """Test loading from environment variables."""
        # Set environment variables
        os.environ["MUX_SEARCH__METHOD"] = "remote"
        os.environ["MUX_SEARCH__MAX_RESULTS"] = "20"

        try:
            config = Config()
            assert config.search.method == "remote"
            assert config.search.max_results == 20
        finally:
            # Clean up
            del os.environ["MUX_SEARCH__METHOD"]
            del os.environ["MUX_SEARCH__MAX_RESULTS"]


class TestConfigPath:
    """Test configuration path handling."""

    def test_default_path(self):
        """Test default configuration path."""
        # Save and clear any existing MUX_CONFIG_PATH
        old_env = os.environ.get("MUX_CONFIG_PATH")
        if old_env:
            del os.environ["MUX_CONFIG_PATH"]

        try:
            path = get_config_path()
            assert path == Path.home() / ".mux" / "config.json"
        finally:
            # Restore original env var if it existed
            if old_env:
                os.environ["MUX_CONFIG_PATH"] = old_env

    def test_explicit_path(self):
        """Test explicit path parameter."""
        custom_path = "/tmp/custom/config.json"
        path = get_config_path(custom_path)
        assert path == Path(custom_path)

    def test_environment_variable(self):
        """Test MUX_CONFIG_PATH environment variable."""
        env_path = "/tmp/env/config.json"
        os.environ["MUX_CONFIG_PATH"] = env_path

        try:
            path = get_config_path()
            assert path == Path(env_path)
        finally:
            del os.environ["MUX_CONFIG_PATH"]

    def test_priority_order(self):
        """Test that explicit path takes precedence over env var."""
        env_path = "/tmp/env/config.json"
        explicit_path = "/tmp/explicit/config.json"
        os.environ["MUX_CONFIG_PATH"] = env_path

        try:
            path = get_config_path(explicit_path)
            assert path == Path(explicit_path)
        finally:
            del os.environ["MUX_CONFIG_PATH"]


class TestConfigIO:
    """Test configuration loading and saving."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config_file = tmp_path / "test_config.json"

        # Create a config
        config = Config(
            mcpServers={"test": MCPServerConfig(command="test-cmd", args=["arg1"])},
            search=SearchConfig(method="remote", max_results=10),
        )

        # Save it
        save_config(config, config_file)
        assert config_file.exists()

        # Load it back
        loaded = load_config(config_file)
        assert loaded.mcpServers["test"].command == "test-cmd"
        assert loaded.mcpServers["test"].args == ["arg1"]
        assert loaded.search.method == "remote"
        assert loaded.search.max_results == 10

    def test_load_nonexistent(self, tmp_path):
        """Test loading from non-existent file returns default config."""
        config_file = tmp_path / "nonexistent.json"
        config = load_config(config_file)

        # Should return default config
        assert isinstance(config, Config)
        assert config.mcpServers == {}
        assert config.search.method == "local"

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises error."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json")

        from mux.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(config_file)
        assert "Invalid JSON in configuration file" in str(exc_info.value)

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test that save_config creates parent directories."""
        config_file = tmp_path / "deep" / "nested" / "dir" / "config.json"
        config = Config()

        save_config(config, config_file)
        assert config_file.exists()
        assert config_file.parent.exists()

    def test_init_config(self, tmp_path):
        """Test init_config creates default configuration."""
        config_file = tmp_path / "init_config.json"
        init_config(config_file)

        assert config_file.exists()

        # Load and verify
        with open(config_file) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "example-server" in data["mcpServers"]
        assert data["mcpServers"]["example-server"]["command"] == "uvx"
        assert data["mcpServers"]["example-server"]["enabled"] is False
        assert "search" in data

    def test_config_round_trip(self, tmp_path):
        """Test that config survives serialization round-trip."""
        config_file = tmp_path / "roundtrip.json"

        # Create complex config
        original = Config(
            mcpServers={
                "stdio-server": MCPServerConfig(
                    command="python",
                    args=["-m", "server"],
                    env={"VAR": "value"},
                    enabled=True,
                ),
                "http-server": MCPServerConfig(
                    url="https://api.example.com",
                    enabled=False,
                ),
            },
            search=SearchConfig(
                method="local",
                model="BAAI/bge-base-en-v1.5",
                cache_embeddings=False,
                max_results=15,
            ),
        )

        # Save and load
        save_config(original, config_file)
        loaded = load_config(config_file)

        # Verify everything matches
        assert len(loaded.mcpServers) == 2

        stdio = loaded.mcpServers["stdio-server"]
        assert stdio.command == "python"
        assert stdio.args == ["-m", "server"]
        assert stdio.env == {"VAR": "value"}
        assert stdio.enabled is True

        http = loaded.mcpServers["http-server"]
        assert http.url == "https://api.example.com"
        assert http.enabled is False

        assert loaded.search.method == "local"
        assert loaded.search.model == "BAAI/bge-base-en-v1.5"
        assert loaded.search.cache_embeddings is False
        assert loaded.search.max_results == 15
