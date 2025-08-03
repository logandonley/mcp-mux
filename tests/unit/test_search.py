"""Unit tests for semantic search engine."""
# pyright: reportAttributeAccessIssue=false

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mux.config import SearchConfig
from mux.search import RECOMMENDED_MODELS, SearchEngine


class TestSearchEngine:
    """Test SearchEngine class."""

    @pytest.fixture
    def search_config(self):
        """Provide a search configuration."""
        return SearchConfig(
            method="local",
            model="BAAI/bge-small-en-v1.5",
            cache_embeddings=True,
            max_results=5,
        )

    @pytest.fixture
    def mock_text_embedding(self):
        """Mock TextEmbedding class."""
        with patch("mux.search.TextEmbedding") as mock:
            # Create a mock instance that will be returned when TextEmbedding
            # is instantiated
            mock_instance = MagicMock()
            mock.return_value = mock_instance

            # Make embed return predictable embeddings
            def mock_embed(texts):
                embeddings = []
                for text in texts:
                    # Create a simple embedding based on text length
                    embedding = np.zeros(384)
                    embedding[0] = len(text) / 100.0
                    embeddings.append(embedding)
                return iter(embeddings)

            mock_instance.embed = mock_embed
            yield mock

    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""
        return [
            {
                "server": "test-server",
                "tool": "echo",
                "description": "Echo a message",
                "input_schema": {},
            },
            {
                "server": "test-server",
                "tool": "add_numbers",
                "description": "Add two numbers",
                "input_schema": {},
            },
            {
                "server": "file-server",
                "tool": "read_file",
                "description": "Read a file from disk",
                "input_schema": {},
            },
        ]

    def test_initialization(self, search_config):
        """Test SearchEngine initialization."""
        engine = SearchEngine(search_config)
        assert engine.config == search_config
        assert engine._models == {}
        assert engine.tools == []
        assert engine.embeddings is None
        assert engine._embedding_cache == {}
        assert engine._current_model_name == "BAAI/bge-small-en-v1.5"

    def test_get_model_caching(self, search_config, mock_text_embedding):
        """Test that models are cached properly."""
        engine = SearchEngine(search_config)

        # First call should create model
        model1 = engine._get_model()
        assert mock_text_embedding.called
        assert engine._models["BAAI/bge-small-en-v1.5"] is not None

        # Second call should return cached model
        mock_text_embedding.reset_mock()
        model2 = engine._get_model()
        assert not mock_text_embedding.called
        assert model1 is model2

    def test_model_switching(self, search_config, mock_text_embedding):  # noqa: ARG002
        """Test switching between models clears embeddings."""
        engine = SearchEngine(search_config)
        engine.embeddings = np.array([[1, 2, 3]])
        engine._embedding_cache = {"test": np.array([1, 2, 3])}

        # Change model
        engine.config.model = "sentence-transformers/all-MiniLM-L6-v2"
        engine._get_model()

        # Should clear embeddings and cache
        assert engine.embeddings is None
        assert engine._embedding_cache == {}
        assert engine._current_model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_create_tool_text(self, search_config):
        """Test tool text creation for embeddings."""
        engine = SearchEngine(search_config)

        tool = {
            "tool": "read_file",
            "description": "Read contents of a file",
            "server": "filesystem",
        }

        text = engine._create_tool_text(tool)

        # Should include tool name multiple times for emphasis
        assert text.count("read_file") == 3
        assert "Tool: read_file" in text
        assert "Description: Read contents of a file" in text
        assert "Server: filesystem" in text
        assert "read file" in text  # Expanded name

    def test_expand_tool_name_snake_case(self, search_config):
        """Test expanding snake_case tool names."""
        engine = SearchEngine(search_config)

        assert engine._expand_tool_name("read_file") == "read file"
        assert engine._expand_tool_name("list_all_files") == "list all files"
        assert engine._expand_tool_name("simple") == "simple"

    def test_expand_tool_name_camel_case(self, search_config):
        """Test expanding camelCase tool names."""
        engine = SearchEngine(search_config)

        assert engine._expand_tool_name("readFile") == "read file"
        assert engine._expand_tool_name("listAllFiles") == "list all files"
        assert engine._expand_tool_name("XMLParser") == "x m l parser"

    def test_expand_query(self, search_config):
        """Test query expansion for better matching."""
        engine = SearchEngine(search_config)

        # Test action verb expansion
        assert "read reading get fetch retrieve" in engine._expand_query("read files")
        assert "write writing save store put" in engine._expand_query("write data")
        assert "create creating make new add" in engine._expand_query("create item")

        # Test domain-specific expansions
        assert "filesystem file" in engine._expand_query("list files")
        assert "http url web" in engine._expand_query("fetch webpage")
        assert "screenshot screen capture" in engine._expand_query("take screenshot")
        assert "execute code run" in engine._expand_query("run python script")
        assert "database sql query" in engine._expand_query("query database")

    @pytest.mark.asyncio
    async def test_index_tools_empty(self, search_config, mock_text_embedding):  # noqa: ARG002
        """Test indexing with no tools."""
        engine = SearchEngine(search_config)
        await engine.index_tools([])

        assert engine.tools == []
        assert isinstance(engine.embeddings, np.ndarray)
        assert engine.embeddings.shape == (0,)

    @pytest.mark.asyncio
    async def test_index_tools(self, search_config, mock_text_embedding, sample_tools):  # noqa: ARG002
        """Test indexing tools."""
        engine = SearchEngine(search_config)
        await engine.index_tools(sample_tools)

        assert engine.tools == sample_tools
        assert engine.embeddings is not None
        assert engine.embeddings.shape == (3, 384)  # 3 tools, 384 dimensions

    @pytest.mark.asyncio
    async def test_index_tools_with_caching(
        self,
        search_config,
        mock_text_embedding,  # noqa: ARG002
        sample_tools,
    ):
        """Test that tools are cached when caching is enabled."""
        engine = SearchEngine(search_config)
        engine.config.cache_embeddings = True

        await engine.index_tools(sample_tools)

        # Check cache
        assert len(engine._embedding_cache) == 3
        assert "test-server:echo" in engine._embedding_cache
        assert "test-server:add_numbers" in engine._embedding_cache
        assert "file-server:read_file" in engine._embedding_cache

    @pytest.mark.asyncio
    async def test_search_empty_index(self, search_config, mock_text_embedding):  # noqa: ARG002
        """Test searching with no indexed tools."""
        engine = SearchEngine(search_config)
        results = await engine.search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_basic(self, search_config, mock_text_embedding, sample_tools):
        """Test basic search functionality."""
        engine = SearchEngine(search_config)

        # Create mock embeddings with different similarities
        def mock_embed(texts):
            embeddings = []
            for text in texts:
                embedding = np.zeros(384)
                if "echo" in text.lower():
                    embedding[0] = 0.9  # High similarity for echo
                elif "add" in text.lower():
                    embedding[0] = 0.5  # Medium similarity for add
                else:
                    embedding[0] = 0.1  # Low similarity for others
                embeddings.append(embedding)
            return iter(embeddings)

        mock_text_embedding.return_value.embed = mock_embed

        await engine.index_tools(sample_tools)

        # Search for echo
        results = await engine.search("echo message", max_results=2)

        assert len(results) <= 2
        assert all("score" in result for result in results)
        # Results should be sorted by score
        if len(results) > 1:
            assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.asyncio
    async def test_search_exact_match_boost(
        self, search_config, mock_text_embedding, sample_tools
    ):
        """Test that exact matches get score boost."""
        engine = SearchEngine(search_config)

        # Simple embeddings
        def mock_embed(texts):
            return iter([np.ones(384) * 0.5 for _ in texts])

        mock_text_embedding.return_value.embed = mock_embed

        await engine.index_tools(sample_tools)

        # Search for exact tool name
        results = await engine.search("echo", max_results=3)

        # Echo tool should have highest score due to exact match
        echo_result = next(r for r in results if r["tool"] == "echo")
        assert echo_result["score"] > 0.9  # Should get significant boost

    @pytest.mark.asyncio
    async def test_search_partial_match_boost(
        self, search_config, mock_text_embedding, sample_tools
    ):
        """Test that partial matches get score boost."""
        engine = SearchEngine(search_config)

        # Simple embeddings
        def mock_embed(texts):
            return iter([np.ones(384) * 0.5 for _ in texts])

        mock_text_embedding.return_value.embed = mock_embed

        await engine.index_tools(sample_tools)

        # Search with partial match
        results = await engine.search("read", max_results=3)

        # read_file should get a boost
        read_result = next(r for r in results if r["tool"] == "read_file")
        assert read_result["score"] > 0.6  # Should get partial match boost

    @pytest.mark.asyncio
    async def test_search_max_results(
        self,
        search_config,
        mock_text_embedding,  # noqa: ARG002
        sample_tools,
    ):
        """Test max_results parameter."""
        engine = SearchEngine(search_config)
        await engine.index_tools(sample_tools)

        # Request only 1 result
        results = await engine.search("tool", max_results=1)
        assert len(results) == 1

        # Request 2 results
        results = await engine.search("tool", max_results=2)
        assert len(results) == 2

    def test_add_tool(self, search_config):
        """Test adding a single tool."""
        engine = SearchEngine(search_config)
        tool = {"server": "test", "tool": "test_tool", "description": "Test"}

        engine.add_tool(tool)
        assert tool in engine.tools

    def test_remove_tool(self, search_config):
        """Test removing a tool."""
        engine = SearchEngine(search_config)
        engine.tools = [
            {"server": "test", "tool": "tool1", "description": "Test 1"},
            {"server": "test", "tool": "tool2", "description": "Test 2"},
            {"server": "other", "tool": "tool1", "description": "Other 1"},
        ]

        engine.remove_tool("test", "tool1")

        assert len(engine.tools) == 2
        assert not any(
            t["server"] == "test" and t["tool"] == "tool1" for t in engine.tools
        )


class TestRecommendedModels:
    """Test recommended models configuration."""

    def test_recommended_models_structure(self):
        """Test that RECOMMENDED_MODELS has the expected structure."""
        assert len(RECOMMENDED_MODELS) == 4

        for model in RECOMMENDED_MODELS:
            assert "name" in model
            assert "description" in model
            assert "dimensions" in model
            assert "speed" in model

            # Check types
            assert isinstance(model["name"], str)
            assert isinstance(model["description"], str)
            assert isinstance(model["dimensions"], int)
            assert model["speed"] in ["fast", "very fast", "moderate", "slow"]

    def test_recommended_models_dimensions(self):
        """Test that model dimensions are correct."""
        dimensions = {m["name"]: m["dimensions"] for m in RECOMMENDED_MODELS}

        assert dimensions["BAAI/bge-small-en-v1.5"] == 384
        assert dimensions["sentence-transformers/all-MiniLM-L6-v2"] == 384
        assert dimensions["BAAI/bge-base-en-v1.5"] == 768
        assert dimensions["BAAI/bge-large-en-v1.5"] == 1024
