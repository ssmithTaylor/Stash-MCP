"""Tests for semantic search module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from stash_mcp.search import (
    BM25Store,
    ChunkFilter,
    IndexMeta,
    SearchEngine,
    SearchResult,
    VectorStore,
    _chunk_text,
    _chunk_text_sliding_window,
    _content_hash,
    _normalize_path,
    _rrf_fuse,
)

# --- Mock embedding function (deterministic, no API calls) ---


async def mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embedding: keyword-based 16-dim vectors.

    Uses a simple keyword-counting approach to produce somewhat meaningful
    embeddings for testing, ensuring related texts produce similar vectors.
    """
    keywords = [
        "auth", "oauth", "flow", "meeting", "notes",
        "config", "database", "test", "search", "content",
        "section", "project", "file", "data", "code", "doc",
    ]
    embeddings = []
    for text in texts:
        text_lower = text.lower()
        vec = []
        for kw in keywords:
            count = text_lower.count(kw)
            vec.append(float(count))
        # Add a small constant to avoid zero vectors
        vec[0] += 0.1
        embeddings.append(vec)
    return embeddings


# --- VectorStore tests ---


class TestVectorStore:

    def test_empty_store(self):
        """Test that a new store is empty."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            assert store.count == 0
            assert store.search([1.0, 0.0, 0.0]) == []

    def test_add_and_search(self):
        """Test adding vectors and searching."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            embeddings = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            metadata = [
                {"file_path": "a.md", "chunk_index": 0, "content": "about A"},
                {"file_path": "b.md", "chunk_index": 0, "content": "about B"},
                {"file_path": "c.md", "chunk_index": 0, "content": "about C"},
            ]
            store.add(embeddings, metadata)
            assert store.count == 3

            # Search for vector close to first embedding
            results = store.search([0.9, 0.1, 0.0], top_n=2)
            assert len(results) == 2
            assert results[0]["file_path"] == "a.md"
            assert "score" in results[0]

    def test_persistence(self):
        """Test that store persists across instances."""
        with TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "vectors.pkl"
            store = VectorStore(store_path)
            store.add(
                [[1.0, 0.0], [0.0, 1.0]],
                [
                    {"file_path": "a.md", "chunk_index": 0},
                    {"file_path": "b.md", "chunk_index": 0},
                ],
            )
            store.save()

            # Reload
            store2 = VectorStore(store_path)
            assert store2.count == 2
            results = store2.search([1.0, 0.0])
            assert len(results) > 0

    def test_remove_by_file(self):
        """Test removing vectors by file path."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                [
                    {"file_path": "a.md", "chunk_index": 0},
                    {"file_path": "a.md", "chunk_index": 1},
                    {"file_path": "b.md", "chunk_index": 0},
                ],
            )
            assert store.count == 3

            removed = store.remove_by_file("a.md")
            assert removed == 2
            assert store.count == 1

            results = store.search([1.0, 0.0])
            assert len(results) == 1
            assert results[0]["file_path"] == "b.md"

    def test_remove_by_file_nonexistent(self):
        """Test removing a file that doesn't exist returns 0."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0]],
                [{"file_path": "a.md", "chunk_index": 0}],
            )
            removed = store.remove_by_file("nonexistent.md")
            assert removed == 0
            assert store.count == 1

    def test_clear(self):
        """Test clearing the store."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0]],
                [{"file_path": "a.md", "chunk_index": 0}],
            )
            store.clear()
            assert store.count == 0

    def test_add_mismatched_lengths_raises(self):
        """Test that mismatched embeddings/metadata lengths raise ValueError."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            with pytest.raises(ValueError, match="same length"):
                store.add(
                    [[1.0, 0.0]],
                    [
                        {"file_path": "a.md"},
                        {"file_path": "b.md"},
                    ],
                )

    def test_search_zero_vector(self):
        """Test searching with a zero query vector returns empty."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0]],
                [{"file_path": "a.md", "chunk_index": 0}],
            )
            results = store.search([0.0, 0.0])
            assert results == []

    def test_remove_all_vectors(self):
        """Test removing all vectors leaves store empty."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0]],
                [{"file_path": "a.md", "chunk_index": 0}],
            )
            store.remove_by_file("a.md")
            assert store.count == 0
            assert store.search([1.0, 0.0]) == []

    def test_search_with_mask_excludes_rows(self):
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]],
                [
                    {"file_path": "_reports/a.md", "chunk_index": 0},
                    {"file_path": "docs/b.md", "chunk_index": 0},
                    {"file_path": "docs/c.md", "chunk_index": 0},
                ],
            )
            mask = np.array([False, True, True])
            results = store.search([1.0, 0.0], top_n=2, mask=mask)
            # the best row is masked out; the two remaining positive-score rows come back in order
            assert [r["file_path"] for r in results] == ["docs/b.md", "docs/c.md"]

    def test_search_mask_length_mismatch_raises(self):
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add([[1.0, 0.0]], [{"file_path": "a.md", "chunk_index": 0}])
            with pytest.raises(ValueError):
                store.search([1.0, 0.0], mask=np.array([True, False]))

    def test_search_mask_all_false_returns_empty(self):
        """A scope filter that excludes every row must return [], not raise
        and not fall back to unmasked results."""
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]],
                [
                    {"file_path": "a.md", "chunk_index": 0},
                    {"file_path": "b.md", "chunk_index": 0},
                    {"file_path": "c.md", "chunk_index": 0},
                ],
            )
            mask = np.array([False, False, False])
            assert store.search([1.0, 0.0], top_n=5, mask=mask) == []


# --- Chunking tests ---


class TestChunking:

    def test_empty_text(self):
        """Test chunking empty text."""
        assert _chunk_text_sliding_window("") == []
        assert _chunk_text_sliding_window("   ") == []

    def test_short_text_single_chunk(self):
        """Test that text shorter than chunk_size returns a single chunk."""
        text = "Hello world"
        result = _chunk_text_sliding_window(text, chunk_size=1500)
        assert result == [text]

    def test_small_file_single_chunk(self):
        """Test that files smaller than chunk_size produce a single chunk naturally."""
        text = "A" * 100
        result = _chunk_text_sliding_window(text, chunk_size=1500)
        assert len(result) == 1
        assert result[0] == text

    def test_correct_number_of_chunks(self):
        """Test that the sliding window produces the correct number of chunks.

        With text=3000, chunk_size=1500, overlap=200 (step=1300):
        chunk 1: [0:1500], chunk 2: [1300:2800], chunk 3: [2600:3000]
        """
        text = "A" * 3000
        chunks = _chunk_text_sliding_window(text, chunk_size=1500, chunk_overlap=200)
        assert len(chunks) == 3

    def test_overlap_present(self):
        """Test that adjacent chunks share overlapping content."""
        # Place a distinctive string at the overlap boundary
        text = "X" * 1300 + "OVERLAP_MARKER" + "Y" * 1300
        chunks = _chunk_text_sliding_window(text, chunk_size=1500, chunk_overlap=200)
        # OVERLAP_MARKER should appear in at least two chunks
        overlap_count = sum(1 for c in chunks if "OVERLAP_MARKER" in c)
        assert overlap_count >= 2

    def test_chunks_cover_full_document(self):
        """Test that beginning and end of document appear in chunks."""
        # Use distinct start/end markers so we can verify full coverage
        text = "START " + "middle " * 250 + "END"
        chunks = _chunk_text_sliding_window(text, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) > 1
        # The first chunk must contain the document start
        assert "START" in chunks[0]
        # The last chunk must contain the document end
        assert "END" in chunks[-1]

    def test_configurable_chunk_size(self):
        """Test that chunk_size parameter controls the chunk size."""
        text = "A" * 1000
        chunks = _chunk_text_sliding_window(text, chunk_size=400, chunk_overlap=50)
        assert len(chunks) > 1
        # Each chunk (except possibly the last) should be at most chunk_size chars
        for chunk in chunks[:-1]:
            assert len(chunk) <= 400

    def test_no_empty_chunks(self):
        """Test that no empty chunks are returned."""
        text = "A" * 3000
        chunks = _chunk_text_sliding_window(text, chunk_size=1500, chunk_overlap=200)
        assert all(c for c in chunks)

    def test_backward_compat_chunk_text(self):
        """Test that the legacy _chunk_text function still works."""
        assert _chunk_text("") == []
        assert _chunk_text("Hello world") == ["Hello world"]


# --- IndexMeta tests ---


class TestIndexMeta:

    def test_save_and_load(self):
        """Test saving and loading index metadata."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.json"
            meta = IndexMeta(
                file_hashes={"a.md": "abc123"},
                chunk_counts={"a.md": 3},
                embedder_model="test-model",
            )
            meta.save(path)

            loaded = IndexMeta.load(path)
            assert loaded.file_hashes == {"a.md": "abc123"}
            assert loaded.chunk_counts == {"a.md": 3}
            assert loaded.embedder_model == "test-model"

    def test_load_missing_file(self):
        """Test loading from a missing file returns empty."""
        with TemporaryDirectory() as tmpdir:
            meta = IndexMeta.load(Path(tmpdir) / "nonexistent_meta.json")
            assert meta.file_hashes == {}
            assert meta.chunk_counts == {}

    def test_schema_version_round_trip_and_legacy_default(self):
        from stash_mcp.search import INDEX_SCHEMA_VERSION

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index_meta.json"
            meta = IndexMeta()
            assert meta.schema_version == INDEX_SCHEMA_VERSION
            meta.save(path)
            assert IndexMeta.load(path).schema_version == INDEX_SCHEMA_VERSION

            # Legacy file without the field loads as version 0
            path.write_text('{"file_hashes": {}, "chunk_counts": {}, "embedder_model": ""}')
            assert IndexMeta.load(path).schema_version == 0


# --- Content hash tests ---


class TestContentHash:

    def test_deterministic(self):
        """Test that hash is deterministic."""
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_content(self):
        """Test that different content produces different hashes."""
        assert _content_hash("hello") != _content_hash("world")


# --- SearchEngine tests ---


class TestSearchEngine:

    @pytest.fixture
    def engine_dirs(self):
        """Create temporary content and index directories."""
        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                yield Path(content_dir), Path(index_dir)

    @pytest.fixture
    def engine(self, engine_dirs):
        """Create a SearchEngine with mock embeddings."""
        content_dir, index_dir = engine_dirs
        # Create sample content
        (content_dir / "docs").mkdir()
        (content_dir / "docs" / "auth.md").write_text(
            "# Authentication\n\nThe OAuth2 flow begins with a redirect."
        )
        (content_dir / "notes.md").write_text(
            "# Meeting Notes\n\nDiscussed project timeline and milestones."
        )
        (content_dir / "config.py").write_text(
            "# Configuration\nDB_HOST = 'localhost'\nDB_PORT = 5432\n"
        )
        return SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
        )

    async def test_build_index(self, engine):
        """Test building the index."""
        total = await engine.build_index([
            "docs/auth.md", "notes.md", "config.py"
        ])
        assert total > 0
        assert engine.ready
        assert engine.indexed_files == 3

    async def test_search_returns_results(self, engine):
        """Test that search returns relevant results."""
        await engine.build_index(["docs/auth.md", "notes.md", "config.py"])
        results = await engine.search("authentication")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)

    async def test_search_empty_index(self, engine):
        """Test searching an empty index."""
        results = await engine.search("anything")
        assert results == []

    async def test_search_max_results(self, engine):
        """Test max_results limits output."""
        await engine.build_index(["docs/auth.md", "notes.md", "config.py"])
        results = await engine.search("anything", max_results=1)
        assert len(results) <= 1

    async def test_search_file_type_filter(self, engine):
        """Test file type filtering."""
        await engine.build_index(["docs/auth.md", "notes.md", "config.py"])
        results = await engine.search("anything", file_types=[".py"])
        for r in results:
            assert r.file_path.endswith(".py")

    async def test_index_file(self, engine):
        """Test indexing a single file."""
        chunks = await engine.index_file("docs/auth.md")
        assert chunks > 0
        assert engine.indexed_files == 1

    async def test_remove_file(self, engine):
        """Test removing a file from the index."""
        await engine.index_file("docs/auth.md")
        await engine.remove_file("docs/auth.md")
        assert engine.indexed_files == 0
        assert engine.indexed_chunks == 0

    async def test_incremental_indexing_skips_unchanged(self, engine):
        """Test that unchanged files are skipped during re-indexing."""
        await engine.build_index(["docs/auth.md"])
        first_count = engine.indexed_chunks

        # Re-index - should skip the unchanged file
        total = await engine.build_index(["docs/auth.md"])
        assert total == first_count

    async def test_reindex(self, engine):
        """Test full reindex."""
        await engine.build_index(["docs/auth.md"])
        total = await engine.reindex()
        assert total > 0
        assert engine.indexed_files == 3  # All files in content dir

    async def test_index_nonexistent_file(self, engine):
        """Test indexing a nonexistent file."""
        chunks = await engine.index_file("nonexistent.md")
        assert chunks == 0

    async def test_persistence_across_engine_instances(self, engine_dirs):
        """Test that index persists across engine instances."""
        content_dir, index_dir = engine_dirs
        (content_dir / "test.md").write_text("# Test\n\nSome content here.")

        # Build index with first engine
        engine1 = SearchEngine(
            content_dir=content_dir, index_dir=index_dir, embed_fn=mock_embed,
        )
        await engine1.build_index(["test.md"])
        assert engine1.indexed_chunks > 0

        # Create new engine - should load persisted index
        engine2 = SearchEngine(
            content_dir=content_dir, index_dir=index_dir, embed_fn=mock_embed,
        )
        assert engine2.indexed_chunks > 0
        assert engine2.ready

    async def test_embed_query_uses_mock(self, engine):
        """Test that _embed_query delegates to the mock embed function."""
        result = await engine._embed_query("authentication")
        assert isinstance(result, list)
        assert len(result) == 16  # 16-dim vectors from mock_embed

    async def test_stale_index_cleared_on_model_change(self, engine_dirs):
        """Test that changing embedder model clears stale index for rebuild."""
        content_dir, index_dir = engine_dirs
        (content_dir / "test.md").write_text("# Test\n\nContent here.")

        # Build index with model A
        engine1 = SearchEngine(
            content_dir=content_dir, index_dir=index_dir,
            embedder_model="model-a", embed_fn=mock_embed,
        )
        await engine1.build_index(["test.md"])
        assert engine1.ready
        assert engine1.store.count > 0

        # Create engine with model B — stale index should be cleared
        engine2 = SearchEngine(
            content_dir=content_dir, index_dir=index_dir,
            embedder_model="model-b", embed_fn=mock_embed,
        )
        assert not engine2.ready
        assert engine2.store.count == 0
        assert engine2.meta.file_hashes == {}
        assert engine2.meta.embedder_model == ""

        # Search should return empty when not ready
        results = await engine2.search("anything")
        assert results == []

        # build_index should re-embed all files (no skipping due to stale hash)
        chunks = await engine2.build_index(["test.md"])
        assert chunks > 0
        assert engine2.ready
        assert engine2.store.count > 0

    async def test_indexing_flag_during_build(self, engine_dirs):
        """Test that indexing property is True during build_index."""
        content_dir, index_dir = engine_dirs
        (content_dir / "test.md").write_text("# Test\n\nContent here.")

        seen_indexing = []

        async def tracking_embed(texts):
            seen_indexing.append(engine.indexing)
            return await mock_embed(texts)

        engine = SearchEngine(
            content_dir=content_dir, index_dir=index_dir,
            embed_fn=tracking_embed,
        )
        assert not engine.indexing
        await engine.build_index(["test.md"])
        # Flag was True during embedding
        assert any(seen_indexing), "indexing should be True during build"
        # After build completes, indexing should be False
        assert not engine.indexing
        assert engine.ready

    async def test_reindex_with_filesystem_filtering(self, engine_dirs):
        """Test that reindex uses FileSystem when provided."""
        content_dir, index_dir = engine_dirs
        (content_dir / "included.md").write_text("# Included\n\nMD content.")
        (content_dir / "excluded.py").write_text("# Excluded\nprint('hello')\n")

        from stash_mcp.filesystem import FileSystem
        fs = FileSystem(content_dir, include_patterns=["*.md"])

        engine = SearchEngine(
            content_dir=content_dir, index_dir=index_dir,
            embed_fn=mock_embed, filesystem=fs,
        )
        total = await engine.reindex()
        assert total > 0
        # Only .md files should be indexed
        assert "included.md" in engine.meta.file_hashes
        assert "excluded.py" not in engine.meta.file_hashes

    async def test_search_result_fields(self, engine):
        """Test that search results contain all expected fields."""
        await engine.build_index(["docs/auth.md", "notes.md"])
        results = await engine.search("authentication OAuth flow")
        assert len(results) > 0
        r = results[0]
        assert isinstance(r.file_path, str)
        assert r.chunk_index >= 0
        assert isinstance(r.content, str)
        assert r.score > 0

    async def test_chunk_size_param(self, engine_dirs):
        """Test that chunk_size and chunk_overlap params are respected."""
        content_dir, index_dir = engine_dirs
        # Write a file larger than a small chunk_size
        (content_dir / "large.md").write_text("Word " * 400)  # ~2000 chars

        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = await engine.index_file("large.md")
        # With chunk_size=500, overlap=50, step=450, ~2000 chars => more than 1 chunk
        assert chunks > 1

    async def test_chunks_carry_metadata_and_frontmatter_is_not_embedded(self, engine_dirs):
        content_dir, index_dir = engine_dirs
        (content_dir / "doc.md").write_text(
            "---\nlayer: frogpilot\nverified: 2026-08-16\n---\n# Doc\n\nOAuth flow text.\n"
        )
        engine = SearchEngine(content_dir=content_dir, index_dir=index_dir, embed_fn=mock_embed)
        await engine.build_index(["doc.md"])
        chunk = engine.store._metadata[0]
        assert chunk["metadata"] == {"layer": "frogpilot", "verified": "2026-08-16"}
        assert chunk["heading_path"] == ["Doc"]
        assert "layer: frogpilot" not in chunk["content"]
        results = await engine.search("oauth flow")
        assert results[0].metadata == {"layer": "frogpilot", "verified": "2026-08-16"}
        assert results[0].heading_path == ["Doc"]

    async def test_chunk_heading_path_follows_sections(self, engine_dirs):
        content_dir, index_dir = engine_dirs
        body = "# Svc\n\n## Role\n\n" + ("auth " * 300) + "\n\n## Config\n\n" + ("oauth " * 300)
        (content_dir / "svc.md").write_text(body)
        engine = SearchEngine(
            content_dir=content_dir, index_dir=index_dir, embed_fn=mock_embed,
            chunk_size=400, chunk_overlap=0,
        )
        await engine.build_index(["svc.md"])
        paths = [tuple(m["heading_path"]) for m in engine.store._metadata]
        assert paths[0] == ("Svc",)                       # first chunk starts on the H1 line
        assert ("Svc", "Role") in paths and ("Svc", "Config") in paths
        assert paths[-1] == ("Svc", "Config")

    def test_sliding_window_offsets_align_with_chunks(self):
        from stash_mcp.search import _chunk_text_sliding_window_with_offsets

        text = "  " + "abcdefghij" * 5 + "  "
        pairs = _chunk_text_sliding_window_with_offsets(text, chunk_size=20, chunk_overlap=5)
        stripped = text.strip()
        assert [c for c, _ in pairs] == _chunk_text_sliding_window(text, 20, 5)
        for chunk, start in pairs:
            assert stripped[start:start + len(chunk)] == chunk

    async def test_schema_mismatch_clears_index_for_rebuild(self, engine_dirs):
        import json

        content_dir, index_dir = engine_dirs
        (content_dir / "a.md").write_text("# A\n\nauth text")
        engine1 = SearchEngine(content_dir=content_dir, index_dir=index_dir, embed_fn=mock_embed)
        await engine1.build_index(["a.md"])
        assert engine1.indexed_chunks > 0

        meta_path = index_dir / "index_meta.json"
        data = json.loads(meta_path.read_text())
        data.pop("schema_version", None)          # simulate a pre-metadata index
        meta_path.write_text(json.dumps(data))

        engine2 = SearchEngine(content_dir=content_dir, index_dir=index_dir, embed_fn=mock_embed)
        assert engine2.indexed_chunks == 0        # cleared
        assert engine2.meta.file_hashes == {}     # so build_index re-embeds everything
        total = await engine2.build_index(["a.md"])
        assert total > 0
        assert engine2.store._metadata[0].get("metadata") == {}


# --- REST API search endpoint tests ---


class TestSearchAPI:

    @pytest.fixture
    def search_client(self):
        """Create a test client with search engine enabled."""
        import asyncio

        from fastapi.testclient import TestClient

        from stash_mcp.api import create_api
        from stash_mcp.filesystem import FileSystem

        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                fs = FileSystem(Path(content_dir))
                fs.write_file(
                    "docs/auth.md", "---\nlayer: frogpilot\n---\n# Auth\n\nOAuth2 flow here."
                )
                fs.write_file("notes.md", "# Notes\n\nMeeting notes.")
                fs.write_file("_reports/scan.md", "# Scan\n\nOAuth2 flow auth auth.")

                engine = SearchEngine(
                    content_dir=Path(content_dir),
                    index_dir=Path(index_dir),
                    embed_fn=mock_embed,
                    default_exclude_patterns=["**/_reports/**"],
                )

                # Build index directly since reindex endpoint is now non-blocking
                asyncio.run(
                    engine.build_index(["docs/auth.md", "notes.md", "_reports/scan.md"])
                )

                app = create_api(fs, search_engine=engine)
                client = TestClient(app)

                yield client

    def test_search_endpoint(self, search_client):
        """Test GET /api/search returns results."""
        response = search_client.get("/api/search", params={"q": "authentication"})
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total" in data

    def test_search_status_endpoint(self, search_client):
        """Test GET /api/search/status returns engine info."""
        response = search_client.get("/api/search/status")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["ready"] is True
        assert data["indexing"] is False
        assert "indexed_files" in data
        assert "indexed_chunks" in data

    def test_reindex_endpoint(self, search_client):
        """Test POST /api/search/reindex returns in_progress status."""
        response = search_client.post("/api/search/reindex")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"
        assert data["message"] == "Reindex started"

    def test_search_with_file_types(self, search_client):
        """Test search with file_types filter."""
        response = search_client.get(
            "/api/search",
            params={"q": "anything", "file_types": ".md"},
        )
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert result["file_path"].endswith(".md")

    def test_search_default_exclusion_and_include_excluded(self, search_client):
        data = search_client.get("/api/search", params={"q": "oauth flow"}).json()
        paths = [r["file_path"] for r in data["results"]]
        assert "docs/auth.md" in paths and "_reports/scan.md" not in paths
        data = search_client.get(
            "/api/search", params={"q": "oauth flow", "include_excluded": "true"},
        ).json()
        assert "_reports/scan.md" in [r["file_path"] for r in data["results"]]

    def test_search_result_fields_and_metadata_filter(self, search_client):
        data = search_client.get(
            "/api/search", params={"q": "oauth flow", "filter": ["layer:frogpilot"]},
        ).json()
        assert [r["file_path"] for r in data["results"]] == ["docs/auth.md"]
        item = data["results"][0]
        assert item["metadata"] == {"layer": "frogpilot"}
        assert item["heading_path"] == ["Auth"]
        for key in ("last_changed_at", "changed_by", "commit_message"):
            assert key in item

    def test_search_boost_prefix_reorders(self, search_client):
        plain = search_client.get(
            "/api/search", params={"q": "oauth flow", "include_excluded": "true"},
        ).json()["results"]
        boosted = search_client.get(
            "/api/search",
            params={"q": "oauth flow", "include_excluded": "true", "boost_prefix": "_reports/"},
        ).json()["results"]
        assert plain[0]["file_path"] == "docs/auth.md"
        assert boosted[0]["file_path"] == "_reports/scan.md"

    def test_search_path_prefix_and_bad_filter(self, search_client):
        data = search_client.get(
            "/api/search", params={"q": "oauth flow", "path_prefix": "docs"},
        ).json()
        assert all(r["file_path"].startswith("docs/") for r in data["results"])
        resp = search_client.get("/api/search", params={"q": "x", "filter": ["nocolon"]})
        assert resp.status_code == 400

    def test_search_status_reports_default_excludes(self, search_client):
        data = search_client.get("/api/search/status").json()
        assert data["default_exclude_patterns"] == ["**/_reports/**"]
        assert data["boost_weight"] == 0.15

    def test_search_rejects_parent_traversal_prefix(self, search_client):
        """REST must reject '..' path segments exactly like the MCP tool does
        (see TestMCPSearchTool.test_search_tool_rejects_parent_traversal_prefix),
        including the backslash-spelled form, for both path_prefix and
        boost_prefix."""
        resp = search_client.get(
            "/api/search", params={"q": "auth", "path_prefix": "../etc"},
        )
        assert resp.status_code == 400
        assert "path_prefix" in resp.json()["detail"]

        resp = search_client.get(
            "/api/search", params={"q": "auth", "path_prefix": "..\\etc"},
        )
        assert resp.status_code == 400
        assert "path_prefix" in resp.json()["detail"]

        resp = search_client.get(
            "/api/search", params={"q": "auth", "boost_prefix": "../etc"},
        )
        assert resp.status_code == 400
        assert "boost_prefix" in resp.json()["detail"]

    def test_search_accepts_prefix_merely_containing_dotdot(self, search_client):
        """A legitimate name that contains '..' without it being its own path
        segment (e.g. "archive..old") is not a traversal and must be accepted."""
        resp = search_client.get(
            "/api/search", params={"q": "oauth flow", "path_prefix": "archive..old"},
        )
        assert resp.status_code == 200


# --- API without search engine ---


class TestAPIWithoutSearch:

    def test_no_search_endpoints_when_disabled(self):
        """Test that search endpoints are not registered when engine is None."""
        from fastapi.testclient import TestClient

        from stash_mcp.api import create_api
        from stash_mcp.filesystem import FileSystem

        with TemporaryDirectory() as tmpdir:
            fs = FileSystem(Path(tmpdir))
            app = create_api(fs)  # No search_engine
            client = TestClient(app)

            response = client.get("/api/search", params={"q": "test"})
            assert response.status_code == 404

            response = client.get("/api/search/status")
            assert response.status_code == 404


# --- MCP search tool tests ---


class TestMCPSearchTool:

    async def test_search_tool_registered_when_engine_present(self):
        """Test that search_content tool is registered when engine is given."""
        from stash_mcp.filesystem import FileSystem
        from stash_mcp.mcp_server import create_mcp_server

        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                fs = FileSystem(Path(content_dir))
                engine = SearchEngine(
                    content_dir=Path(content_dir),
                    index_dir=Path(index_dir),
                    embed_fn=mock_embed,
                )
                mcp = create_mcp_server(fs, search_engine=engine)
                tools = await mcp.get_tools()
                assert "search_content" in tools

    async def test_search_tool_not_registered_without_engine(self):
        """Test that search_content tool is NOT registered without engine."""
        from stash_mcp.filesystem import FileSystem
        from stash_mcp.mcp_server import create_mcp_server

        with TemporaryDirectory() as content_dir:
            fs = FileSystem(Path(content_dir))
            mcp = create_mcp_server(fs)
            tools = await mcp.get_tools()
            assert "search_content" not in tools

    async def test_search_tool_returns_results(self):
        """Test search_content tool returns formatted results."""
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        from stash_mcp.filesystem import FileSystem
        from stash_mcp.mcp_server import create_mcp_server

        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                fs = FileSystem(Path(content_dir))
                fs.write_file("test.md", "# Test\n\nSome searchable content.")

                engine = SearchEngine(
                    content_dir=Path(content_dir),
                    index_dir=Path(index_dir),
                    embed_fn=mock_embed,
                )
                await engine.build_index(["test.md"])

                mcp = create_mcp_server(fs, search_engine=engine)
                tool = await mcp.get_tool("search_content")

                # Set up mock context
                ctx = MagicMock(spec=Context)
                ctx.session = AsyncMock()
                token = _current_context.set(ctx)
                try:
                    result = await tool.run({"query": "searchable content"})
                    text = str(result.content)
                    assert "test.md" in text
                finally:
                    _current_context.reset(token)

    async def test_search_tool_empty_index(self):
        """Test search_content tool with empty index."""
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        from stash_mcp.filesystem import FileSystem
        from stash_mcp.mcp_server import create_mcp_server

        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                fs = FileSystem(Path(content_dir))
                engine = SearchEngine(
                    content_dir=Path(content_dir),
                    index_dir=Path(index_dir),
                    embed_fn=mock_embed,
                )
                mcp = create_mcp_server(fs, search_engine=engine)
                tool = await mcp.get_tool("search_content")

                ctx = MagicMock(spec=Context)
                ctx.session = AsyncMock()
                token = _current_context.set(ctx)
                try:
                    result = await tool.run({"query": "anything"})
                    assert "No results found" in str(result.content)
                finally:
                    _current_context.reset(token)

    async def _tool_with_store(self, content_dir, index_dir, files, **engine_kw):
        from stash_mcp.filesystem import FileSystem
        from stash_mcp.mcp_server import create_mcp_server

        fs = FileSystem(Path(content_dir))
        for path, text in files.items():
            fs.write_file(path, text)
        engine = SearchEngine(
            content_dir=Path(content_dir), index_dir=Path(index_dir),
            embed_fn=mock_embed, **engine_kw,
        )
        await engine.build_index(list(files))
        mcp = create_mcp_server(fs, search_engine=engine)
        return await mcp.get_tool("search_content")

    async def test_search_tool_default_exclusion_and_override(self):
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            tool = await self._tool_with_store(
                content_dir, index_dir,
                {
                    "_reports/scan.md": "# Scan\n\nauth oauth flow",
                    "docs/auth.md": "---\nlayer: frogpilot\n---\n# Auth\n\nauth oauth flow",
                },
                default_exclude_patterns=["**/_reports/**"],
            )
            ctx = MagicMock(spec=Context)
            ctx.session = AsyncMock()
            token = _current_context.set(ctx)
            try:
                text = str((await tool.run({"query": "auth oauth"})).content)
                assert "docs/auth.md" in text
                assert "_reports/scan.md" not in text
                assert "Meta: layer=frogpilot" in text
                assert "Section: Auth" in text

                text = str((await tool.run({
                    "query": "auth oauth", "include_excluded": True,
                })).content)
                assert "_reports/scan.md" in text

                text = str((await tool.run({
                    "query": "auth oauth", "include_excluded": True, "boost_prefix": "_reports/",
                })).content)
                assert text.index("_reports/scan.md") < text.index("docs/auth.md")

                text = str((await tool.run({
                    "query": "auth oauth", "metadata_filters": {"layer": "nope"},
                })).content)
                assert "No results found" in text
            finally:
                _current_context.reset(token)

    async def test_search_tool_rejects_parent_traversal_prefix(self):
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            tool = await self._tool_with_store(
                content_dir, index_dir, {"a.md": "# A\n\nauth"},
            )
            ctx = MagicMock(spec=Context)
            ctx.session = AsyncMock()
            token = _current_context.set(ctx)
            try:
                with pytest.raises(ValueError, match="path_prefix"):
                    await tool.run({"query": "auth", "path_prefix": "../etc"})

                # Backslash-spelled traversal must be caught the same way as the
                # forward-slash form — _normalize_path (search.py) treats '\\' as a
                # path separator too, so the validation must match that convention.
                with pytest.raises(ValueError, match="path_prefix"):
                    await tool.run({"query": "auth", "path_prefix": "..\\etc"})
            finally:
                _current_context.reset(token)

    async def test_search_tool_accepts_prefix_merely_containing_dotdot(self):
        """A legitimate name that contains '..' without it being its own path
        segment (e.g. "archive..old") is not a traversal and must be accepted
        — mirrors TestSearchAPI.test_search_accepts_prefix_merely_containing_dotdot,
        confirming REST and MCP share the same accept/reject decision."""
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            tool = await self._tool_with_store(
                content_dir, index_dir,
                {"archive..old/notes.md": "# A\n\nauth oauth flow"},
            )
            ctx = MagicMock(spec=Context)
            ctx.session = AsyncMock()
            token = _current_context.set(ctx)
            try:
                text = str((await tool.run({
                    "query": "auth oauth", "path_prefix": "archive..old",
                })).content)
                assert "archive..old/notes.md" in text
            finally:
                _current_context.reset(token)

    async def test_search_tool_path_prefix_narrows_results(self):
        """A valid path_prefix actually scopes results, not just the rejection path."""
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            tool = await self._tool_with_store(
                content_dir, index_dir,
                {
                    "docs/auth.md": "# Auth\n\nauth oauth flow",
                    "other/auth.md": "# Auth\n\nauth oauth flow",
                },
            )
            ctx = MagicMock(spec=Context)
            ctx.session = AsyncMock()
            token = _current_context.set(ctx)
            try:
                text = str((await tool.run({
                    "query": "auth oauth", "path_prefix": "docs",
                })).content)
                assert "docs/auth.md" in text
                assert "other/auth.md" not in text
            finally:
                _current_context.reset(token)

    async def test_search_tool_exclude_patterns_filters_results(self):
        """The exclude_patterns tool argument is threaded through, not just
        the engine-level default_exclude_patterns."""
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            tool = await self._tool_with_store(
                content_dir, index_dir,
                {
                    "docs/auth.md": "# Auth\n\nauth oauth flow",
                    "scratch/auth.md": "# Auth\n\nauth oauth flow",
                },
            )
            ctx = MagicMock(spec=Context)
            ctx.session = AsyncMock()
            token = _current_context.set(ctx)
            try:
                text = str((await tool.run({
                    "query": "auth oauth", "exclude_patterns": "scratch/**",
                })).content)
                assert "docs/auth.md" in text
                assert "scratch/auth.md" not in text
            finally:
                _current_context.reset(token)

    async def test_search_tool_omits_section_and_meta_labels_when_absent(self):
        """A chunk with no heading and a document with no frontmatter must not
        emit dangling Section:/Meta: labels."""
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp.server.context import Context, _current_context

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            tool = await self._tool_with_store(
                content_dir, index_dir,
                {"plain.md": "auth oauth flow with no heading and no frontmatter"},
            )
            ctx = MagicMock(spec=Context)
            ctx.session = AsyncMock()
            token = _current_context.set(ctx)
            try:
                text = str((await tool.run({"query": "auth oauth"})).content)
                assert "plain.md" in text
                assert "Section:" not in text
                assert "Meta:" not in text
            finally:
                _current_context.reset(token)


# --- Startup index build via lifespan ---


class TestStartupIndexBuild:

    def test_lifespan_builds_index_for_preexisting_files(self, monkeypatch):
        """Test that create_app's lifespan builds the search index for files
        that already exist when the server starts.

        Previously this used @app.on_event('startup') which is silently
        ignored when a lifespan handler is set on the FastAPI app.
        """
        import asyncio

        from fastapi.testclient import TestClient

        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            cd = Path(content_dir)
            idx = Path(index_dir)

            # Pre-populate content
            (cd / "docs").mkdir()
            (cd / "docs" / "auth.md").write_text(
                "# Authentication\n\nThe OAuth2 flow begins."
            )
            (cd / "notes.md").write_text(
                "# Meeting Notes\n\nDiscussed project timeline."
            )

            monkeypatch.setattr("stash_mcp.config.Config.CONTENT_DIR", cd)
            monkeypatch.setattr("stash_mcp.config.Config.SEARCH_ENABLED", True)
            monkeypatch.setattr("stash_mcp.config.Config.SEARCH_INDEX_DIR", idx)
            monkeypatch.setattr("stash_mcp.config.Config.CONTENT_PATHS", None)

            # Patch _create_search_engine to use mock_embed
            from stash_mcp import main as main_mod

            _original = main_mod._create_search_engine

            def _patched():
                engine = SearchEngine(
                    content_dir=cd,
                    index_dir=idx,
                    embed_fn=mock_embed,
                )
                return engine

            monkeypatch.setattr(main_mod, "_create_search_engine", _patched)

            from stash_mcp.main import create_app

            app = create_app()

            # TestClient triggers the lifespan (startup + shutdown)
            with TestClient(app) as client:
                # Poll for background index build to complete
                import time
                for _ in range(50):
                    resp = client.get("/api/search/status")
                    data = resp.json()
                    if resp.status_code == 200 and data.get("ready") is True:
                        break
                    time.sleep(0.1)

                # Verify search status shows indexed files
                resp = client.get("/api/search/status")
                assert resp.status_code == 200
                data = resp.json()
                assert data["ready"] is True
                assert data["indexed_files"] == 2

                # Verify search returns results
                resp = client.get("/api/search", params={"q": "authentication"})
                assert resp.status_code == 200
                data = resp.json()
                assert data["total"] > 0


class TestSearchConfig:

    def test_search_disabled_by_default(self):
        """Test that search is disabled by default."""
        from stash_mcp.config import Config

        assert Config.SEARCH_ENABLED is False

    def test_search_config_defaults(self):
        """Test search config default values."""
        from stash_mcp.config import Config

        assert Config.SEARCH_INDEX_DIR == Path("/data/.stash-index")
        assert "sentence-transformers" in Config.SEARCH_EMBEDDER_MODEL
        assert Config.CONTEXTUAL_RETRIEVAL is False
        assert Config.CONTEXTUAL_MODEL == "claude-haiku-4-5-20251001"
        assert Config.SEARCH_CHUNK_SIZE == 1000
        assert Config.SEARCH_CHUNK_OVERLAP == 100

    def test_model_cache_dir_default(self):
        """Test that MODEL_CACHE_DIR defaults to /data/models."""
        from stash_mcp.config import Config

        assert Config.MODEL_CACHE_DIR == Path("/data/models")

    def test_search_exclude_patterns_default_unset(self):
        from stash_mcp.config import Config

        assert Config.SEARCH_EXCLUDE_PATTERNS is None

    def test_search_exclude_patterns_parse(self):
        from stash_mcp.config import _parse_content_paths

        assert _parse_content_paths("**/_reports/, **/_archive/**") == [
            "**/_reports/**", "**/_archive/**",
        ]


# --- Path normalization tests ---


class TestNormalizePath:

    def test_forward_slash_unchanged(self):
        """Test that a well-formed path is unchanged."""
        assert _normalize_path("docs/api.md") == "docs/api.md"

    def test_strips_leading_slash(self):
        """Test that a leading slash is stripped."""
        assert _normalize_path("/docs/api.md") == "docs/api.md"

    def test_strips_trailing_slash(self):
        """Test that a trailing slash is stripped."""
        assert _normalize_path("docs/api.md/") == "docs/api.md"

    def test_strips_both_slashes(self):
        """Test that leading and trailing slashes are both stripped."""
        assert _normalize_path("/docs/api.md/") == "docs/api.md"

    def test_normalizes_backslashes(self):
        """Test that backslashes are converted to forward slashes (Windows paths)."""
        assert _normalize_path("docs\\api.md") == "docs/api.md"

    def test_normalizes_backslashes_and_leading_slash(self):
        """Test backslash normalization combined with leading slash removal."""
        assert _normalize_path("\\docs\\api.md") == "docs/api.md"

    def test_empty_string(self):
        """Test that empty string stays empty."""
        assert _normalize_path("") == ""


# --- ChunkFilter tests ---


class TestChunkFilter:
    def _chunk(self, path, **meta):
        return {"file_path": path, "chunk_index": 0, "content": "", "metadata": meta}

    def test_inactive_when_empty(self):
        assert ChunkFilter().active is False
        assert ChunkFilter(path_prefixes=["docs"]).active is True

    def test_path_prefix_is_subtree_not_string_prefix(self):
        pred = ChunkFilter(path_prefixes=["docs/"]).compile()
        assert pred(self._chunk("docs/a.md"))
        assert pred(self._chunk("docs/sub/a.md"))
        assert not pred(self._chunk("docs2/a.md"))
        assert not pred(self._chunk("a.md"))

    def test_multiple_prefixes_are_any_of(self):
        pred = ChunkFilter(path_prefixes=["projects/stash-mcp", "systems/homelab/"]).compile()
        assert pred(self._chunk("projects/stash-mcp/services/x.md"))
        assert pred(self._chunk("systems/homelab/operations/y.md"))
        assert not pred(self._chunk("projects/openpilot/services/z.md"))

    def test_normalize_prefixes_and_path_under_any(self):
        from stash_mcp.search import normalize_prefixes, path_under_any

        assert normalize_prefixes("projects/a/, /systems/b ,,") == ["projects/a", "systems/b"]
        assert normalize_prefixes(["x/"]) == ["x"]
        assert normalize_prefixes(None) == []
        assert path_under_any("projects/a/f.md", ["projects/a"])
        assert path_under_any("projects/a", ["projects/a"])
        assert not path_under_any("projects/ab/f.md", ["projects/a"])

    def test_exclude_any_depth_vs_root_anchored(self):
        pred = ChunkFilter(exclude_patterns=["**/_reports/**"]).compile()
        assert not pred(self._chunk("_reports/scan.md"))
        assert not pred(self._chunk("openpilot/_reports/x.md"))
        assert pred(self._chunk("openpilot/services/pandad.md"))

        pred_root = ChunkFilter(exclude_patterns=["_reports/"]).compile()
        assert not pred_root(self._chunk("_reports/scan.md"))
        assert pred_root(self._chunk("openpilot/_reports/x.md"))

    def test_file_types(self):
        pred = ChunkFilter(file_types=[".md", ".py"]).compile()
        assert pred(self._chunk("a.md"))
        assert pred(self._chunk("b.py"))
        assert not pred(self._chunk("c.json"))

    def test_metadata_equality_all_keys_must_match(self):
        pred = ChunkFilter(metadata={"layer": "frogpilot", "verified": "2026-08-16"}).compile()
        assert pred(self._chunk("a.md", layer="frogpilot", verified="2026-08-16"))
        assert not pred(self._chunk("a.md", layer="frogpilot"))
        assert not pred(self._chunk("a.md", layer="moretore", verified="2026-08-16"))
        assert not pred({"file_path": "a.md"})  # no metadata key at all

    def test_combined(self):
        pred = ChunkFilter(
            path_prefixes=["openpilot"], exclude_patterns=["**/_reports/**"], file_types=[".md"],
        ).compile()
        assert pred(self._chunk("openpilot/services/x.md"))
        assert not pred(self._chunk("openpilot/_reports/x.md"))
        assert not pred(self._chunk("openpilot/services/x.py"))
        assert not pred(self._chunk("services/x.md"))


# --- Search index integrity tests (delete/move) ---


class TestSearchIndexIntegrity:

    @pytest.fixture
    def engine_with_files(self):
        """Create a SearchEngine with pre-indexed files."""
        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                cd = Path(content_dir)
                (cd / "docs").mkdir()
                (cd / "docs" / "auth.md").write_text(
                    "# Authentication\n\nThe OAuth2 flow is used for authorization."
                )
                (cd / "notes.md").write_text(
                    "# Meeting Notes\n\nDiscussed project milestones and deliverables."
                )
                engine = SearchEngine(
                    content_dir=cd,
                    index_dir=Path(index_dir),
                    embed_fn=mock_embed,
                )
                yield engine, cd

    async def test_delete_file_removed_from_search(self, engine_with_files):
        """Test that deleted files no longer appear in search results."""
        engine, content_dir = engine_with_files
        await engine.build_index(["docs/auth.md", "notes.md"])
        assert engine.indexed_files == 2

        # Delete the file and remove from index
        (content_dir / "docs" / "auth.md").unlink()
        await engine.remove_file("docs/auth.md")

        assert engine.indexed_files == 1
        assert engine.indexed_chunks < engine.store.count or engine.indexed_chunks >= 0
        assert "docs/auth.md" not in engine.meta.file_hashes

        # Search should no longer return results for the deleted file
        results = await engine.search("OAuth2 authorization")
        file_paths = [r.file_path for r in results]
        assert "docs/auth.md" not in file_paths

    async def test_delete_file_with_leading_slash_normalization(self, engine_with_files):
        """Test that remove_file works even when path has a leading slash."""
        engine, content_dir = engine_with_files
        await engine.build_index(["docs/auth.md", "notes.md"])

        (content_dir / "docs" / "auth.md").unlink()
        # Pass path with leading slash (as might come from user input/API)
        await engine.remove_file("/docs/auth.md")

        assert "docs/auth.md" not in engine.meta.file_hashes
        results = await engine.search("OAuth2 authorization")
        file_paths = [r.file_path for r in results]
        assert "docs/auth.md" not in file_paths

    async def test_move_file_updates_index(self, engine_with_files):
        """Test that moved files: old path gone, new path searchable."""
        engine, content_dir = engine_with_files
        await engine.build_index(["docs/auth.md", "notes.md"])

        # Move the file on disk
        old_path = content_dir / "docs" / "auth.md"
        new_path = content_dir / "docs" / "auth-guide.md"
        old_path.rename(new_path)

        # Use the atomic move_file_index method
        await engine.move_file_index("docs/auth.md", "docs/auth-guide.md")

        # Old path should be gone
        assert "docs/auth.md" not in engine.meta.file_hashes
        # New path should be indexed
        assert "docs/auth-guide.md" in engine.meta.file_hashes

        # Search should return the new path, not the old
        results = await engine.search("OAuth2 authorization")
        file_paths = [r.file_path for r in results]
        assert "docs/auth.md" not in file_paths
        assert "docs/auth-guide.md" in file_paths

    async def test_move_file_with_path_normalization(self, engine_with_files):
        """Test move_file_index with paths that need normalization."""
        engine, content_dir = engine_with_files
        await engine.build_index(["docs/auth.md"])

        old_path = content_dir / "docs" / "auth.md"
        new_path = content_dir / "docs" / "auth-guide.md"
        old_path.rename(new_path)

        # Pass with leading slashes (as from user input)
        await engine.move_file_index("/docs/auth.md", "/docs/auth-guide.md")

        assert "docs/auth.md" not in engine.meta.file_hashes
        assert "docs/auth-guide.md" in engine.meta.file_hashes

    async def test_embedder_loaded_at_init(self):
        """Test that the embedder is None when embed_fn is provided (not lazily created)."""
        with TemporaryDirectory() as content_dir:
            with TemporaryDirectory() as index_dir:
                engine = SearchEngine(
                    content_dir=Path(content_dir),
                    index_dir=Path(index_dir),
                    embed_fn=mock_embed,
                )
                # With embed_fn, _embedder should be None (no model to load)
                assert engine._embedder is None

    async def test_path_normalization_in_vector_store(self):
        """Test that remove_by_file normalizes paths for correct matching."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [[1.0, 0.0], [0.0, 1.0]],
                [
                    {"file_path": "docs/api.md", "chunk_index": 0},
                    {"file_path": "notes.md", "chunk_index": 0},
                ],
            )
            assert store.count == 2

            # Remove with leading slash - should still match "docs/api.md"
            removed = store.remove_by_file("/docs/api.md")
            assert removed == 1
            assert store.count == 1

            results = store.search([0.0, 1.0])
            assert results[0]["file_path"] == "notes.md"


# --- MMR + per-file cap tests ---


class TestVectorStoreMMR:
    """search_mmr() reranks the cosine pool for diversity + per-file cap."""

    def _store(self, tmpdir):
        store = VectorStore(Path(tmpdir) / "vectors.pkl")
        # vec_a and vec_b nearly identical, vec_c has a small positive
        # component on the query axis so it survives the >0 filter but
        # is far away from a/b in the orthogonal direction.
        store.add(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.1, 0.0],
                [0.1, 0.0, 1.0],
            ],
            [
                {"file_path": "a.md", "chunk_index": 0, "content": "A"},
                {"file_path": "b.md", "chunk_index": 0, "content": "B"},
                {"file_path": "c.md", "chunk_index": 0, "content": "C"},
            ],
        )
        return store

    def test_mmr_diversifies_vs_pure_cosine(self):
        """MMR with lambda<1 prefers the diverse vector over the redundant one."""
        with TemporaryDirectory() as tmpdir:
            store = self._store(tmpdir)
            cosine = store.search([1.0, 0.0, 0.0], top_n=2)
            assert [r["file_path"] for r in cosine] == ["a.md", "b.md"]

            mmr = store.search_mmr(
                [1.0, 0.0, 0.0], top_n=2, candidate_pool=3, mmr_lambda=0.3
            )
            assert [r["file_path"] for r in mmr] == ["a.md", "c.md"]

    def test_mmr_lambda_one_matches_cosine(self):
        """mmr_lambda=1.0 collapses to pure relevance ordering."""
        with TemporaryDirectory() as tmpdir:
            store = self._store(tmpdir)
            mmr = store.search_mmr(
                [1.0, 0.0, 0.0],
                top_n=3,
                candidate_pool=3,
                mmr_lambda=1.0,
                max_per_file=None,
            )
            assert [r["file_path"] for r in mmr] == ["a.md", "b.md", "c.md"]

    def test_mmr_enforces_max_per_file(self):
        """When all chunks come from one file, max_per_file caps the result."""
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [
                    [1.0, 0.0, 0.0],
                    [0.95, 0.05, 0.0],
                    [0.9, 0.1, 0.0],
                ],
                [
                    {"file_path": "same.md", "chunk_index": i, "content": f"C{i}"}
                    for i in range(3)
                ],
            )
            mmr = store.search_mmr(
                [1.0, 0.0, 0.0],
                top_n=5,
                candidate_pool=10,
                mmr_lambda=1.0,
                max_per_file=2,
            )
            assert len(mmr) == 2
            assert all(r["file_path"] == "same.md" for r in mmr)

    def test_mmr_empty_store_returns_empty(self):
        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            assert store.search_mmr([1.0, 0.0, 0.0], top_n=5) == []

    def test_mmr_zero_query_returns_empty(self):
        with TemporaryDirectory() as tmpdir:
            store = self._store(tmpdir)
            assert store.search_mmr([0.0, 0.0, 0.0], top_n=2) == []

    def test_mmr_rerank_seeds_by_cosine_not_input_order(self):
        """mmr_rerank's first pick must be the highest-cosine candidate,
        not whatever the caller put at index 0 of the candidate list.

        Regression guard for the hybrid path, where the input is
        RRF-ranked rather than similarity-ranked.
        """
        with TemporaryDirectory() as tmpdir:
            store = self._store(tmpdir)
            # Pass candidates in *reverse* similarity order (c, b, a).
            # If MMR seeded from input order it would pick c.md first;
            # the cosine-correct seed is a.md.
            candidates = [
                {"file_path": "c.md", "chunk_index": 0,
                 "content": "C", "score": 999.0},
                {"file_path": "b.md", "chunk_index": 0,
                 "content": "B", "score": 100.0},
                {"file_path": "a.md", "chunk_index": 0,
                 "content": "A", "score": 1.0},
            ]
            picked = store.mmr_rerank(
                [1.0, 0.0, 0.0],
                candidates,
                top_n=1,
                mmr_lambda=1.0,
                max_per_file=None,
            )
            assert picked[0]["file_path"] == "a.md"

    def test_search_mmr_with_mask_diversifies_within_scope(self):
        """Masking must leave a real multi-candidate choice for MMR to make,
        not just a pool that happens to equal top_n.

        _reports/scratch.md has the single highest raw cosine similarity of
        all four rows (1.0, a perfect match) but is masked out and must
        never surface. Of the three unmasked survivors, docs/a.md and
        docs/b.md point in nearly the same direction (high mutual cosine),
        while docs/c.md is far more diverse but far less relevant. With
        top_n=2 and mmr_lambda=0.3 (diversity-weighted), the second pick is
        a genuine 2-way argmax between docs/b.md and docs/c.md — and the
        redundancy penalty on docs/b.md (near-duplicate of the already
        selected docs/a.md) flips the choice to docs/c.md, contradicting
        plain cosine order (which would pick docs/a.md, docs/b.md).
        """
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "vectors.pkl")
            store.add(
                [
                    [1.0, 0.0, 0.0],
                    [0.999, 0.045, 0.0],
                    [0.99, 0.1, 0.0],
                    [0.1, 0.0, 1.0],
                ],
                [
                    {"file_path": "_reports/scratch.md", "chunk_index": 0},
                    {"file_path": "docs/a.md", "chunk_index": 0},
                    {"file_path": "docs/b.md", "chunk_index": 0},
                    {"file_path": "docs/c.md", "chunk_index": 0},
                ],
            )
            mask = np.array([False, True, True, True])
            results = store.search_mmr(
                [1.0, 0.0, 0.0],
                top_n=2,
                candidate_pool=4,
                mmr_lambda=0.3,
                max_per_file=2,
                mask=mask,
            )
            assert [r["file_path"] for r in results] == ["docs/a.md", "docs/c.md"]

    def test_search_mmr_mask_all_false_returns_empty(self):
        """A scope filter that excludes every row must return [], not raise
        and not fall back to unmasked results.

        This is the only input class that reaches the ``if not pool: return
        []`` early return in search_mmr — masking makes it a realistic
        caller input (an all-excluding scope filter), not just a defensive
        branch for an organically all-negative-similarity query.
        """
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            store = self._store(tmpdir)
            mask = np.array([False, False, False])
            assert store.search_mmr([1.0, 0.0, 0.0], top_n=2, mask=mask) == []

    def test_search_mmr_mask_length_mismatch_raises(self):
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            store = self._store(tmpdir)
            with pytest.raises(ValueError):
                store.search_mmr([1.0, 0.0, 0.0], mask=np.array([True, False]))


# --- SearchEngine recency reranking tests ---


class TestSearchEngineRecency:
    """SearchEngine blends a recency boost into the final score when enabled."""

    @pytest.fixture
    def engine_with_recency(self, tmp_path):
        """Build a small engine with a fake git_backend serving controlled blame."""
        from datetime import datetime, timedelta, timezone
        from stash_mcp.git_backend import BlameLine

        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        (content_dir / "fresh.md").write_text("authentication content")
        (content_dir / "stale.md").write_text("authentication content")

        now = datetime.now(timezone.utc)

        class FakeGit:
            def blame(self, path):
                ts = now if path == "fresh.md" else now - timedelta(days=720)
                return [
                    BlameLine(
                        line_number=1,
                        commit_hash="abc",
                        author="dev",
                        timestamp=ts,
                        summary="msg",
                        content="authentication content",
                    )
                ]

        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            git_backend=FakeGit(),
            mmr_enabled=False,
            recency_weight=0.0,
            recency_half_life_days=180.0,
        )
        return engine, now

    async def test_recency_off_preserves_semantic_order(self, engine_with_recency):
        """recency_weight=0 leaves ordering driven by semantic score alone."""
        engine, _ = engine_with_recency
        await engine.build_index(["fresh.md", "stale.md"])
        # Both files have identical content → identical embeddings; the
        # order is implementation-defined but blame must NOT swap them
        # when recency is disabled.
        results = await engine.search("authentication", max_results=2)
        scores = [r.score for r in results]
        # Scores equal because content is identical and recency is off.
        assert scores[0] == pytest.approx(scores[1])

    async def test_recency_on_boosts_fresh_over_stale(self, engine_with_recency):
        """With recency_weight>0 and equal semantic scores, fresh wins."""
        engine, _ = engine_with_recency
        engine.recency_weight = 0.5
        await engine.build_index(["fresh.md", "stale.md"])
        results = await engine.search("authentication", max_results=2)
        assert results[0].file_path == "fresh.md"
        assert results[0].score > results[1].score

    async def test_recency_neutral_for_unblamed_files(self, tmp_path):
        """Files the git_backend cannot blame fall back to neutral (0.5) recency."""
        from stash_mcp.git_backend import BlameLine

        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        (content_dir / "tracked.md").write_text("authentication content")
        (content_dir / "untracked.md").write_text("authentication content")

        class PartialGit:
            def blame(self, path):
                if path == "untracked.md":
                    return []
                from datetime import datetime, timedelta, timezone
                return [
                    BlameLine(
                        line_number=1,
                        commit_hash="abc",
                        author="dev",
                        # 5 half-lives old → boost ~0.03
                        timestamp=datetime.now(timezone.utc) - timedelta(days=900),
                        summary="msg",
                        content="authentication content",
                    )
                ]

        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            git_backend=PartialGit(),
            mmr_enabled=False,
            recency_weight=0.5,
            recency_half_life_days=180.0,
        )
        await engine.build_index(["tracked.md", "untracked.md"])
        results = await engine.search("authentication", max_results=2)
        # Untracked file gets neutral 0.5 recency; tracked file gets a
        # near-zero recency (very old). Untracked should rank higher.
        order = [r.file_path for r in results]
        assert order[0] == "untracked.md"


# --- MMR config integration tests ---


class TestSearchEngineMMRConfig:
    """SearchEngine pipeline behaviour with mmr_enabled toggled."""

    async def test_mmr_caps_results_per_file(self, tmp_path):
        """A single file with many overlapping chunks is capped by max_per_file."""
        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        # Generate a long file so it produces many overlapping chunks.
        body = "authentication " * 500
        (content_dir / "auth.md").write_text(body)
        (content_dir / "other.md").write_text("authentication once")

        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            mmr_enabled=True,
            max_per_file=2,
            candidate_pool_multiplier=6,
        )
        await engine.build_index(["auth.md", "other.md"])
        results = await engine.search("authentication", max_results=5)
        auth_hits = [r for r in results if r.file_path == "auth.md"]
        assert len(auth_hits) <= 2

    async def test_mmr_disabled_skips_per_file_cap(self, tmp_path):
        """mmr_enabled=False matches the legacy pipeline (no per-file cap)."""
        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        body = "authentication " * 500
        (content_dir / "auth.md").write_text(body)

        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            mmr_enabled=False,
            max_per_file=2,
        )
        await engine.build_index(["auth.md"])
        results = await engine.search("authentication", max_results=5)
        # No cap when MMR is off — all results may come from the one file.
        assert all(r.file_path == "auth.md" for r in results)
        assert len(results) > 2


# --- BM25Store unit tests ---


class TestBM25Store:
    """BM25Store mirrors VectorStore shape and persists to disk."""

    def test_bm25_search_finds_literal_term(self, tmp_path):
        store = BM25Store(tmp_path / "bm25")
        store.rebuild([
            {"file_path": "a.md", "chunk_index": 0,
             "content": "the transaction manager handles rollback"},
            {"file_path": "b.md", "chunk_index": 0,
             "content": "search engine indexes embeddings nightly"},
        ])
        results = store.search("transaction", top_n=2)
        assert results
        assert results[0][0] == "a.md"

    def test_bm25_empty_query_returns_empty(self, tmp_path):
        store = BM25Store(tmp_path / "bm25")
        store.rebuild([
            {"file_path": "a.md", "chunk_index": 0, "content": "anything"},
        ])
        assert store.search("", top_n=5) == []

    def test_bm25_persists_across_instances(self, tmp_path):
        store_a = BM25Store(tmp_path / "bm25")
        store_a.rebuild([
            {"file_path": "a.md", "chunk_index": 0,
             "content": "authentication and oauth flows"},
        ])
        store_a.save()
        store_b = BM25Store(tmp_path / "bm25")
        assert store_b.count == 1
        assert store_b.search("authentication", top_n=1)

    def test_bm25_clear_wipes_disk(self, tmp_path):
        store = BM25Store(tmp_path / "bm25")
        store.rebuild([
            {"file_path": "a.md", "chunk_index": 0, "content": "alpha"},
        ])
        store.save()
        assert (tmp_path / "bm25" / BM25Store.IDS_FILE).exists()
        store.clear()
        assert not (tmp_path / "bm25" / BM25Store.IDS_FILE).exists()
        reloaded = BM25Store(tmp_path / "bm25")
        assert reloaded.count == 0

    def test_bm25_dirty_flag_resets_on_rebuild(self, tmp_path):
        store = BM25Store(tmp_path / "bm25")
        store.mark_dirty()
        assert store.dirty
        store.rebuild([
            {"file_path": "a.md", "chunk_index": 0, "content": "alpha"},
        ])
        assert not store.dirty


# --- RRF fusion tests ---


class TestRRFFuse:
    """_rrf_fuse combines dense and sparse rankings."""

    def test_rrf_disjoint_lists(self):
        dense = [{"file_path": "a.md", "chunk_index": 0, "content": "A"}]
        sparse = [("b.md", 0, 5.0)]
        fused = _rrf_fuse(dense, sparse, k=60)
        keys = [(r["file_path"], r["chunk_index"]) for r in fused]
        assert ("a.md", 0) in keys
        assert ("b.md", 0) in keys
        # Both at rank 0 in their lists → identical fused score
        assert fused[0]["score"] == pytest.approx(fused[1]["score"])

    def test_rrf_overlap_boosts_shared_item(self):
        """An item ranked in both lists outscores items in only one."""
        dense = [
            {"file_path": "shared.md", "chunk_index": 0, "content": "S"},
            {"file_path": "dense.md", "chunk_index": 0, "content": "D"},
        ]
        sparse = [
            ("shared.md", 0, 9.0),
            ("sparse.md", 0, 4.0),
        ]
        fused = _rrf_fuse(dense, sparse, k=60)
        # shared appears in both lists at rank 0 → should be #1
        assert fused[0]["file_path"] == "shared.md"
        # And it carries the dense metadata (content "S")
        assert fused[0].get("content") == "S"

    def test_rrf_preserves_dense_metadata_when_present(self):
        """When an item is in dense, RRF preserves the dense dict's keys."""
        dense = [{
            "file_path": "a.md",
            "chunk_index": 0,
            "content": "full text",
            "context": "section header",
        }]
        sparse = [("a.md", 0, 3.0)]
        fused = _rrf_fuse(dense, sparse, k=60)
        assert fused[0]["context"] == "section header"
        assert fused[0]["content"] == "full text"

    def test_rrf_sparse_only_returns_stub(self):
        """Sparse-only items get a minimal stub the caller must hydrate."""
        fused = _rrf_fuse([], [("only.md", 7, 2.0)], k=60)
        assert len(fused) == 1
        assert fused[0]["file_path"] == "only.md"
        assert fused[0]["chunk_index"] == 7
        # No 'content' key — caller is expected to look it up
        assert "content" not in fused[0]


# --- Hybrid SearchEngine integration tests ---


class TestHybridSearchEngine:
    """End-to-end hybrid search behavior."""

    @pytest.fixture
    def hybrid_engine(self, tmp_path):
        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        (content_dir / "auth.md").write_text(
            "OAuth2 authentication flow and tokens"
        )
        (content_dir / "db.md").write_text(
            "Database transaction handling and rollback"
        )
        (content_dir / "search.md").write_text(
            "Search engine config STASH_SEARCH_CHUNK_SIZE"
        )
        return SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            hybrid_enabled=True,
            mmr_enabled=True,
        )

    async def test_hybrid_returns_results(self, hybrid_engine):
        engine = hybrid_engine
        await engine.build_index(["auth.md", "db.md", "search.md"])
        results = await engine.search("authentication", max_results=2)
        assert results
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_hybrid_finds_literal_token_dense_misses(self, hybrid_engine):
        """A literal token only the BM25 side knows about still surfaces."""
        engine = hybrid_engine
        # mock_embed doesn't know "STASH_SEARCH_CHUNK_SIZE" — but BM25 does.
        await engine.build_index(["auth.md", "db.md", "search.md"])
        results = await engine.search(
            "STASH_SEARCH_CHUNK_SIZE", max_results=3
        )
        paths = [r.file_path for r in results]
        assert "search.md" in paths

    async def test_reindex_clears_the_bm25_store_too(self, hybrid_engine):
        """``reindex`` cleared ``store`` and ``meta`` but not ``bm25_store``.

        The two indexes must be wiped together or they drift. Deleting every
        file first makes the drift observable: with nothing left to re-index,
        nothing marks BM25 dirty, so nothing triggers the incidental full
        rebuild that otherwise papers over the missing clear.
        """
        engine = hybrid_engine
        await engine.build_index(["auth.md", "db.md", "search.md"])
        assert engine.bm25_store.count > 0

        for name in ("auth.md", "db.md", "search.md"):
            (engine.content_dir / name).unlink()

        await engine.reindex()
        assert engine.store.count == 0
        assert engine.bm25_store.count == 0      # was: stale postings survived

    async def test_hybrid_disabled_is_dense_only(self, tmp_path):
        """hybrid_enabled=False bypasses BM25 entirely."""
        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        (content_dir / "a.md").write_text("authentication content")
        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            hybrid_enabled=False,
        )
        await engine.build_index(["a.md"])
        # BM25 index should never have been built
        assert engine.bm25_store.count == 0
        results = await engine.search("authentication", max_results=1)
        assert results

    async def test_hybrid_dep_missing_at_init_raises(
        self, tmp_path, monkeypatch
    ):
        """Init fails fast when hybrid_enabled but bm25s import fails."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "bm25s":
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(RuntimeError, match="bm25s is required"):
            SearchEngine(
                content_dir=tmp_path / "c",
                index_dir=tmp_path / "i",
                embed_fn=mock_embed,
                hybrid_enabled=True,
            )

    async def test_hybrid_lifecycle_remove_keeps_indexes_consistent(
        self, hybrid_engine
    ):
        """Removing a file drops it from both vector and BM25 indexes."""
        engine = hybrid_engine
        await engine.build_index(["auth.md", "db.md"])
        bm25_before = engine.bm25_store.count
        await engine.remove_file("auth.md")
        # Both indexes shrink — vector store directly, BM25 via rebuild
        assert engine.indexed_chunks < bm25_before
        assert engine.bm25_store.count == engine.indexed_chunks
        # BM25 should no longer return auth content
        sparse = engine.bm25_store.search("authentication", top_n=5)
        paths = [s[0] for s in sparse]
        assert "auth.md" not in paths

    async def test_hybrid_upgrade_path_rebuilds_bm25_from_vectors(
        self, tmp_path
    ):
        """Pre-existing vectors.pkl without BM25 → BM25 rebuilt on init."""
        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        (content_dir / "a.md").write_text("authentication content")
        # First engine: hybrid OFF so BM25 is never built/saved.
        engine_legacy = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            hybrid_enabled=False,
        )
        await engine_legacy.build_index(["a.md"])
        assert engine_legacy.bm25_store.count == 0  # never built

        # Second engine: hybrid ON, no bm25 index on disk yet → rebuild.
        engine_hybrid = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embed_fn=mock_embed,
            hybrid_enabled=True,
        )
        assert engine_hybrid.bm25_store.count == engine_hybrid.indexed_chunks
        assert engine_hybrid.bm25_store.count > 0

    async def test_hybrid_embedder_change_clears_both_indexes(self, tmp_path):
        """Changing embedder_model wipes vector AND bm25 indexes."""
        content_dir = tmp_path / "content"
        index_dir = tmp_path / "index"
        content_dir.mkdir()
        index_dir.mkdir()
        (content_dir / "a.md").write_text("authentication content")

        engine = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embedder_model="model-A",
            embed_fn=mock_embed,
            hybrid_enabled=True,
        )
        await engine.build_index(["a.md"])
        assert engine.indexed_chunks > 0
        assert engine.bm25_store.count > 0

        # New engine with different model — both indexes should be cleared.
        engine2 = SearchEngine(
            content_dir=content_dir,
            index_dir=index_dir,
            embedder_model="model-B",
            embed_fn=mock_embed,
            hybrid_enabled=True,
        )
        assert engine2.indexed_chunks == 0
        assert engine2.bm25_store.count == 0

    async def test_hybrid_filters_sparse_candidates(self, hybrid_engine):
        root = hybrid_engine.content_dir
        (root / "_reports").mkdir(exist_ok=True)
        (root / "_reports" / "r.md").write_text("# R\n\nzebra zebra zebra")
        (root / "keep.md").write_text("# K\n\nzebra")
        await hybrid_engine.reindex()
        results = await hybrid_engine.search(
            "zebra", max_results=5, exclude_patterns=["**/_reports/**"],
        )
        paths = [r.file_path for r in results]
        assert "keep.md" in paths
        assert "_reports/r.md" not in paths


# --- SearchEngine.search scoping, exclusions, metadata filters, boost ---


class TestSearchScoping:

    @pytest.fixture
    def scoped_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "_reports").mkdir()
            (root / "openpilot" / "_reports").mkdir(parents=True)
            (root / "openpilot" / "services").mkdir(parents=True)
            (root / "_reports" / "scan.md").write_text("# Scan\n\nauth auth auth oauth flow")
            (root / "openpilot" / "_reports" / "delta.md").write_text(
                "# Delta\n\nauth oauth flow flow"
            )
            (root / "openpilot" / "services" / "auth.md").write_text(
                "---\nlayer: frogpilot\n---\n# Auth service\n\nauth oauth flow"
            )
            (root / "openpilot" / "services" / "notes.md").write_text(
                "---\nlayer: upstream\n---\n# Notes\n\nauth flow"
            )
            engine = SearchEngine(
                content_dir=root, index_dir=Path(index_dir), embed_fn=mock_embed,
                default_exclude_patterns=["**/_reports/**"],
            )
            yield engine

    async def _paths(self, engine, **kw):
        return [r.file_path for r in await engine.search("auth oauth flow", max_results=10, **kw)]

    async def test_default_exclusion_hides_reports_at_any_depth(self, scoped_engine):
        await scoped_engine.reindex()
        paths = await self._paths(scoped_engine)
        assert paths
        assert not any("_reports/" in p for p in paths)

    async def test_include_excluded_readmits_defaults_only(self, scoped_engine):
        await scoped_engine.reindex()
        paths = await self._paths(scoped_engine, include_excluded=True)
        assert any(p.startswith("_reports/") for p in paths)
        paths = await self._paths(
            scoped_engine, include_excluded=True, exclude_patterns=["_reports/**"],
        )
        assert not any(p.startswith("_reports/") for p in paths)
        assert any(p.startswith("openpilot/_reports/") for p in paths)

    async def test_path_prefix_scopes_to_subtree(self, scoped_engine):
        await scoped_engine.reindex()
        paths = await self._paths(scoped_engine, path_prefix="openpilot/services")
        assert paths and all(p.startswith("openpilot/services/") for p in paths)

    async def test_metadata_filter(self, scoped_engine):
        await scoped_engine.reindex()
        paths = await self._paths(scoped_engine, metadata_filters={"layer": "frogpilot"})
        assert paths == ["openpilot/services/auth.md"]

    async def test_max_results_holds_under_exclusion(self, scoped_engine):
        await scoped_engine.reindex()
        results = await scoped_engine.search("auth oauth flow", max_results=2)
        assert len(results) == 2
        assert not any("_reports/" in r.file_path for r in results)

    async def test_file_types_still_filters(self, scoped_engine):
        (scoped_engine.content_dir / "config.py").write_text("# auth\nAUTH = 1\n")
        await scoped_engine.reindex()
        paths = await self._paths(scoped_engine, file_types=[".py"])
        assert paths == ["config.py"]

    async def test_multiple_path_prefixes(self, scoped_engine):
        root = scoped_engine.content_dir
        (root / "systems" / "homelab").mkdir(parents=True)
        (root / "systems" / "homelab" / "ops.md").write_text("# Ops\n\nauth oauth flow")
        await scoped_engine.reindex()
        paths = await self._paths(
            scoped_engine, path_prefix="openpilot/services,systems/homelab",
        )
        assert set(paths) == {
            "openpilot/services/auth.md", "openpilot/services/notes.md", "systems/homelab/ops.md",
        }
        paths = await self._paths(scoped_engine, path_prefix=["systems/homelab/"])
        assert paths == ["systems/homelab/ops.md"]


class TestSearchBoost:
    """Soft scoping: boosted prefixes rank first but nothing is hidden."""

    @pytest.fixture
    def boost_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "projects" / "a").mkdir(parents=True)
            (root / "projects" / "b").mkdir(parents=True)
            # b's doc is the better lexical match; a's doc is what the agent works in
            (root / "projects" / "b" / "strong.md").write_text("# B\n\nauth oauth flow oauth")
            (root / "projects" / "a" / "weak.md").write_text("# A\n\nauth flow")
            (root / "projects" / "a" / "other.md").write_text("# A2\n\nauth")
            engine = SearchEngine(
                content_dir=root, index_dir=Path(index_dir), embed_fn=mock_embed,
                default_boost_weight=0.5,
            )
            yield engine

    async def test_boost_reorders_without_hiding(self, boost_engine):
        await boost_engine.reindex()
        plain = [r.file_path for r in await boost_engine.search("auth oauth flow", max_results=3)]
        assert plain[0] == "projects/b/strong.md"
        boosted = await boost_engine.search(
            "auth oauth flow", max_results=3, boost_prefixes="projects/a",
        )
        paths = [r.file_path for r in boosted]
        assert paths[0].startswith("projects/a/")
        assert "projects/b/strong.md" in paths            # still present

    async def test_boost_zero_matches_plain_order(self, boost_engine):
        await boost_engine.reindex()
        plain = [r.file_path for r in await boost_engine.search("auth oauth flow", max_results=3)]
        zero = [
            r.file_path
            for r in await boost_engine.search(
                "auth oauth flow", max_results=3, boost_prefixes=["projects/a"], boost_weight=0.0,
            )
        ]
        assert zero == plain

    async def test_boost_pulls_in_scoped_doc_missing_from_general_pool(self, boost_engine):
        root = boost_engine.content_dir
        for i in range(30):   # flood the general pool with strong matches elsewhere
            (root / "projects" / "b" / f"flood{i}.md").write_text("# F\n\nauth oauth flow oauth")
        await boost_engine.reindex()
        results = await boost_engine.search(
            "auth oauth flow", max_results=2, boost_prefixes="projects/a",
        )
        assert any(r.file_path.startswith("projects/a/") for r in results)

    async def test_boost_never_readmits_default_exclusions(self, boost_engine):
        boost_engine.default_exclude_patterns = ["**/projects/a/**"]
        await boost_engine.reindex()
        results = await boost_engine.search(
            "auth oauth flow", max_results=5, boost_prefixes="projects/a",
        )
        assert not any(r.file_path.startswith("projects/a/") for r in results)


class TestSearchMetadataFilterKeyNormalization:
    """Caller metadata_filters keys are normalized the same way the index side is.

    frontmatter.extract_metadata (Task 2) normalizes frontmatter keys before
    they're stored as chunk metadata (Task 6). Without normalizing the
    caller's filter keys the same way, a filter like {"Describes": ...}
    would silently match nothing against the stored "describes" key —
    "silently matches nothing" being worse than an error.
    """

    @pytest.fixture
    def metadata_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "doc.md").write_text(
                "---\ndescribes: openpilot\nlast_verified: 2026-08-16\n---\n"
                "# Doc\n\nauth oauth flow"
            )
            (root / "other.md").write_text("# Other\n\nauth oauth flow")
            engine = SearchEngine(
                content_dir=root, index_dir=Path(index_dir), embed_fn=mock_embed,
            )
            yield engine

    async def test_title_case_key_matches_normalized_stored_key(self, metadata_engine):
        await metadata_engine.reindex()
        results = await metadata_engine.search(
            "auth oauth flow", max_results=10, metadata_filters={"Describes": "openpilot"},
        )
        assert [r.file_path for r in results] == ["doc.md"]

    async def test_hyphenated_key_matches_normalized_stored_key(self, metadata_engine):
        await metadata_engine.reindex()
        results = await metadata_engine.search(
            "auth oauth flow",
            max_results=10,
            metadata_filters={"last-verified": "2026-08-16"},
        )
        assert [r.file_path for r in results] == ["doc.md"]


# --- Fix round 1 regressions: boost vs. max_per_file, boost vs. negative scores ---


class TestSearchBoostMaxPerFileCap:
    """Regression: boost_prefixes must not let the union of two independently
    -capped MMR passes (the general pool and the boost-scoped pool) exceed
    max_per_file.

    mock_embed can't reproduce this: its keyword-count vectors are
    non-negative and never happen to make the two independent MMR passes
    (general vs. boost-scoped) disagree on which chunks of a file to keep,
    for any of this file's fixtures. This uses a hand-crafted embed_fn whose
    vectors were verified (by direct experimentation against
    VectorStore.search_mmr, see the fix-round-1 report) to make the two
    passes independently select DIFFERENT chunks of the same 3-chunk file
    under the engine's DEFAULT mmr_lambda=0.7 and max_per_file=2 — the
    "general" pass picks chunks {2, 1} while the pass scoped to just this
    file's own 3 chunks picks {2, 0}; naive union = all 3.
    """

    @staticmethod
    async def _cap_repro_embed(texts: list[str]) -> list[list[float]]:
        # Vectors are 3-D and hand-verified (not derived from the text
        # content — this is a pure lookup keyed by a marker substring) to
        # reproduce the max_per_file violation under mmr_lambda=0.7.
        table = {
            "QUERYMARK": [1.0, 0.0, 0.0],
            "ZERO": [0.1300602624868516, -0.00949558973049732, -0.9914606204471872],
            "ONE": [0.41487906317265155, -0.7756009301561246, 0.47572950306023404],
            "TWO": [0.8362323236972545, -0.07510504497355922, 0.5432078175278867],
            "NOISE": [0.22063688266072035, -0.12463224371628472, -0.9673604136184217],
        }
        out = []
        for t in texts:
            for tag, vec in table.items():
                if tag in t:
                    out.append(vec)
                    break
            else:
                out.append([0.01, 0.0, 0.0])
        return out

    @pytest.fixture
    def cap_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "boosted").mkdir()
            (root / "noise").mkdir()
            # chunk_size=30/overlap=0 makes each 30-char block below its own
            # chunk with no boundary drift; everything else about the engine
            # (mmr_enabled, hybrid_enabled, mmr_lambda, max_per_file) is left
            # at its DEFAULT, since the cap contract is scoped to those defaults.
            block0 = ("ZERO" * 8)[:30]
            block1 = ("ONE-" * 8)[:30]
            block2 = ("TWO-" * 8)[:30]
            (root / "boosted" / "multi.md").write_text(block0 + block1 + block2)
            (root / "noise" / "other.md").write_text("NOISE" * 6)
            engine = SearchEngine(
                content_dir=root,
                index_dir=Path(index_dir),
                embed_fn=self._cap_repro_embed,
                chunk_size=30,
                chunk_overlap=0,
            )
            yield engine

    async def test_boost_respects_max_per_file_cap(self, cap_engine):
        await cap_engine.reindex()
        # Sanity: the file really did produce 3 distinct indexed chunks —
        # otherwise this test would trivially pass for the wrong reason.
        assert cap_engine.meta.chunk_counts["boosted/multi.md"] == 3

        results = await cap_engine.search(
            "QUERYMARK", max_results=10, boost_prefixes="boosted",
        )
        boosted_chunks = [r for r in results if r.file_path == "boosted/multi.md"]
        assert len(boosted_chunks) <= 2, (
            f"max_per_file=2 violated: got {len(boosted_chunks)} chunks from "
            f"boosted/multi.md: {sorted(r.chunk_index for r in boosted_chunks)}"
        )
        # The boost must still be able to rescue this file into the result
        # set at all — the cap fix must not turn boosting into a no-op.
        assert boosted_chunks


class TestSearchBoostNegativeScore:
    """Regression: the boost multiplier must never make a boosted result
    rank *worse* by multiplying a negative score further into the negative.

    Only mmr_rerank (used in the hybrid + MMR path) can surface a negative
    'score': search()/search_mmr() both explicitly drop non-positive cosine
    similarities before a candidate is ever selectable, but mmr_rerank
    overwrites 'score' with the raw cosine of whatever candidates hybrid
    fusion handed it — including a BM25-rescued, dense-irrelevant candidate,
    which is exactly the class of document hybrid mode exists to surface.
    mock_embed cannot produce a negative similarity (its vectors are
    non-negative keyword counts), so this uses a hand-crafted embed_fn.
    """

    @staticmethod
    async def _neg_embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t == "findme":                 # the literal query string
                out.append([1.0, 0.0])
            elif "NEGDOC" in t:                # the chunk (also contains
                out.append([-1.0, 0.0])        # "findme" for BM25 to find it)
            else:
                out.append([0.01, 0.0])
        return out

    @pytest.fixture
    def neg_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "boosted").mkdir()
            # Contains "findme" (so BM25 finds it via literal term overlap)
            # and "NEGDOC" (a marker so the embed_fn can hand it a vector
            # that is exactly opposite the query's — cosine similarity -1.0).
            (root / "boosted" / "neg.md").write_text("findme NEGDOC findme findme")
            engine = SearchEngine(
                content_dir=root,
                index_dir=Path(index_dir),
                embed_fn=self._neg_embed,
                hybrid_enabled=True,
                default_boost_weight=0.5,   # matches the reviewer's -1.0 -> -1.5 repro ratio
            )
            yield engine

    async def test_boost_never_demotes_negative_score_candidate(self, neg_engine):
        await neg_engine.reindex()

        unboosted = await neg_engine.search("findme", max_results=5)
        assert unboosted
        assert unboosted[0].score < 0, "fixture must exercise the negative-score path"

        boosted = await neg_engine.search(
            "findme", max_results=5, boost_prefixes="boosted",
        )
        assert boosted
        # Chosen fix: only positive scores are eligible for the multiplier,
        # so a negative score must come through completely unchanged — not
        # just "not worse", but byte-identical to the unboosted run.
        assert boosted[0].score == unboosted[0].score
        assert boosted[0].score >= unboosted[0].score  # never demoted (the reported symptom)


class TestSearchBoostCombinations:
    """Combinations the fix-round-1 review flagged as untested."""

    @pytest.fixture
    def combo_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "projects" / "a").mkdir(parents=True)
            (root / "projects" / "b").mkdir(parents=True)
            (root / "outside").mkdir(parents=True)
            (root / "projects" / "b" / "strong.md").write_text("# B\n\nauth oauth flow oauth")
            (root / "projects" / "a" / "weak.md").write_text("# A\n\nauth flow")
            # Strongest lexical match of all three, but outside the hard
            # path_prefix scope — proves the hard filter really excludes it
            # rather than it just losing on relevance.
            (root / "outside" / "doc.md").write_text(
                "# O\n\nauth oauth flow oauth auth oauth flow"
            )
            engine = SearchEngine(
                content_dir=root, index_dir=Path(index_dir), embed_fn=mock_embed,
                default_boost_weight=0.5,
            )
            yield engine

    async def test_path_prefix_and_boost_prefixes_compose_as_and(self, combo_engine):
        await combo_engine.reindex()
        results = await combo_engine.search(
            "auth oauth flow",
            max_results=10,
            path_prefix="projects",
            boost_prefixes="projects/a",
        )
        paths = [r.file_path for r in results]
        assert paths
        # Hard filter: the stronger outside/ match never appears.
        assert all(p.startswith("projects/") for p in paths)
        # Soft boost still reorders within the allowed scope.
        assert paths[0].startswith("projects/a/")
        assert "projects/b/strong.md" in paths  # still present, just not first


class TestSearchBoostHybrid:
    """hybrid_enabled=True combined with boost_prefixes — previously untested
    in any form (not just the negative-score edge case above).
    """

    @pytest.fixture
    def hybrid_boost_engine(self):
        with TemporaryDirectory() as content_dir, TemporaryDirectory() as index_dir:
            root = Path(content_dir)
            (root / "projects" / "a").mkdir(parents=True)
            (root / "projects" / "b").mkdir(parents=True)
            (root / "projects" / "b" / "strong.md").write_text("# B\n\nauth oauth flow oauth")
            (root / "projects" / "a" / "weak.md").write_text("# A\n\nauth flow")
            engine = SearchEngine(
                content_dir=root, index_dir=Path(index_dir), embed_fn=mock_embed,
                hybrid_enabled=True, default_boost_weight=0.5,
            )
            yield engine

    async def test_boost_reorders_in_hybrid_mode(self, hybrid_boost_engine):
        await hybrid_boost_engine.reindex()
        plain = [
            r.file_path
            for r in await hybrid_boost_engine.search("auth oauth flow", max_results=3)
        ]
        assert plain[0] == "projects/b/strong.md"

        boosted = await hybrid_boost_engine.search(
            "auth oauth flow", max_results=3, boost_prefixes="projects/a",
        )
        paths = [r.file_path for r in boosted]
        assert paths[0].startswith("projects/a/")
        assert "projects/b/strong.md" in paths
