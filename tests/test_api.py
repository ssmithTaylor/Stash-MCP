"""Tests for REST API."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from stash_mcp.api import create_api
from stash_mcp.events import _listeners, add_listener
from stash_mcp.filesystem import FileSystem
from stash_mcp.transactions import TransactionManager


@pytest.fixture
def test_client():
    """Create a test client with temporary filesystem."""
    with TemporaryDirectory() as tmpdir:
        fs = FileSystem(Path(tmpdir))
        # Add some test content
        fs.write_file("test.md", "# Test Content")
        fs.write_file("docs/readme.md", "# README")

        app = create_api(fs)
        client = TestClient(app)
        yield client


@pytest.fixture
def event_listener():
    """Register and return a mock event listener, cleaning up after test."""
    listener = MagicMock()
    add_listener(listener)
    yield listener
    _listeners.remove(listener)


def test_root_endpoint(test_client):
    """Test root endpoint returns API info."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Stash-MCP API"
    assert "endpoints" in data


def test_list_content(test_client):
    """Test listing root directory contents (shallow by default)."""
    response = test_client.get("/api/content")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    paths = [item["path"] for item in data["items"]]
    assert "test.md" in paths
    assert "docs" in paths  # directory, not its contents


def test_list_content_recursive(test_client):
    """Test listing all content recursively."""
    response = test_client.get("/api/content", params={"recursive": "true"})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    paths = [item["path"] for item in data["items"]]
    assert "test.md" in paths
    assert "docs/readme.md" in paths


def test_read_content(test_client):
    """Test reading a specific file."""
    response = test_client.get("/api/content/test.md")
    assert response.status_code == 200
    data = response.json()
    assert data["path"] == "test.md"
    assert data["content"] == "# Test Content"
    assert data["is_directory"] is False


def test_read_content_has_mime_type_and_updated_at(test_client):
    """Test reading a file returns mime_type and updated_at."""
    response = test_client.get("/api/content/test.md")
    assert response.status_code == 200
    data = response.json()
    assert data["mime_type"] == "text/markdown"
    assert data["updated_at"] is not None


def test_read_nonexistent_content(test_client):
    """Test reading a nonexistent file returns 404."""
    response = test_client.get("/api/content/nonexistent.md")
    assert response.status_code == 404


def test_create_content(test_client):
    """Test creating new content."""
    response = test_client.put(
        "/api/content/new.md",
        json={"content": "# New Content"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "saved successfully" in data["message"]

    # Verify content was created
    response = test_client.get("/api/content/new.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# New Content"


def test_post_create_content_nested_path(test_client):
    """Test POST creates a file in a nested directory that doesn't exist."""
    response = test_client.post(
        "/api/content/a/b/c/nested.md",
        json={"content": "# Nested Content"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "created successfully" in data["message"]

    # Verify content was created
    response = test_client.get("/api/content/a/b/c/nested.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# Nested Content"


def test_put_create_content_nested_path(test_client):
    """Test PUT creates a file in a nested directory that doesn't exist."""
    response = test_client.put(
        "/api/content/x/y/z/deep.md",
        json={"content": "# Deep Content"}
    )
    assert response.status_code == 200

    # Verify content was created
    response = test_client.get("/api/content/x/y/z/deep.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# Deep Content"


def test_patch_move_to_nested_path(test_client):
    """Test PATCH moves a file to a nested directory that doesn't exist."""
    response = test_client.patch(
        "/api/content/test.md",
        json={"destination": "new/nested/dir/moved.md"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["destination"] == "new/nested/dir/moved.md"

    # Verify old path is gone, new path exists
    response = test_client.get("/api/content/test.md")
    assert response.status_code == 404
    response = test_client.get("/api/content/new/nested/dir/moved.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# Test Content"


def test_update_content(test_client):
    """Test updating existing content."""
    response = test_client.put(
        "/api/content/test.md",
        json={"content": "# Updated Content"}
    )
    assert response.status_code == 200

    # Verify content was updated
    response = test_client.get("/api/content/test.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# Updated Content"


def test_delete_content(test_client):
    """Test deleting content."""
    response = test_client.delete("/api/content/test.md")
    assert response.status_code == 200
    data = response.json()
    assert "deleted successfully" in data["message"]

    # Verify content was deleted
    response = test_client.get("/api/content/test.md")
    assert response.status_code == 404


def test_delete_nonexistent_content(test_client):
    """Test deleting a nonexistent file returns 404."""
    response = test_client.delete("/api/content/nonexistent.md")
    assert response.status_code == 404


# --- New endpoint tests ---


def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_tree_endpoint(test_client):
    """Test directory tree endpoint returns nested structure."""
    response = test_client.get("/api/tree")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "root"
    assert data["type"] == "directory"
    assert data["children"] is not None
    # Should have "docs" dir and "test.md" file
    names = [c["name"] for c in data["children"]]
    assert "docs" in names
    assert "test.md" in names
    # Check nested structure
    docs_node = next(c for c in data["children"] if c["name"] == "docs")
    assert docs_node["type"] == "directory"
    assert any(c["name"] == "readme.md" for c in docs_node["children"])


def test_post_create_content(test_client):
    """Test POST creates a new file with 201 status."""
    response = test_client.post(
        "/api/content/created.md",
        json={"content": "# Created via POST"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "created successfully" in data["message"]

    # Verify content was created
    response = test_client.get("/api/content/created.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# Created via POST"


def test_post_create_conflict(test_client):
    """Test POST returns 409 when file already exists."""
    response = test_client.post(
        "/api/content/test.md",
        json={"content": "# Conflict"}
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_patch_move_content(test_client):
    """Test PATCH moves/renames a file."""
    response = test_client.patch(
        "/api/content/test.md",
        json={"destination": "renamed.md"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "test.md"
    assert data["destination"] == "renamed.md"

    # Verify old path is gone, new path exists
    response = test_client.get("/api/content/test.md")
    assert response.status_code == 404
    response = test_client.get("/api/content/renamed.md")
    assert response.status_code == 200
    assert response.json()["content"] == "# Test Content"


def test_patch_move_nonexistent(test_client):
    """Test PATCH returns 404 for nonexistent source."""
    response = test_client.patch(
        "/api/content/nonexistent.md",
        json={"destination": "other.md"}
    )
    assert response.status_code == 404


def test_patch_move_conflict(test_client):
    """Test PATCH returns 409 when destination already exists."""
    response = test_client.patch(
        "/api/content/test.md",
        json={"destination": "docs/readme.md"}
    )
    assert response.status_code == 409


def test_list_content_with_file_type_filter(test_client):
    """Test listing content filtered by file type."""
    # First create a .txt file
    test_client.put("/api/content/notes.txt", json={"content": "notes"})

    response = test_client.get("/api/content", params={"file_type": ".md", "recursive": "true"})
    assert response.status_code == 200
    data = response.json()
    paths = [item["path"] for item in data["items"]]
    assert len(paths) >= 1  # At least one .md file exists
    assert all(p.endswith(".md") for p in paths)
    assert "notes.txt" not in paths


def test_list_content_directory_with_path(test_client):
    """Test listing a specific directory's contents."""
    response = test_client.get("/api/content", params={"path": "docs", "recursive": "false"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["path"] == "docs/readme.md"


def test_list_content_items_have_mime_type(test_client):
    """Test that listed file items include mime_type."""
    response = test_client.get("/api/content", params={"recursive": "true"})
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        if not item["is_directory"]:
            assert item["mime_type"] is not None


def test_path_traversal_post(test_client):
    """Test path traversal is blocked on POST."""
    # FastAPI normalizes URL paths with ../, so traversal is blocked at HTTP level
    response = test_client.post(
        "/api/content/../../../etc/passwd",
        json={"content": "malicious"}
    )
    # Path traversal blocked either by URL normalization (404) or filesystem validation (400)
    assert response.status_code in (400, 404)


def test_path_traversal_patch(test_client):
    """Test path traversal is blocked on PATCH destination."""
    response = test_client.patch(
        "/api/content/test.md",
        json={"destination": "../../../etc/passwd"}
    )
    assert response.status_code == 400


def test_event_emitted_on_post_create(test_client, event_listener):
    """Test that POST create emits an event."""
    test_client.post("/api/content/evented.md", json={"content": "event test"})
    event_listener.assert_called_once()
    args = event_listener.call_args[0]
    assert args[0] == "content_created"
    assert args[1] == "evented.md"


def test_event_emitted_on_put_update(test_client, event_listener):
    """Test that PUT update emits an event."""
    test_client.put("/api/content/test.md", json={"content": "updated"})
    event_listener.assert_called_once()
    args = event_listener.call_args[0]
    assert args[0] == "content_updated"
    assert args[1] == "test.md"


def test_event_emitted_on_delete(test_client, event_listener):
    """Test that DELETE emits an event."""
    test_client.delete("/api/content/test.md")
    event_listener.assert_called_once()
    args = event_listener.call_args[0]
    assert args[0] == "content_deleted"
    assert args[1] == "test.md"


def test_event_emitted_on_patch_move(test_client, event_listener):
    """Test that PATCH move emits an event."""
    test_client.patch("/api/content/test.md", json={"destination": "moved.md"})
    event_listener.assert_called_once()
    args = event_listener.call_args[0]
    kwargs = event_listener.call_args[1]
    assert args[0] == "content_moved"
    assert args[1] == "moved.md"
    assert kwargs.get("source_path") == "test.md"


# --- REST autocommit / write-lock tests ---


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@example.com"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"],
                   check=True, capture_output=True)
    (path / "seed.md").write_text("seed")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"],
                   check=True, capture_output=True)


class TestRestAutocommit:
    def _client(self, tmpdir: Path):
        from stash_mcp.git_backend import GitBackend

        _init_git_repo(tmpdir)
        fs = FileSystem(tmpdir)
        git = GitBackend(tmpdir)
        tm = TransactionManager(fs, git, autocommit=True, author_default="Bot <bot@x>")
        return TestClient(create_api(fs, transaction_manager=tm))

    def _last_commit(self, tmpdir: Path) -> str:
        return subprocess.run(
            ["git", "-C", str(tmpdir), "show", "--name-only", "--format=%s%n%an", "HEAD"],
            capture_output=True, text=True,
        ).stdout

    def test_post_put_patch_delete_commit(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            client = self._client(path)
            r = client.post("/api/content/a.md", json={"content": "A"},
                            params={"commit_message": "made a", "author": "Me <me@x>"})
            assert r.status_code == 201 and r.json()["commit"]
            assert "made a" in self._last_commit(path) and "Me" in self._last_commit(path)

            r = client.put("/api/content/a.md", json={"content": "A2"})
            assert r.status_code == 200 and r.json()["commit"]
            assert "Update a.md" in self._last_commit(path)

            r = client.patch("/api/content/a.md", json={"destination": "b.md"})
            assert r.status_code == 200 and r.json()["commit"]
            assert "Move a.md -> b.md" in self._last_commit(path)

            r = client.delete("/api/content/b.md")
            assert r.status_code == 200 and r.json()["commit"]
            assert "Delete b.md" in self._last_commit(path)

    def test_put_without_change_has_null_commit(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            client = self._client(path)
            r = client.put("/api/content/seed.md", json={"content": "seed"})
            assert r.status_code == 200 and r.json()["commit"] is None

    def test_no_manager_keeps_old_shape(self):
        with TemporaryDirectory() as tmp:
            client = TestClient(create_api(FileSystem(Path(tmp))))
            r = client.post("/api/content/x.md", json={"content": "x"})
            assert r.status_code == 201 and r.json().get("commit") is None

    def test_rest_commits_even_with_autocommit_off(self):
        """REST callers pass session_id=None, so they skip the transaction gate
        and commit immediately even in the default STASH_GIT_AUTOCOMMIT=false
        deployment. That is a real behavior change for existing
        tracking=true/autocommit=false servers — whose REST writes used to
        leave the working tree dirty — so it is pinned here rather than only
        at manager level."""
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            _init_git_repo(path)
            fs = FileSystem(path)
            tm = TransactionManager(
                fs, GitBackend(path), autocommit=False, author_default="Bot <bot@x>"
            )
            client = TestClient(create_api(fs, transaction_manager=tm))

            r = client.post("/api/content/gated.md", json={"content": "G"})
            assert r.status_code == 201 and r.json()["commit"]
            assert "Create gated.md" in self._last_commit(path)

            r = client.put("/api/content/gated.md", json={"content": "G2"})
            assert r.status_code == 200 and r.json()["commit"]
            assert "Update gated.md" in self._last_commit(path)

            assert tm.open_transaction_count == 0        # no transaction was involved
            assert not tm.git.has_staged_changes()       # index invariant holds

    def test_rest_rejects_a_malformed_author(self):
        """`author` is an unauthenticated query parameter; a bare name would
        make `git commit --author` exit 128 after the file is already on disk."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            client = self._client(path)
            r = client.post("/api/content/bad.md", json={"content": "B"},
                            params={"author": "doc-writer"})
            assert r.status_code == 500
            assert "Invalid author" in r.json()["detail"]
            assert "bad.md" not in self._last_commit(path)
