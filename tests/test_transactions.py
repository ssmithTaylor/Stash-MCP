"""Tests for TransactionManager and git backend transaction methods."""

import asyncio
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stash_mcp.filesystem import FileSystem
from stash_mcp.transactions import TransactionError, TransactionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    """Initialise a bare git repo at *path* with a single commit."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "Initial commit"],
        check=True,
        capture_output=True,
    )


def _make_tm(tmpdir: Path) -> tuple[TransactionManager, FileSystem]:
    """Create a TransactionManager backed by a real git repo."""
    from stash_mcp.git_backend import GitBackend

    _init_repo(tmpdir)
    fs = FileSystem(tmpdir)
    git = GitBackend(tmpdir)
    return TransactionManager(fs, git), fs


# ---------------------------------------------------------------------------
# GitBackend — commit / reset_hard / push
# ---------------------------------------------------------------------------


class TestGitBackendNewMethods:
    def test_commit_stages_and_commits(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            from stash_mcp.git_backend import GitBackend

            git = GitBackend(path)
            (path / "new.txt").write_text("hello")
            git.commit("Add new.txt")
            result = subprocess.run(
                ["git", "-C", tmpdir, "log", "--oneline"],
                capture_output=True,
                text=True,
            )
            assert "Add new.txt" in result.stdout

    def test_commit_with_author(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            from stash_mcp.git_backend import GitBackend

            git = GitBackend(path)
            (path / "authored.txt").write_text("authored content")
            git.commit("Add authored file", author="Custom Author <custom@example.com>")
            result = subprocess.run(
                ["git", "-C", tmpdir, "log", "--format=%an <%ae>", "-1"],
                capture_output=True,
                text=True,
            )
            assert "Custom Author" in result.stdout
            assert "custom@example.com" in result.stdout

    def test_reset_hard_discards_changes(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            from stash_mcp.git_backend import GitBackend

            git = GitBackend(path)
            (path / "README.md").write_text("changed content")
            git.reset_hard()
            assert (path / "README.md").read_text() == "# Test\n"

    def test_commit_raises_on_nothing_to_commit(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            from stash_mcp.git_backend import GitBackend

            git = GitBackend(path)
            # Nothing changed — commit should fail
            with pytest.raises(RuntimeError, match="git commit failed"):
                git.commit("Empty commit")

    def test_push_raises_on_no_remote(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            from stash_mcp.git_backend import GitBackend

            git = GitBackend(path)
            with pytest.raises(RuntimeError, match="git push failed"):
                git.push("nonexistent-remote", "main")

    def _log(self, path: Path, *fmt: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), "log", *fmt], capture_output=True, text=True,
        ).stdout

    def test_commit_paths_commits_only_the_given_paths(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "a.md").write_text("A")
            (path / "b.md").write_text("B")           # dirty but not part of the commit
            short = git.commit_paths(["a.md"], "Add a", author="Agent A <a@example.com>")
            assert short and len(short) >= 7
            names = subprocess.run(
                ["git", "-C", tmpdir, "show", "--name-only", "--format=%an", "HEAD"],
                capture_output=True, text=True,
            ).stdout
            assert "Agent A" in names and "a.md" in names and "b.md" not in names
            status = subprocess.run(
                ["git", "-C", tmpdir, "status", "--porcelain"], capture_output=True, text=True,
            ).stdout
            assert "?? b.md" in status                # b.md still untracked, not staged
            assert not git.has_staged_changes()

    def test_commit_paths_records_deletion_and_move(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "old.md").write_text("x")
            git.commit_paths(["old.md"], "Add old")
            (path / "old.md").rename(path / "new.md")
            assert git.commit_paths(["old.md", "new.md"], "Move old -> new")
            (path / "new.md").unlink()
            assert git.commit_paths(["new.md"], "Delete new")
            assert "Delete new" in self._log(path, "--oneline")
            tracked = subprocess.run(
                ["git", "-C", tmpdir, "ls-files"], capture_output=True, text=True,
            ).stdout.split()
            assert "new.md" not in tracked and "old.md" not in tracked

    def test_commit_paths_noop_when_nothing_changed_or_path_unknown(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            assert git.commit_paths(["README.md"], "no change") is None
            assert git.commit_paths(["never-existed.md"], "ghost") is None
            assert git.commit_paths([], "empty") is None

    def test_commit_paths_does_not_sweep_in_unrelated_staged_changes(self):
        """Regression test: ``git commit`` with no trailing pathspec commits the
        *entire* index, not just the paths this call just staged. A concurrent
        writer's own staged-but-uncommitted path must survive untouched."""
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "a.md").write_text("A")
            (path / "other.md").write_text("OTHER")
            # Simulate a second, concurrent session that staged its own change
            # but hasn't committed yet.
            subprocess.run(
                ["git", "-C", tmpdir, "add", "other.md"], check=True, capture_output=True,
            )
            short = git.commit_paths(["a.md"], "Add a")
            assert short
            names = subprocess.run(
                ["git", "-C", tmpdir, "show", "--name-only", "--format=", "HEAD"],
                capture_output=True, text=True,
            ).stdout
            assert "a.md" in names and "other.md" not in names
            assert git.has_staged_changes(["other.md"])      # untouched: still staged
            assert not git.has_staged_changes(["a.md"])       # committed: index clean

    def test_commit_paths_unstages_on_commit_failure(self):
        """A malformed --author makes ``git commit`` fail deterministically;
        the index invariant requires the just-staged paths to be unstaged."""
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "b.md").write_text("B")
            with pytest.raises(RuntimeError, match="git commit failed"):
                git.commit_paths(["b.md"], "msg", author="not a valid author")
            assert not git.has_staged_changes(["b.md"])

    def test_restore_paths_checks_out_tracked_and_deletes_untracked(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "README.md").write_text("corrupted")
            (path / "brand-new.md").write_text("temp")
            (path / "other.md").write_text("keep me")   # not part of the restore
            restored, deleted = git.restore_paths(["README.md", "brand-new.md"])
            assert restored == ["README.md"] and deleted == ["brand-new.md"]
            assert (path / "README.md").read_text() == "# Test\n"
            assert not (path / "brand-new.md").exists()
            assert (path / "other.md").read_text() == "keep me"

    def test_restore_paths_recreates_deleted_tracked_file(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "README.md").unlink()
            restored, deleted = git.restore_paths(["README.md"])
            assert restored == ["README.md"] and deleted == []
            assert (path / "README.md").exists()

    def test_restore_paths_raises_on_checkout_failure(self):
        """Pre-creating an empty ``.git/index.lock`` makes the underlying
        ``git checkout`` fail deterministically (exit 128, "Unable to create
        ... index.lock: File exists") with no second process and no repo
        corruption — the same class of clean trigger as a malformed
        --author. The lock is removed in ``finally`` so a failing assertion
        can't leave it behind and break the temp repo's teardown."""
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "README.md").write_text("corrupted")
            lock = path / ".git" / "index.lock"
            lock.write_text("")
            try:
                with pytest.raises(RuntimeError, match="git checkout failed"):
                    git.restore_paths(["README.md"])
            finally:
                lock.unlink(missing_ok=True)
            # Repo is fully functional again once the lock is gone — no
            # corruption from the failed attempt.
            restored, deleted = git.restore_paths(["README.md"])
            assert restored == ["README.md"] and deleted == []
            assert (path / "README.md").read_text() == "# Test\n"

    def test_unstage_and_has_staged_changes(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "s.md").write_text("s")
            subprocess.run(["git", "-C", tmpdir, "add", "s.md"], check=True, capture_output=True)
            assert git.has_staged_changes()
            git.unstage()
            assert not git.has_staged_changes()
            assert (path / "s.md").exists()      # worktree untouched

    def test_degenerate_paths_are_a_noop_not_whole_index(self):
        """A non-empty ``paths`` that normalizes to nothing (e.g. ``["/"]``)
        must be a no-op for has_staged_changes/unstage — it must never
        silently widen to the whole index, which is reserved for
        paths=None."""
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            git = GitBackend(path)
            (path / "x.md").write_text("x")
            subprocess.run(["git", "-C", tmpdir, "add", "x.md"], check=True, capture_output=True)

            assert not git.has_staged_changes(["/"])
            assert not git.has_staged_changes([""])
            git.unstage(["/"])
            assert git.has_staged_changes()           # x.md still staged: no-op confirmed
            assert git.has_staged_changes(["x.md"])

            # paths=None keeps its documented whole-index meaning.
            git.unstage()
            assert not git.has_staged_changes()

    def test_ahead_count_without_remote_is_zero(self):
        from stash_mcp.git_backend import GitBackend

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _init_repo(path)
            assert GitBackend(path).ahead_count("origin", "main") == 0


# ---------------------------------------------------------------------------
# TransactionManager — write gating
# ---------------------------------------------------------------------------


class TestTransactionManagerDelegationCoverage:
    """TransactionManager wraps FileSystem with an explicit allowlist.

    It has no ``__getattr__`` fallback, so any FileSystem method the MCP
    server calls but the wrapper forgets to delegate raises
    ``AttributeError`` at runtime — and only in the configuration that
    actually installs the wrapper (writes enabled + git backend), which
    tests using a bare FileSystem never exercise.
    """

    def test_delegates_every_public_filesystem_method(self):
        public = {
            name
            for name in vars(FileSystem)
            if not name.startswith("_") and callable(getattr(FileSystem, name))
        }
        missing = {name for name in public if not hasattr(TransactionManager, name)}
        assert not missing, f"TransactionManager is missing delegations for: {missing}"

    def test_delegates_every_filesystem_attribute_the_server_uses(self):
        import ast

        import stash_mcp.mcp_server as server_mod

        # Parse rather than regex-scan so only real attribute accesses on the
        # injected `filesystem` count — not mentions in comments or docstrings.
        tree = ast.parse(Path(server_mod.__file__).read_text(encoding="utf-8"))
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "filesystem"
        }
        assert used, "found no filesystem attribute accesses — is the scan still valid?"
        missing = {name for name in used if not hasattr(TransactionManager, name)}
        assert not missing, (
            f"mcp_server calls filesystem.{{{', '.join(sorted(missing))}}} but "
            "TransactionManager does not delegate it"
        )

    def test_try_read_text_passes_through(self):
        with TemporaryDirectory() as tmpdir:
            tm, _fs = _make_tm(Path(tmpdir))
            assert tm.try_read_text("README.md") == "# Test\n"
            assert tm.try_read_text("does-not-exist.md") is None

    def test_resolve_path_passes_through(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            assert tm._resolve_path("README.md") == fs._resolve_path("README.md")


class TestTransactionManagerWriteGating:
    @pytest.mark.asyncio
    async def test_write_blocked_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            with pytest.raises(TransactionError, match="No active transaction"):
                tm.write_file("test.txt", "content")

    @pytest.mark.asyncio
    async def test_delete_blocked_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            fs.write_file("README.md", "x")  # write directly to fs
            with pytest.raises(TransactionError, match="No active transaction"):
                tm.delete_file("README.md")

    @pytest.mark.asyncio
    async def test_move_blocked_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            fs.write_file("README.md", "x")
            with pytest.raises(TransactionError, match="No active transaction"):
                tm.move_file("README.md", "moved.md")

    @pytest.mark.asyncio
    async def test_read_passes_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            # README.md created by _init_repo
            content = tm.read_file("README.md")
            assert "Test" in content

    @pytest.mark.asyncio
    async def test_list_passes_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            files = tm.list_all_files()
            assert "README.md" in files

    @pytest.mark.asyncio
    async def test_file_exists_passes_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            assert tm.file_exists("README.md")


# ---------------------------------------------------------------------------
# TransactionManager — lifecycle
# ---------------------------------------------------------------------------


class TestTransactionManagerLifecycle:
    @pytest.mark.asyncio
    async def test_start_returns_uuid(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            txn_id = await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            assert isinstance(txn_id, str)
            import uuid as _uuid
            # Verify it's a valid UUID
            assert _uuid.UUID(txn_id)
            await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_write_allowed_during_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            # Should not raise
            tm.write_file("new.txt", "hello")
            assert fs.file_exists("new.txt")
            await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_end_transaction_commits(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            tm.write_file("committed.txt", "content")
            await tm.end_transaction("session-1", "Add committed.txt")
            result = subprocess.run(
                ["git", "-C", tmpdir, "log", "--oneline"],
                capture_output=True,
                text=True,
            )
            assert "Add committed.txt" in result.stdout

    @pytest.mark.asyncio
    async def test_end_transaction_commits_with_author(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            tm.write_file("authored.txt", "content")
            await tm.end_transaction(
                "session-1",
                "Add authored file",
                author="Agent Smith <agent@example.com>",
            )
            result = subprocess.run(
                ["git", "-C", tmpdir, "log", "--format=%an <%ae>", "-1"],
                capture_output=True,
                text=True,
            )
            assert "Agent Smith" in result.stdout
            assert "agent@example.com" in result.stdout

    @pytest.mark.asyncio
    async def test_abort_transaction_resets(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            original = (path / "README.md").read_text()
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            tm.write_file("README.md", "corrupted")
            await tm.abort_transaction("session-1")
            assert (path / "README.md").read_text() == original

    @pytest.mark.asyncio
    async def test_lock_released_after_end(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            await tm.abort_transaction("session-1")
            assert not tm._lock.locked()

    @pytest.mark.asyncio
    async def test_cannot_start_second_transaction_same_session(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            with pytest.raises(TransactionError, match="already active"):
                await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_second_session_opens_concurrently(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            a = await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            b = await tm.start_transaction("session-2", timeout=30, lock_wait=0.1)
            assert a != b
            assert tm.open_transaction_count == 2
            await tm.abort_transaction("session-1")
            await tm.abort_transaction("session-2")
            assert tm.open_transaction_count == 0

    @pytest.mark.asyncio
    async def test_end_by_wrong_session_raises(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            with pytest.raises(TransactionError, match="No active transaction for this session"):
                await tm.end_transaction("session-2", "should fail")
            await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_abort_by_wrong_session_raises(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            with pytest.raises(TransactionError, match="No active transaction for this session"):
                await tm.abort_transaction("session-2")
            await tm.abort_transaction("session-1")


# ---------------------------------------------------------------------------
# TransactionManager — timeout
# ---------------------------------------------------------------------------


class TestTransactionManagerTimeout:
    @pytest.mark.asyncio
    async def test_transaction_auto_aborts_on_timeout(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            original = (path / "README.md").read_text()
            await tm.start_transaction("session-1", timeout=0.1, lock_wait=5)
            tm.write_file("README.md", "should be reverted")
            # Wait for auto-abort
            await asyncio.sleep(0.5)
            assert (path / "README.md").read_text() == original
            assert not tm._lock.locked()
            assert tm.open_transaction_count == 0

    @pytest.mark.asyncio
    async def test_second_session_acquires_lock_after_timeout(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=0.1, lock_wait=5)
            # Wait for auto-abort
            await asyncio.sleep(0.5)
            # Session 2 should now be able to acquire
            txn_id = await tm.start_transaction("session-2", timeout=30, lock_wait=5)
            assert txn_id is not None
            await tm.abort_transaction("session-2")


# ---------------------------------------------------------------------------
# TransactionManager — sync callbacks
# ---------------------------------------------------------------------------


class TestTransactionManagerSyncCallbacks:
    @pytest.mark.asyncio
    async def test_pause_called_on_start(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            pause = MagicMock()
            resume = MagicMock()
            tm.set_sync_callbacks(pause, resume)
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            pause.assert_called_once()
            resume.assert_not_called()
            await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_resume_called_on_end(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            pause = MagicMock()
            resume = MagicMock()
            tm.set_sync_callbacks(pause, resume)
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            await tm.abort_transaction("session-1")
            resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_called_on_abort(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            pause = MagicMock()
            resume = MagicMock()
            tm.set_sync_callbacks(pause, resume)
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            await tm.abort_transaction("session-1")
            resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_called_on_timeout(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            pause = MagicMock()
            resume = MagicMock()
            tm.set_sync_callbacks(pause, resume)
            await tm.start_transaction("session-1", timeout=0.1, lock_wait=5)
            await asyncio.sleep(0.5)
            resume.assert_called_once()


# ---------------------------------------------------------------------------
# MCP transaction tools
# ---------------------------------------------------------------------------


class TestMCPTransactionTools:
    """Tests for MCP start/end/abort_content_transaction tools."""

    def _make_mcp(self, tmpdir: Path):
        from stash_mcp.git_backend import GitBackend
        from stash_mcp.mcp_server import create_mcp_server

        _init_repo(tmpdir)
        fs = FileSystem(tmpdir)
        git = GitBackend(tmpdir)
        tm = TransactionManager(fs, git)
        with (
            patch("stash_mcp.mcp_server.Config.READ_ONLY", False),
            patch("stash_mcp.mcp_server.Config.GIT_SYNC_ENABLED", False),
        ):
            mcp = create_mcp_server(tm, git_backend=git)
        return mcp, tm, fs

    def _mock_context(self, session_obj=None):
        from fastmcp.server.context import Context, _current_context

        ctx = MagicMock(spec=Context)
        ctx.session = session_obj or MagicMock()
        ctx.session.send_resource_updated = AsyncMock()
        ctx.send_resource_list_changed = AsyncMock()
        token = _current_context.set(ctx)
        return ctx, token

    @pytest.mark.asyncio
    async def test_find_content_works_through_transaction_manager(self):
        """find_content reads via try_read_text, which the wrapper must delegate.

        Every other find_content test injects a bare FileSystem, so a missing
        delegation only surfaces in the wrapped (production) configuration.
        """
        import json

        with TemporaryDirectory() as tmpdir:
            mcp, tm, fs = self._make_mcp(Path(tmpdir))
            fs.write_file("notes.md", "alpha\nfindme here\nomega")
            tool = await mcp.get_tool("find_content")
            data = json.loads(str((await tool.run({"pattern": "findme"})).content[0].text))
            assert [(m["file_path"], m["line_number"]) for m in data["matches"]] == [
                ("notes.md", 2)
            ]

    @pytest.mark.asyncio
    async def test_move_content_batch_validation_works_through_transaction_manager(self):
        """move_content_batch resolves destinations via the wrapper's _resolve_path."""
        with TemporaryDirectory() as tmpdir:
            mcp, tm, fs = self._make_mcp(Path(tmpdir))
            fs.write_file("src.md", "body")
            fs.write_file("taken.md", "body")
            ctx, token = self._mock_context()
            try:
                await tm.start_transaction(str(id(ctx.session)), timeout=30, lock_wait=1)
                tool = await mcp.get_tool("move_content_batch")
                with pytest.raises(ValueError, match="Destination already exists"):
                    await tool.run(
                        {"moves": [{"source_path": "src.md", "dest_path": "taken.md"}]}
                    )
            finally:
                from fastmcp.server.context import _current_context

                _current_context.reset(token)

    @pytest.mark.asyncio
    async def test_transaction_tools_registered(self):
        with TemporaryDirectory() as tmpdir:
            mcp, tm, fs = self._make_mcp(Path(tmpdir))
            tool_names = set((await mcp.get_tools()).keys())
            assert "start_content_transaction" in tool_names
            assert "commit_content_transaction" in tool_names
            assert "abort_content_transaction" in tool_names

    @pytest.mark.asyncio
    async def test_start_returns_uuid(self):
        with TemporaryDirectory() as tmpdir:
            mcp, tm, fs = self._make_mcp(Path(tmpdir))
            ctx, token = self._mock_context()
            try:
                tool = await mcp.get_tool("start_content_transaction")
                result = await tool.run({})
                import uuid as _uuid

                # The result text should contain a valid UUID string
                text = str(result.content).strip()
                # UUID may be wrapped in quotes/brackets by the serializer; extract it
                import re

                uuid_match = re.search(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    text,
                )
                assert uuid_match is not None, f"No UUID found in result: {text}"
                _uuid.UUID(uuid_match.group())
            finally:
                from fastmcp.server.context import _current_context

                _current_context.reset(token)
                for sid in list(tm._txns):
                    await tm.abort_transaction(sid)

    @pytest.mark.asyncio
    async def test_write_blocked_without_transaction(self):
        with TemporaryDirectory() as tmpdir:
            mcp, tm, fs = self._make_mcp(Path(tmpdir))
            ctx, token = self._mock_context()
            try:
                tool = await mcp.get_tool("create_content")
                with pytest.raises((ValueError, TransactionError)):
                    await tool.run({"path": "new.md", "content": "hello"})
            finally:
                from fastmcp.server.context import _current_context

                _current_context.reset(token)

    @pytest.mark.asyncio
    async def test_write_allowed_after_start(self):
        with TemporaryDirectory() as tmpdir:
            mcp, tm, fs = self._make_mcp(Path(tmpdir))
            session_obj = MagicMock()
            ctx, token = self._mock_context(session_obj)
            try:
                start_tool = await mcp.get_tool("start_content_transaction")
                await start_tool.run({})

                create_tool = await mcp.get_tool("create_content")
                result = await create_tool.run({"path": "new.md", "content": "hello"})
                assert "Created" in str(result.content)
            finally:
                from fastmcp.server.context import _current_context

                _current_context.reset(token)
                for sid in list(tm._txns):
                    await tm.abort_transaction(sid)

    @pytest.mark.asyncio
    async def test_abort_resets_changes(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            mcp, tm, fs = self._make_mcp(path)
            original = (path / "README.md").read_text()
            session_obj = MagicMock()
            ctx, token = self._mock_context(session_obj)
            try:
                start_tool = await mcp.get_tool("start_content_transaction")
                await start_tool.run({})

                overwrite_tool = await mcp.get_tool("overwrite_content")
                import hashlib

                sha = hashlib.sha256(original.encode()).hexdigest()
                await overwrite_tool.run(
                    {"path": "README.md", "content": "corrupted", "sha": sha}
                )

                abort_tool = await mcp.get_tool("abort_content_transaction")
                result = await abort_tool.run({})
                assert "aborted" in str(result.content).lower()
            finally:
                from fastmcp.server.context import _current_context

                _current_context.reset(token)

            assert (path / "README.md").read_text() == original


# ---------------------------------------------------------------------------
# Mode matrix
# ---------------------------------------------------------------------------


class TestModeMatrix:
    """Verify the mode matrix: transaction tools only appear in the right mode."""

    @pytest.mark.asyncio
    async def test_no_transaction_tools_when_read_only(self):
        with TemporaryDirectory() as tmpdir:
            from stash_mcp.git_backend import GitBackend
            from stash_mcp.mcp_server import create_mcp_server

            _init_repo(Path(tmpdir))
            fs = FileSystem(Path(tmpdir))
            git = GitBackend(Path(tmpdir))
            with patch("stash_mcp.mcp_server.Config.READ_ONLY", True):
                mcp = create_mcp_server(fs, git_backend=git)
            tool_names = set((await mcp.get_tools()).keys())
            assert "start_content_transaction" not in tool_names
            assert "commit_content_transaction" not in tool_names
            assert "abort_content_transaction" not in tool_names

    @pytest.mark.asyncio
    async def test_no_transaction_tools_without_git_tracking(self):
        with TemporaryDirectory() as tmpdir:
            from stash_mcp.mcp_server import create_mcp_server

            fs = FileSystem(Path(tmpdir))
            with patch("stash_mcp.mcp_server.Config.READ_ONLY", False):
                # No git_backend passed → no transaction tools
                mcp = create_mcp_server(fs, git_backend=None)
            tool_names = set((await mcp.get_tools()).keys())
            assert "start_content_transaction" not in tool_names

    @pytest.mark.asyncio
    async def test_no_transaction_tools_when_plain_filesystem(self):
        """When git_backend is passed but filesystem is plain FileSystem, no txn tools."""
        with TemporaryDirectory() as tmpdir:
            from stash_mcp.git_backend import GitBackend
            from stash_mcp.mcp_server import create_mcp_server

            _init_repo(Path(tmpdir))
            fs = FileSystem(Path(tmpdir))
            git = GitBackend(Path(tmpdir))
            with patch("stash_mcp.mcp_server.Config.READ_ONLY", False):
                # filesystem is a plain FileSystem (not TransactionManager)
                mcp = create_mcp_server(fs, git_backend=git)
            tool_names = set((await mcp.get_tools()).keys())
            assert "start_content_transaction" not in tool_names


# ---------------------------------------------------------------------------
# get_transaction_status / list_content_transactions
# ---------------------------------------------------------------------------


class TestGetTransactionStatus:
    """Unit tests for TransactionManager.get_transaction_status()."""

    @pytest.mark.asyncio
    async def test_no_active_transaction(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            status = tm.get_transaction_status()
            assert status == {"has_active_transaction": False}

    @pytest.mark.asyncio
    async def test_no_active_transaction_with_session_id(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            status = tm.get_transaction_status("some-session")
            assert status == {"has_active_transaction": False}

    @pytest.mark.asyncio
    async def test_active_transaction_owned_by_caller(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            txn_id = await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            try:
                status = tm.get_transaction_status("session-1")
                assert status["has_active_transaction"] is True
                assert status["transaction_id"] == txn_id
                assert status["session_id"] == "session-1"
                assert status["owned_by_current_session"] is True
            finally:
                await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_active_transaction_owned_by_other_session(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            txn_id = await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            try:
                status = tm.get_transaction_status("session-2")
                assert status["has_active_transaction"] is True
                assert status["transaction_id"] == txn_id
                assert status["session_id"] == "session-1"
                assert status["owned_by_current_session"] is False
            finally:
                await tm.abort_transaction("session-1")

    @pytest.mark.asyncio
    async def test_no_owned_by_field_without_session_id(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("session-1", timeout=30, lock_wait=5)
            try:
                status = tm.get_transaction_status()
                assert "owned_by_current_session" not in status
            finally:
                await tm.abort_transaction("session-1")


class TestListContentTransactionsTool:
    """Integration tests for the list_content_transactions MCP tool."""

    def _make_mcp(self, tmpdir: Path):
        from stash_mcp.git_backend import GitBackend
        from stash_mcp.mcp_server import create_mcp_server

        _init_repo(tmpdir)
        fs = FileSystem(tmpdir)
        git = GitBackend(tmpdir)
        tm = TransactionManager(fs, git)
        with (
            patch("stash_mcp.mcp_server.Config.READ_ONLY", False),
            patch("stash_mcp.mcp_server.Config.GIT_SYNC_ENABLED", False),
        ):
            mcp = create_mcp_server(tm, git_backend=git)
        return mcp, tm

    def _mock_context(self, session_obj=None):
        from fastmcp.server.context import Context, _current_context

        ctx = MagicMock(spec=Context)
        ctx.session = session_obj or MagicMock()
        ctx.session.send_resource_updated = AsyncMock()
        ctx.send_resource_list_changed = AsyncMock()
        token = _current_context.set(ctx)
        return ctx, token

    @pytest.mark.asyncio
    async def test_tool_registered(self):
        with TemporaryDirectory() as tmpdir:
            mcp, _ = self._make_mcp(Path(tmpdir))
            tool_names = set((await mcp.get_tools()).keys())
            assert "list_content_transactions" in tool_names

    @pytest.mark.asyncio
    async def test_returns_no_active_transaction(self):
        with TemporaryDirectory() as tmpdir:
            mcp, _ = self._make_mcp(Path(tmpdir))
            ctx, token = self._mock_context()
            try:
                tool = await mcp.get_tool("list_content_transactions")
                result = await tool.run({})
                text = str(result.content)
                assert "has_active_transaction" in text
                assert "false" in text.lower()
            finally:
                from fastmcp.server.context import _current_context
                _current_context.reset(token)

    @pytest.mark.asyncio
    async def test_returns_active_transaction_owned_by_caller(self):
        with TemporaryDirectory() as tmpdir:
            mcp, tm = self._make_mcp(Path(tmpdir))
            session_obj = MagicMock()
            ctx, token = self._mock_context(session_obj)
            try:
                start_tool = await mcp.get_tool("start_content_transaction")
                await start_tool.run({})

                list_tool = await mcp.get_tool("list_content_transactions")
                result = await list_tool.run({})
                text = str(result.content)
                assert "has_active_transaction" in text
                assert "true" in text.lower()
                assert "owned_by_current_session" in text
            finally:
                from fastmcp.server.context import _current_context
                _current_context.reset(token)
                for sid in list(tm._txns):
                    await tm.abort_transaction(sid)

    @pytest.mark.asyncio
    async def test_not_registered_when_read_only(self):
        with TemporaryDirectory() as tmpdir:
            from stash_mcp.git_backend import GitBackend
            from stash_mcp.mcp_server import create_mcp_server

            _init_repo(Path(tmpdir))
            fs = FileSystem(Path(tmpdir))
            git = GitBackend(Path(tmpdir))
            with patch("stash_mcp.mcp_server.Config.READ_ONLY", True):
                mcp = create_mcp_server(fs, git_backend=git)
            tool_names = set((await mcp.get_tools()).keys())
            assert "list_content_transactions" not in tool_names


# ---------------------------------------------------------------------------
# TransactionManager — write lock, path-scoped commit/abort, autocommit
# ---------------------------------------------------------------------------


def _git_log(path: Path, *fmt: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "log", *fmt], capture_output=True, text=True,
    ).stdout


def _show_names(path: Path, ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(path), "show", "--name-only", "--format=%s%n%an", ref],
        capture_output=True, text=True,
    ).stdout


class TestPathScopedTransactions:
    @pytest.mark.asyncio
    async def test_concurrent_transactions_commit_only_their_own_files(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            await tm.start_transaction("s1", timeout=30)
            await tm.start_transaction("s2", timeout=30)
            async with tm.guard("s1"):
                fs.write_file("one.md", "1")
                await tm.after_write(["one.md"], session_id="s1", message="wrote one")
            async with tm.guard("s2"):
                fs.write_file("two.md", "2")
                await tm.after_write(["two.md"], session_id="s2")
            h1 = await tm.end_transaction("s1", "Txn one")
            assert h1
            shown = _show_names(path)
            assert "one.md" in shown and "two.md" not in shown
            assert "- wrote one" in _git_log(path, "-1", "--format=%b")
            assert (path / "two.md").exists()           # untouched, still uncommitted
            h2 = await tm.end_transaction("s2", "Txn two")
            assert h2 and "two.md" in _show_names(path)
            assert tm.open_transaction_count == 0

    @pytest.mark.asyncio
    async def test_abort_restores_only_own_paths(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            await tm.start_transaction("s1", timeout=30)
            await tm.start_transaction("s2", timeout=30)
            async with tm.guard("s1"):
                fs.write_file("README.md", "corrupted")
                fs.write_file("created.md", "new")
                await tm.after_write(["README.md", "created.md"], session_id="s1")
            async with tm.guard("s2"):
                fs.write_file("other.md", "keep")
                await tm.after_write(["other.md"], session_id="s2")
            await tm.abort_transaction("s1")
            assert (path / "README.md").read_text() == "# Test\n"
            assert not (path / "created.md").exists()
            assert (path / "other.md").read_text() == "keep"
            await tm.abort_transaction("s2")

    @pytest.mark.asyncio
    async def test_abort_leaves_nothing_staged(self):
        """Abort must clear the index entries of the paths it reverts.

        ``restore_paths`` deletes a file that is absent from HEAD but leaves
        its staged addition behind (``AD`` in ``git status``), so aborting a
        transaction that created a brand-new file would otherwise violate the
        "nothing stays staged between operations" invariant. The explicit
        ``git add`` reproduces exactly that half-staged state.
        """
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            await tm.start_transaction("s1", timeout=30)
            async with tm.guard("s1"):
                fs.write_file("created.md", "new")
                await tm.after_write(["created.md"], session_id="s1")
            subprocess.run(
                ["git", "-C", str(path), "add", "created.md"],
                check=True, capture_output=True,
            )
            assert tm.git.has_staged_changes(["created.md"])
            await tm.abort_transaction("s1")
            assert not (path / "created.md").exists()
            assert not tm.git.has_staged_changes()

    @pytest.mark.asyncio
    async def test_guard_rechecks_the_gate_after_waiting_for_the_lock(self):
        """A transaction that closes while its session is queued on the write
        lock must not let the queued write through ungated.

        ``_txns`` is cleared directly because ``abort_transaction`` needs the
        very lock this test is holding to create the queue.
        """
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("s1", timeout=0)      # no idle timer
            await tm.write_lock.acquire()                    # force s1 to queue

            async def queued_write():
                async with tm.guard("s1"):
                    pass

            task = asyncio.create_task(queued_write())
            await asyncio.sleep(0.05)                        # task is now waiting
            tm._txns.clear()                                 # its txn ends meanwhile
            tm.write_lock.release()
            with pytest.raises(TransactionError, match="No active transaction"):
                await task
            assert not tm.write_lock.locked()

    @pytest.mark.asyncio
    async def test_abort_is_a_noop_once_its_transaction_was_replaced(self):
        """A timeout timer that fires late must not revert paths belonging to
        whatever transaction replaced the one it was armed for."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            await tm.start_transaction("s1", timeout=0)
            stale = tm._txns["s1"]
            async with tm.guard("s1"):
                fs.write_file("keep.md", "v1")
                await tm.after_write(["keep.md"], session_id="s1")
            await tm.end_transaction("s1", "Committed v1")
            await tm.start_transaction("s1", timeout=0)      # a fresh transaction
            async with tm.guard("s1"):
                fs.write_file("keep.md", "v2")
                await tm.after_write(["keep.md"], session_id="s1")
            await tm._abort(stale)                           # the late timer fires
            assert (path / "keep.md").read_text() == "v2"    # v2 survives
            assert tm.open_transaction_count == 1
            await tm.abort_transaction("s1")

    @pytest.mark.asyncio
    async def test_abort_emits_content_events_for_reverted_paths(self):
        from stash_mcp import events as events_mod

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            seen: list[tuple[str, str]] = []
            listener = lambda event_type, p, **kw: seen.append((event_type, p))  # noqa: E731
            events_mod.add_listener(listener)
            try:
                await tm.start_transaction("s1", timeout=30)
                async with tm.guard("s1"):
                    fs.write_file("README.md", "x")
                    fs.write_file("tmp.md", "y")
                    await tm.after_write(["README.md", "tmp.md"], session_id="s1")
                await tm.abort_transaction("s1")
            finally:
                events_mod._listeners.remove(listener)   # events.py has no remove_listener
            assert ("content_updated", "README.md") in seen
            assert ("content_deleted", "tmp.md") in seen

    @pytest.mark.asyncio
    async def test_wrapper_writes_record_touched_paths(self):
        """Gated-mode writes through the FileSystem wrapper are commit-scoped too."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = _make_tm(path)
            fs.write_file("stray.md", "not mine")          # dirty, outside any txn
            await tm.start_transaction("s1", timeout=30)
            tm.write_file("mine.md", "mine")
            await tm.end_transaction("s1", "Only mine")
            shown = _show_names(path)
            assert "mine.md" in shown and "stray.md" not in shown

    @pytest.mark.asyncio
    async def test_end_transaction_with_no_changes_returns_none(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = _make_tm(Path(tmpdir))
            await tm.start_transaction("s1", timeout=30)
            assert await tm.end_transaction("s1", "nothing") is None
            assert tm.open_transaction_count == 0

    @pytest.mark.asyncio
    async def test_status_lists_all_open_transactions(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))
            a = await tm.start_transaction("s1", timeout=30)
            b = await tm.start_transaction("s2", timeout=30)
            async with tm.guard("s2"):
                fs.write_file("f.md", "f")
                await tm.after_write(["f.md"], session_id="s2")
            status = tm.get_transaction_status("s2")
            assert status["has_active_transaction"] is True
            assert status["transaction_id"] == b            # caller's own txn is primary
            assert status["owned_by_current_session"] is True
            assert status["count"] == 2
            by_id = {t["transaction_id"]: t for t in status["transactions"]}
            assert by_id[a]["session_id"] == "s1" and by_id[a]["owned_by_current_session"] is False
            assert by_id[b]["touched_paths"] == ["f.md"]
            await tm.abort_transaction("s1")
            await tm.abort_transaction("s2")


class TestAutocommit:
    def _make_autocommit_tm(self, tmpdir: Path):
        from stash_mcp.git_backend import GitBackend

        _init_repo(tmpdir)
        fs = FileSystem(tmpdir)
        git = GitBackend(tmpdir)
        return TransactionManager(fs, git, autocommit=True,
                                  author_default="Bot <bot@example.com>"), fs

    @pytest.mark.asyncio
    async def test_after_write_outside_transaction_commits_immediately(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = self._make_autocommit_tm(path)
            async with tm.guard("session-x"):
                fs.write_file("doc.md", "hello")
                short = await tm.after_write(
                    ["doc.md"], session_id="session-x",
                    message="doc-writer s1 describes=openpilot@v0.11.1",
                    author="Agent Smith <agent@example.com>",
                )
            assert short
            shown = _show_names(path)
            assert "doc.md" in shown and "doc-writer s1" in shown and "Agent Smith" in shown

    @pytest.mark.asyncio
    async def test_default_message_and_author(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = self._make_autocommit_tm(path)
            async with tm.guard(None):                          # REST/UI caller
                fs.write_file("doc.md", "hello")
                await tm.after_write(["doc.md"], session_id=None, default_message="Create doc.md")
            shown = _show_names(path)
            assert "Create doc.md" in shown and "Bot" in shown

    @pytest.mark.asyncio
    async def test_no_change_returns_none(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = self._make_autocommit_tm(Path(tmpdir))
            async with tm.guard(None):
                assert await tm.after_write(["README.md"], session_id=None) is None

    @pytest.mark.asyncio
    async def test_open_transaction_still_groups_in_autocommit_mode(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = self._make_autocommit_tm(path)
            await tm.start_transaction("s1", timeout=30)
            async with tm.guard("s1"):
                fs.write_file("a.md", "a")
                # recorded on the open transaction, not committed
                assert await tm.after_write(["a.md"], session_id="s1") is None
            assert "a.md" not in _show_names(path)
            await tm.end_transaction("s1", "Grouped")
            assert "a.md" in _show_names(path)

    @pytest.mark.asyncio
    async def test_gated_mode_guard_requires_transaction_for_sessions_only(self):
        with TemporaryDirectory() as tmpdir:
            tm, fs = _make_tm(Path(tmpdir))                     # autocommit=False
            with pytest.raises(TransactionError, match="No active transaction"):
                async with tm.guard("session-1"):
                    pass
            async with tm.guard(None):                          # REST/UI: lock only
                fs.write_file("rest.md", "r")
                assert await tm.after_write(["rest.md"], session_id=None) is not None

    @pytest.mark.asyncio
    async def test_write_lock_wait_timeout(self):
        with TemporaryDirectory() as tmpdir:
            tm, _ = self._make_autocommit_tm(Path(tmpdir))
            tm.lock_wait = 0.05
            await tm.write_lock.acquire()
            try:
                with pytest.raises(TransactionError, match="busy"):
                    async with tm.guard(None):
                        pass
            finally:
                tm.write_lock.release()

    @pytest.mark.asyncio
    async def test_concurrent_writers_each_get_their_own_commit(self):
        """Lock covers write+commit, so interleaved writers keep exact attribution."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            tm, fs = self._make_autocommit_tm(path)

            async def writer(name: str, n: int):
                for i in range(n):
                    async with tm.guard(name):
                        fs.write_file("shared.md", f"{name}-{i}")
                        await tm.after_write(
                            ["shared.md"], session_id=name,
                            message=f"{name} write {i}", author=f"{name} <{name}@x>",
                        )
                    await asyncio.sleep(0)

            await asyncio.gather(writer("alpha", 3), writer("beta", 3))
            log = _git_log(path, "--format=%an|%s", "-6")
            lines = [line for line in log.splitlines() if line]
            assert len(lines) == 6
            for line in lines:
                author, subject = line.split("|")
                assert subject.startswith(author)      # each commit attributed to its writer
