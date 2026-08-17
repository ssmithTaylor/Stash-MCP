"""Tests for the shared startup helpers and both entrypoints' use of them.

The bug these guard against: main.py (HTTP) and server.py (stdio) each kept a
private copy of the bootstrap sequence, and the copies drifted. stdio ended up
without auto-init, without the autocommit/sync config validation, without the
search scoping defaults, and without a TransactionManager at all — so
STASH_GIT_TRACKING=true registered the write tools but left every write
uncommitted and unlocked, silently, depending only on which transport you
started.
"""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

import pytest

from stash_mcp import main as main_mod
from stash_mcp import server as server_mod
from stash_mcp import startup as startup_mod
from stash_mcp.config import Config
from stash_mcp.filesystem import FileSystem


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e.com",
         "commit", "--allow-empty", "-m", "init"],
        cwd=path, capture_output=True, check=True,
    )


class TestNoDuplicatedBootstrap:
    """Both entrypoints must share one implementation, not two copies."""

    @pytest.mark.parametrize(
        "name",
        ["maybe_clone_repo", "maybe_init_repo", "create_search_engine",
         "create_git_backend"],
    )
    def test_both_entrypoints_use_the_same_function_object(self, name):
        shared = getattr(startup_mod, name)
        assert getattr(main_mod, f"_{name}") is shared
        assert getattr(server_mod, f"_{name}") is shared

    def test_neither_entrypoint_defines_its_own_copy(self):
        """A local `def` would shadow the import and let drift start again."""
        for module in (main_mod, server_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for name in ("maybe_clone_repo", "maybe_init_repo",
                         "create_search_engine", "create_git_backend"):
                assert f"def _{name}" not in source, (
                    f"{module.__name__} redefines _{name} instead of importing it"
                )


class TestBuildWriteStack:
    def test_no_git_backend_means_no_transaction_manager(self):
        with TemporaryDirectory() as tmp:
            fs = FileSystem(Path(tmp))
            tm, fs_for_mcp = startup_mod.build_write_stack(fs, None)
            assert tm is None
            assert fs_for_mcp is fs

    def test_read_only_mode_means_no_transaction_manager(self, monkeypatch):
        with TemporaryDirectory() as tmp:
            _init_repo(Path(tmp))
            from stash_mcp.git_backend import GitBackend

            monkeypatch.setattr(Config, "READ_ONLY", True)
            fs = FileSystem(Path(tmp))
            tm, fs_for_mcp = startup_mod.build_write_stack(
                fs, GitBackend(Path(tmp), author_default="T <t@e.com>")
            )
            assert tm is None
            assert fs_for_mcp is fs

    def test_gated_mode_wraps_the_filesystem(self, monkeypatch):
        with TemporaryDirectory() as tmp:
            _init_repo(Path(tmp))
            from stash_mcp.git_backend import GitBackend

            monkeypatch.setattr(Config, "READ_ONLY", False)
            monkeypatch.setattr(Config, "GIT_AUTOCOMMIT", False)
            fs = FileSystem(Path(tmp))
            tm, fs_for_mcp = startup_mod.build_write_stack(
                fs, GitBackend(Path(tmp), author_default="T <t@e.com>")
            )
            assert tm is not None
            # Gated mode routes writes through the manager so writes outside a
            # transaction are rejected.
            assert fs_for_mcp is tm

    def test_autocommit_mode_uses_the_raw_filesystem(self, monkeypatch):
        """Load-bearing: _resolve_txn_for_wrapper_write demands an open
        transaction even under autocommit, so handing the wrapper to the MCP
        server would make every autocommit write fail."""
        with TemporaryDirectory() as tmp:
            _init_repo(Path(tmp))
            from stash_mcp.git_backend import GitBackend

            monkeypatch.setattr(Config, "READ_ONLY", False)
            monkeypatch.setattr(Config, "GIT_AUTOCOMMIT", True)
            fs = FileSystem(Path(tmp))
            tm, fs_for_mcp = startup_mod.build_write_stack(
                fs, GitBackend(Path(tmp), author_default="T <t@e.com>")
            )
            assert tm is not None
            assert fs_for_mcp is fs


class TestStdioEntrypointWiring:
    """The regression itself: stdio must build the same write stack as HTTP."""

    @pytest.mark.asyncio
    async def test_stdio_passes_a_transaction_manager_to_the_mcp_server(
        self, monkeypatch
    ):
        with TemporaryDirectory() as tmp:
            content = Path(tmp) / "content"
            content.mkdir()
            _init_repo(content)

            monkeypatch.setattr(Config, "CONTENT_DIR", content)
            monkeypatch.setattr(Config, "CONTENT_PATHS", None)
            monkeypatch.setattr(Config, "READ_ONLY", False)
            monkeypatch.setattr(Config, "GIT_TRACKING", True)
            monkeypatch.setattr(Config, "GIT_AUTOCOMMIT", False)
            monkeypatch.setattr(Config, "GIT_CLONE_URL", None)
            monkeypatch.setattr(Config, "GIT_SYNC_URL", None)
            monkeypatch.setattr(Config, "GIT_SYNC_ENABLED", False)
            monkeypatch.setattr(Config, "SEARCH_ENABLED", False)
            monkeypatch.setattr(Config, "METRICS_ENABLED", False)
            monkeypatch.setattr(Config, "GIT_AUTHOR_DEFAULT", "T <t@e.com>")

            captured = {}
            real_create = server_mod.create_mcp_server

            def _capture(fs_arg, **kwargs):
                captured["fs"] = fs_arg
                captured["kwargs"] = kwargs
                mcp = real_create(fs_arg, **kwargs)
                mcp.run_stdio_async = AsyncMock()
                captured["mcp"] = mcp
                return mcp

            monkeypatch.setattr(server_mod, "create_mcp_server", _capture)
            await server_mod.main()

            from stash_mcp.transactions import TransactionManager

            tm = captured["kwargs"].get("transaction_manager")
            assert isinstance(tm, TransactionManager), (
                "stdio must wire a TransactionManager when git tracking is on"
            )
            # Gated mode: the MCP server writes through the manager.
            assert captured["fs"] is tm

            # And the transaction tools are actually reachable over stdio.
            tools = await captured["mcp"].get_tools()
            for name in ("start_content_transaction", "commit_content_transaction",
                         "abort_content_transaction", "list_content_transactions"):
                assert name in tools, f"{name} missing from the stdio tool surface"

    @pytest.mark.asyncio
    async def test_stdio_auto_inits_a_repo_on_a_fresh_content_dir(self, monkeypatch):
        """Auto-init was one of the drifted-away behaviours: stdio used to hard
        fail on a fresh volume with tracking on."""
        with TemporaryDirectory() as tmp:
            content = Path(tmp) / "not-created-yet"

            monkeypatch.setattr(Config, "CONTENT_DIR", content)
            monkeypatch.setattr(Config, "CONTENT_PATHS", None)
            monkeypatch.setattr(Config, "READ_ONLY", False)
            monkeypatch.setattr(Config, "GIT_TRACKING", True)
            monkeypatch.setattr(Config, "GIT_AUTOCOMMIT", False)
            monkeypatch.setattr(Config, "GIT_CLONE_URL", None)
            monkeypatch.setattr(Config, "GIT_SYNC_URL", None)
            monkeypatch.setattr(Config, "GIT_SYNC_ENABLED", False)
            monkeypatch.setattr(Config, "SEARCH_ENABLED", False)
            monkeypatch.setattr(Config, "METRICS_ENABLED", False)
            monkeypatch.setattr(Config, "GIT_AUTHOR_DEFAULT", "T <t@e.com>")

            real_create = server_mod.create_mcp_server

            def _capture(fs_arg, **kwargs):
                mcp = real_create(fs_arg, **kwargs)
                mcp.run_stdio_async = AsyncMock()
                return mcp

            monkeypatch.setattr(server_mod, "create_mcp_server", _capture)
            await server_mod.main()

            assert (content / ".git").is_dir(), "stdio did not auto-init the repo"
            log = subprocess.run(
                ["git", "log", "--oneline"], cwd=content,
                capture_output=True, text=True,
            )
            assert log.returncode == 0
            assert log.stdout.strip(), "auto-init left an unborn HEAD"


# The search settings both entrypoints must forward (including the scoping
# defaults the stdio copy used to drop) are covered by
# tests/test_search.py::TestSearchConfig::test_entrypoints_pass_search_settings_from_config,
# which parametrizes over both modules.


class TestGitBackendConfigValidation:
    """The stdio copy of _create_git_backend lacked these guards entirely."""

    def test_sync_without_tracking_fails_fast(self, monkeypatch):
        monkeypatch.setattr(Config, "GIT_SYNC_ENABLED", True)
        monkeypatch.setattr(Config, "GIT_TRACKING", False)
        with pytest.raises(SystemExit):
            startup_mod.create_git_backend()

    def test_tracking_off_returns_no_backend(self, monkeypatch):
        monkeypatch.setattr(Config, "GIT_SYNC_ENABLED", False)
        monkeypatch.setattr(Config, "GIT_TRACKING", False)
        assert startup_mod.create_git_backend() is None
