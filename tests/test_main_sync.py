"""Tests for main.py git wiring: autocommit config and the sync loop."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from stash_mcp import main as main_mod
from stash_mcp.filesystem import FileSystem
from stash_mcp.git_backend import PullResult
from stash_mcp.transactions import TransactionManager


def test_git_autocommit_config_default_false():
    from stash_mcp.config import Config

    assert Config.GIT_AUTOCOMMIT is False


async def _run_one_iteration(monkeypatch, git, tm, sync_enabled=True):
    monkeypatch.setattr(main_mod.Config, "GIT_SYNC_ENABLED", sync_enabled)
    monkeypatch.setattr(main_mod.Config, "GIT_SYNC_REMOTE", "origin")
    monkeypatch.setattr(main_mod.Config, "GIT_SYNC_BRANCH", "main")
    monkeypatch.setattr(main_mod.Config, "GIT_SYNC_INTERVAL", 0)

    async def stop_after_first(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(main_mod.asyncio, "sleep", stop_after_first)
    event = asyncio.Event()
    event.set()
    with pytest.raises(asyncio.CancelledError):
        await main_mod._git_sync_loop(git, None, event, tm)


async def test_sync_loop_pulls_under_lock_and_pushes_when_ahead(monkeypatch):
    with TemporaryDirectory() as tmp:
        git = MagicMock()
        git.pull.return_value = PullResult(success=True)
        git.ahead_count.return_value = 2
        tm = TransactionManager(FileSystem(Path(tmp)), git, autocommit=True)
        held: list[bool] = []
        real_pull = git.pull

        def pull_and_check(*a, **k):
            held.append(tm.write_lock.locked())
            return real_pull.return_value

        git.pull.side_effect = pull_and_check
        await _run_one_iteration(monkeypatch, git, tm)
        assert held == [True]                     # pull ran while the write lock was held
        git.push.assert_called_once_with("origin", "main")


async def test_sync_loop_skips_push_when_not_ahead(monkeypatch):
    with TemporaryDirectory() as tmp:
        git = MagicMock()
        git.pull.return_value = PullResult(success=True)
        git.ahead_count.return_value = 0
        tm = TransactionManager(FileSystem(Path(tmp)), git, autocommit=True)
        await _run_one_iteration(monkeypatch, git, tm)
        git.push.assert_not_called()


async def test_sync_loop_skips_pull_while_transaction_open(monkeypatch):
    with TemporaryDirectory() as tmp:
        git = MagicMock()
        git.pull.return_value = PullResult(success=True)
        git.ahead_count.return_value = 0
        tm = TransactionManager(FileSystem(Path(tmp)), git, autocommit=True)
        event = asyncio.Event()
        event.set()
        tm.set_sync_callbacks(pause=event.clear, resume=event.set)
        await tm.start_transaction("s1", timeout=30)
        monkeypatch.setattr(main_mod.Config, "GIT_SYNC_ENABLED", True)
        monkeypatch.setattr(main_mod.Config, "GIT_SYNC_REMOTE", "origin")
        monkeypatch.setattr(main_mod.Config, "GIT_SYNC_BRANCH", "main")
        monkeypatch.setattr(main_mod.Config, "GIT_SYNC_INTERVAL", 0)

        async def stop_after_first(_seconds):
            raise asyncio.CancelledError

        monkeypatch.setattr(main_mod.asyncio, "sleep", stop_after_first)
        with pytest.raises(asyncio.CancelledError):
            await main_mod._git_sync_loop(git, None, event, tm)
        git.pull.assert_not_called()
        await tm.abort_transaction("s1")
