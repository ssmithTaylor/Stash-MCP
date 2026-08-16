"""Transaction management for Stash-MCP.

Provides :class:`TransactionManager`, which wraps a :class:`~.filesystem.FileSystem`
instance, owns the single short *write lock* every mutation runs under, and
tracks per-session transactions.

Transactions are **non-exclusive**: any number of MCP sessions may hold one
at the same time. A transaction is a list of touched paths; committing it
commits exactly those paths and aborting it restores exactly those paths.
Writes made outside a transaction are either rejected (gated mode,
``autocommit=False``) or committed immediately (``autocommit=True``).
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .events import CONTENT_DELETED, CONTENT_UPDATED, emit

logger = logging.getLogger(__name__)

_NO_ACTIVE_TRANSACTION = "No active transaction. Call start_content_transaction first."


class TransactionError(Exception):
    """Raised when a transaction operation is invalid."""


def _get_current_session_id() -> str | None:
    """Return the calling MCP session's identity string, or *None*.

    Reads the current FastMCP :class:`Context` from its context-variable so
    the caller does not need to pass a session reference explicitly.
    """
    try:
        from fastmcp.server.context import _current_context

        ctx = _current_context.get()
        return str(id(ctx.session))
    except Exception:
        return None


def _clean_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        q = (p or "").replace("\\", "/").strip("/")
        if q and q not in out:
            out.append(q)
    return out


def default_commit_message(paths: list[str], verb: str = "Update") -> str:
    """``"Update <path>"`` for one path, ``"Update N files"`` otherwise."""
    if len(paths) == 1:
        return f"{verb} {paths[0]}"
    return f"{verb} {len(paths)} files"


@dataclass
class _Txn:
    txn_id: str
    session_id: str
    started_at: str
    timeout_seconds: float
    touched: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    timeout_task: asyncio.Task | None = None

    def record(self, paths: list[str], message: str | None) -> None:
        for p in paths:
            if p not in self.touched:
                self.touched.append(p)
        if message:
            self.messages.append(message)


class TransactionManager:
    """Wraps a :class:`~.filesystem.FileSystem` with a write lock and transactions.

    Read methods delegate directly to the inner filesystem. In gated mode
    (``autocommit=False``) the delegated write methods require an open
    transaction (some transaction is open and, when an MCP session context
    is available, that session owns one). In autocommit mode the wrapper is
    normally not installed at all — tools use the raw filesystem plus
    :meth:`guard` / :meth:`after_write`.

    Every mutation must run inside :meth:`guard` (which takes the write
    lock) and finish with :meth:`after_write`, which records the paths in the
    caller's open transaction or commits them immediately.
    """

    def __init__(
        self,
        fs,
        git,
        *,
        autocommit: bool = False,
        author_default: str | None = None,
        lock_wait: float = 120.0,
    ) -> None:
        self.fs = fs
        self.git = git
        self.autocommit = autocommit
        self.author_default = author_default
        self.lock_wait = lock_wait
        self._lock: asyncio.Lock = asyncio.Lock()
        self._txns: dict[str, _Txn] = {}
        self._pause_sync: Callable[[], None] | None = None
        self._resume_sync: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Sync-pause wiring (pause while any transaction is open)
    # ------------------------------------------------------------------

    def set_sync_callbacks(
        self,
        pause: Callable[[], None],
        resume: Callable[[], None],
    ) -> None:
        """Register callbacks that pause/resume the periodic git-sync loop."""
        self._pause_sync = pause
        self._resume_sync = resume

    # ------------------------------------------------------------------
    # Write lock
    # ------------------------------------------------------------------

    @property
    def write_lock(self) -> asyncio.Lock:
        """The single short lock every mutation, commit, abort and pull runs under."""
        return self._lock

    @property
    def open_transaction_count(self) -> int:
        return len(self._txns)

    async def _acquire(self) -> None:
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self.lock_wait)
        except TimeoutError:
            raise TransactionError(
                "Write lock unavailable (server busy); try again later."
            )

    def _check_gate(self, session_id: str | None) -> None:
        """Raise unless *session_id* may mutate content right now.

        REST/UI callers (``session_id=None``) and autocommit mode are always
        allowed; a gated MCP session must own an open transaction.
        """
        if session_id is not None and not self.autocommit and session_id not in self._txns:
            raise TransactionError(_NO_ACTIVE_TRANSACTION)

    @asynccontextmanager
    async def guard(self, session_id: str | None):
        """Take the write lock for one mutation.

        In gated mode an MCP session (``session_id`` not None) must own an
        open transaction; ``session_id=None`` (REST/UI callers) only takes
        the lock. Resets the owning transaction's idle timer.
        """
        self._check_gate(session_id)  # fail fast rather than queue on the lock
        await self._acquire()
        try:
            # The transaction may have been committed or timed out while this
            # caller sat in the lock queue, so the gate only becomes
            # authoritative once the lock is actually held.
            self._check_gate(session_id)
            txn = self._txns.get(session_id) if session_id is not None else None
            if txn is not None:
                self._reset_timeout(txn)
            yield
        finally:
            self._lock.release()

    async def after_write(
        self,
        paths: list[str],
        *,
        session_id: str | None,
        message: str | None = None,
        author: str | None = None,
        default_message: str | None = None,
    ) -> str | None:
        """Record or commit *paths* after a filesystem mutation (call inside guard()).

        Returns the short commit hash when a commit was made, otherwise None
        (recorded in an open transaction, or nothing changed).
        """
        clean = _clean_paths(paths)
        if not clean:
            return None
        txn = self._txns.get(session_id) if session_id is not None else None
        if txn is not None:
            txn.record(clean, message)
            return None
        commit_message = message or default_message or default_commit_message(clean)
        try:
            return await asyncio.to_thread(
                self.git.commit_paths, clean, commit_message, author or self.author_default
            )
        except RuntimeError as exc:
            raise TransactionError(
                f"File written but git commit failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Transaction lifecycle
    # ------------------------------------------------------------------

    async def start_transaction(
        self,
        session_id: str,
        timeout: float,
        lock_wait: float | None = None,
    ) -> str:
        """Open a transaction for *session_id* and return its UUID.

        Transactions are non-exclusive; ``lock_wait`` is accepted for
        backward compatibility and ignored.

        Raises:
            TransactionError: If this session already holds a transaction.
        """
        if session_id in self._txns:
            raise TransactionError(
                "A transaction is already active for this session."
            )
        txn = _Txn(
            txn_id=str(uuid.uuid4()),
            session_id=session_id,
            started_at=datetime.now(UTC).isoformat(),
            timeout_seconds=timeout,
        )
        first = not self._txns
        self._txns[session_id] = txn
        if first and self._pause_sync is not None:
            self._pause_sync()
        self._reset_timeout(txn)
        logger.info("Transaction started: %s (session=%s)", txn.txn_id, session_id)
        return txn.txn_id

    async def end_transaction(
        self,
        session_id: str,
        message: str,
        author: str | None = None,
        sync_remote: str | None = None,
        sync_branch: str | None = None,
    ) -> str | None:
        """Commit the transaction's touched paths and close it.

        Per-write messages recorded via :meth:`after_write` are appended to
        the commit body as bullets. Returns the short commit hash, or None
        when nothing changed.

        Raises:
            TransactionError: If *session_id* does not own an open transaction.
            RuntimeError: If the git commit or push fails.
        """
        txn = self._require_txn(session_id)
        await self._acquire()  # if this times out the txn stays open with its timer armed
        try:
            self._cancel_timeout(txn)
            body = message
            if txn.messages:
                body = message + "\n\n" + "\n".join(f"- {m}" for m in txn.messages)
            commit_hash = await asyncio.to_thread(
                self.git.commit_paths, list(txn.touched), body, author or self.author_default
            )
            if sync_remote and sync_branch and commit_hash:
                await asyncio.to_thread(self.git.push, sync_remote, sync_branch)
        finally:
            self._close(txn)
            self._lock.release()
        return commit_hash

    async def abort_transaction(self, session_id: str) -> None:
        """Restore the transaction's touched paths to HEAD and close it.

        Raises:
            TransactionError: If *session_id* does not own an open transaction.
        """
        txn = self._require_txn(session_id)
        await self._abort(txn)

    async def _abort(self, txn: _Txn) -> None:
        restored: list[str] = []
        deleted: list[str] = []
        await self._acquire()  # on timeout the txn simply stays open; nothing is torn down
        try:
            if self._txns.get(txn.session_id) is not txn:
                # Committed, aborted or replaced while we waited for the lock;
                # reverting now would clobber whatever took its place.
                return
            self._cancel_timeout(txn)  # no-op for the timeout task itself
            restored, deleted = await asyncio.to_thread(
                self._restore_and_unstage, list(txn.touched)
            )
        except Exception as exc:
            logger.error("Restore on abort failed for %s: %s", txn.txn_id, exc)
        finally:
            self._close(txn)
            self._lock.release()
        for p in restored:
            emit(CONTENT_UPDATED, p)
        for p in deleted:
            emit(CONTENT_DELETED, p)

    def _restore_and_unstage(self, paths: list[str]) -> tuple[list[str], list[str]]:
        """Revert *paths* to HEAD and clear whatever they left in the index.

        ``restore_paths`` deletes a path that HEAD does not know about but
        leaves its staged addition behind, so the mixed reset is what keeps
        the "nothing stays staged between operations" invariant. Runs on a
        worker thread; *paths* is never widened to the whole index because
        an empty list short-circuits before ``unstage`` is reached.
        """
        if not paths:
            return [], []
        try:
            return self.git.restore_paths(paths)
        finally:
            self.git.unstage(paths)

    async def _auto_abort(self, txn: _Txn, timeout: float) -> None:
        """Abort *txn* automatically after *timeout* idle seconds."""
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if self._txns.get(txn.session_id) is not txn:
            return  # Ended normally before we ran
        logger.warning("Transaction %s timed out; restoring its paths.", txn.txn_id)
        try:
            await self._abort(txn)
        except Exception as exc:  # never let the timer task die noisily
            logger.error("Auto-abort of %s failed: %s", txn.txn_id, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_txn(self, session_id: str) -> _Txn:
        txn = self._txns.get(session_id)
        if txn is None:
            raise TransactionError("No active transaction for this session.")
        return txn

    def _cancel_timeout(self, txn: _Txn) -> None:
        task = txn.timeout_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        txn.timeout_task = None

    def _reset_timeout(self, txn: _Txn) -> None:
        self._cancel_timeout(txn)
        if txn.timeout_seconds > 0:
            txn.timeout_task = asyncio.create_task(
                self._auto_abort(txn, txn.timeout_seconds),
                name=f"txn-timeout-{txn.txn_id[:8]}",
            )

    def _close(self, txn: _Txn) -> None:
        if self._txns.get(txn.session_id) is not txn:
            return  # Already closed by a concurrent commit/abort/timeout
        del self._txns[txn.session_id]
        if not self._txns and self._resume_sync is not None:
            self._resume_sync()

    def _resolve_txn_for_wrapper_write(self) -> _Txn:
        """Gated-mode check for the delegated write methods."""
        if not self._txns:
            raise TransactionError(_NO_ACTIVE_TRANSACTION)
        session_id = _get_current_session_id()
        if session_id is not None:
            txn = self._txns.get(session_id)
            if txn is None:
                raise TransactionError(_NO_ACTIVE_TRANSACTION)
        elif len(self._txns) == 1:
            txn = next(iter(self._txns.values()))
        else:
            raise TransactionError(
                "Ambiguous transaction: several are open and no session context is set."
            )
        self._reset_timeout(txn)
        return txn

    def get_transaction_status(self, session_id: str | None = None) -> dict:
        """Return the current transaction state.

        With no open transaction: ``{"has_active_transaction": False}``.
        Otherwise ``transaction_id``/``session_id`` describe the caller's own
        transaction when it has one (else the oldest open one), plus
        ``count`` and a ``transactions`` list. ``owned_by_current_session``
        fields are included only when *session_id* is given.
        """
        if not self._txns:
            return {"has_active_transaction": False}
        own = self._txns.get(session_id) if session_id is not None else None
        primary = own or next(iter(self._txns.values()))
        result: dict = {
            "has_active_transaction": True,
            "transaction_id": primary.txn_id,
            "session_id": primary.session_id,
            "count": len(self._txns),
            "transactions": [],
        }
        if session_id is not None:
            result["owned_by_current_session"] = own is not None
        for txn in self._txns.values():
            entry = {
                "transaction_id": txn.txn_id,
                "session_id": txn.session_id,
                "started_at": txn.started_at,
                "touched_paths": list(txn.touched),
            }
            if session_id is not None:
                entry["owned_by_current_session"] = txn.session_id == session_id
            result["transactions"].append(entry)
        return result

    # ------------------------------------------------------------------
    # FileSystem delegation — reads pass through, writes are gated
    # ------------------------------------------------------------------

    @property
    def content_dir(self):
        return self.fs.content_dir

    @property
    def include_patterns(self):
        return self.fs.include_patterns

    # --- Read methods (unconditional) ---

    def _resolve_path(self, relative_path: str):
        return self.fs._resolve_path(relative_path)

    def read_file(self, path: str) -> str:
        return self.fs.read_file(path)

    def try_read_text(self, path: str) -> str | None:
        return self.fs.try_read_text(path)

    def list_files(self, relative_path: str = "") -> list:
        return self.fs.list_files(relative_path)

    def list_all_files(self, relative_path: str = "") -> list:
        return self.fs.list_all_files(relative_path)

    def file_exists(self, path: str) -> bool:
        return self.fs.file_exists(path)

    # --- Write methods (require an open transaction; record touched paths) ---

    def write_file(self, path: str, content: str) -> None:
        txn = self._resolve_txn_for_wrapper_write()
        self.fs.write_file(path, content)
        txn.record(_clean_paths([path]), None)

    def delete_file(self, path: str) -> None:
        txn = self._resolve_txn_for_wrapper_write()
        self.fs.delete_file(path)
        txn.record(_clean_paths([path]), None)

    def move_file(self, source_path: str, dest_path: str) -> None:
        txn = self._resolve_txn_for_wrapper_write()
        self.fs.move_file(source_path, dest_path)
        txn.record(_clean_paths([source_path, dest_path]), None)

    def move_directory(self, source_path: str, dest_path: str) -> list[tuple[str, str]]:
        txn = self._resolve_txn_for_wrapper_write()
        moved = self.fs.move_directory(source_path, dest_path)
        txn.record(_clean_paths([p for pair in moved for p in pair]), None)
        return moved

    def create_directory(self, path: str) -> None:
        self._resolve_txn_for_wrapper_write()
        return self.fs.create_directory(path)
