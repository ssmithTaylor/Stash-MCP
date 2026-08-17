"""FastAPI app entrypoint for Stash-MCP."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from .api import create_api
from .config import Config
from .events import CONTENT_CREATED, CONTENT_DELETED, CONTENT_UPDATED, add_listener, emit
from .filesystem import FileSystem
from .mcp_server import create_mcp_server
from .metrics import get_metrics, init_metrics

# The startup helpers live in startup.py so the stdio entrypoint (server.py)
# runs exactly the same bootstrap. They are imported under their historical
# private names because the tests reach for stash_mcp.main._maybe_* and
# monkeypatch stash_mcp.main._create_search_engine — always call them through
# this module's namespace (bare names) so those patches keep taking effect.
from .startup import build_write_stack
from .startup import create_git_backend as _create_git_backend
from .startup import create_search_engine as _create_search_engine
from .startup import maybe_clone_repo as _maybe_clone_repo
from .startup import maybe_init_repo as _maybe_init_repo
from .ui import create_ui_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mapping from event bus event types to metric event names
_CONTENT_EVENT_METRIC_MAP = {
    "content_created": "created",
    "content_updated": "updated",
    "content_deleted": "deleted",
    "content_moved": "moved",
}


def _task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from background tasks."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        logger.error(f"Background task {task.get_name()} failed: {exc}", exc_info=exc)


# Consecutive skipped pulls before the operator hears about it. Transactions
# are per-session and non-exclusive, so the pause window is the union of every
# open transaction's window and can be near-continuous on a busy server. A repo
# that keeps pushing but silently stops pulling is the failure mode this warns
# about; it repeats every N skips so a single line cannot be missed.
PULL_SKIP_WARN_AFTER = 5


async def _git_sync_loop(
    git_backend,
    search_engine,
    sync_event: asyncio.Event | None = None,
    transaction_manager=None,
) -> None:
    """Periodic git pull (+ push when ahead) task. Runs until cancelled.

    Pull and push run under the write lock so they never interleave with a
    write+commit; pull is skipped while any transaction is open (the event
    is cleared by the TransactionManager's sync callbacks). Push is not gated
    on that event — it never touches the working tree, so it stays safe
    mid-transaction and keeps local commits reaching the remote even while
    pulls are starved; the starvation itself is surfaced by a WARNING.
    """
    remote = Config.GIT_SYNC_REMOTE
    branch = Config.GIT_SYNC_BRANCH
    interval = Config.GIT_SYNC_INTERVAL
    recursive = Config.GIT_SYNC_RECURSIVE
    lock = transaction_manager.write_lock if transaction_manager is not None else None
    skipped_pulls = 0

    async def _locked(fn, *args):
        if lock is None:
            return await asyncio.to_thread(fn, *args)
        async with lock:
            return await asyncio.to_thread(fn, *args)

    def _push_if_ahead() -> int:
        """Count and push in one lock hold, on a worker thread."""
        ahead = git_backend.ahead_count(remote, branch)
        if ahead > 0:
            git_backend.push(remote, branch)
        return ahead

    while True:
        try:
            if sync_event is None or sync_event.is_set():
                skipped_pulls = 0
                result = await _locked(git_backend.pull, remote, branch, recursive)
                if result.success:
                    logger.info("Git sync: %s", result.message or "up to date")
                    for path in result.added_files:
                        emit(CONTENT_CREATED, path)
                    for path in result.modified_files:
                        emit(CONTENT_UPDATED, path)
                    for path in result.deleted_files:
                        emit(CONTENT_DELETED, path)
                else:
                    logger.warning("Git sync pull failed: %s", result.message)
                    await _warn_if_merge_in_progress(git_backend)
            else:
                skipped_pulls += 1
                if skipped_pulls % PULL_SKIP_WARN_AFTER == 0:
                    logger.warning(
                        "Git sync has skipped %d consecutive pulls (~%ds) because a "
                        "transaction was open every time; this server is still pushing "
                        "but is no longer pulling from %s/%s.",
                        skipped_pulls,
                        skipped_pulls * interval,
                        remote,
                        branch,
                    )
                else:
                    logger.debug("Git sync pull skipped: transaction in progress")
            if Config.GIT_SYNC_ENABLED:
                ahead = await _locked(_push_if_ahead)
                if ahead > 0:
                    logger.info("Git sync: pushed %d commit(s)", ahead)
        except Exception as exc:
            logger.warning("Git sync error: %s", exc)
        await asyncio.sleep(interval)


async def _warn_if_merge_in_progress(git_backend) -> None:
    """Log an ERROR when a failed pull left a conflicted merge behind.

    ``git commit -- <pathspec>`` refuses to run during a merge, so from this
    point every write still lands on disk but no commit can be made — an
    otherwise invisible hard-down state. Resolution is deliberately manual.
    """
    try:
        if not await asyncio.to_thread(git_backend.merge_in_progress):
            return
    except Exception as exc:  # never let diagnostics break the sync loop
        logger.debug("Could not check for an in-progress merge: %s", exc)
        return
    logger.error(
        "A conflicted merge is in progress in %s (MERGE_HEAD is present). "
        "Path-scoped commits fail during a merge, so writes will keep landing "
        "on disk but NOTHING will be committed until an operator resolves it "
        "manually (resolve the conflicts and 'git commit', or 'git merge --abort').",
        Config.CONTENT_DIR,
    )


def create_app():
    """Create and configure the FastAPI application."""
    _maybe_clone_repo()
    _maybe_init_repo()
    Config.ensure_content_dir()
    filesystem = FileSystem(Config.CONTENT_DIR, include_patterns=Config.CONTENT_PATHS)

    # Initialise metrics collector (no-op when disabled).
    # In read-only/stateless mode, metrics default to disabled to avoid file
    # corruption from multiple pods writing concurrently.  Users can still opt
    # in by setting STASH_METRICS_ENABLED=true explicitly.
    init_metrics(
        db_path=str(Config.METRICS_PATH),
        enabled=Config.get_effective_metrics_enabled(),
        retention_days=Config.METRICS_RETENTION_DAYS,
    )

    # Optionally create search engine
    search_engine = _create_search_engine()
    if search_engine is not None:
        search_engine._filesystem = filesystem

    # Optionally create git backend (may raise SystemExit on misconfiguration)
    git_backend = _create_git_backend()
    if git_backend is not None and search_engine is not None:
        search_engine._git_backend = git_backend

    transaction_manager, fs_for_mcp = build_write_stack(filesystem, git_backend)

    # Create MCP http app first so we can wire its lifespan into FastAPI.
    # In read-only mode, use stateless HTTP so each request is self-contained
    # and any pod can serve any client (safe for horizontal scaling).
    mcp = create_mcp_server(
        fs_for_mcp,
        search_engine=search_engine,
        git_backend=git_backend,
        transaction_manager=transaction_manager,
    )
    mcp_http_app = mcp.http_app(path="/", stateless_http=Config.READ_ONLY)

    # Build a combined lifespan that wraps the MCP lifespan and also
    # triggers the search index build on startup.  Using on_event("startup")
    # does NOT work when a lifespan handler is set on the FastAPI app.
    mcp_lifespan = mcp_http_app.lifespan

    @asynccontextmanager
    async def _combined_lifespan(fastapi_app):
        # Set up a sync-pause event and register callbacks on the TransactionManager
        # so that git sync is suspended for the duration of any active transaction.
        sync_event = asyncio.Event()
        sync_event.set()  # set = sync allowed; cleared = sync paused during a transaction
        if transaction_manager is not None:
            transaction_manager.set_sync_callbacks(
                pause=lambda: sync_event.clear(),
                resume=lambda: sync_event.set(),
            )

        async with mcp_lifespan(fastapi_app):
            # Startup: build search index in background (non-blocking)
            get_metrics().record_server_event("startup")
            if search_engine is not None:

                async def _do_build():
                    files = filesystem.list_all_files()
                    total = await search_engine.build_index(files)
                    logger.info(f"Startup search index build complete: {total} chunks")

                task = asyncio.create_task(_do_build())
                task.add_done_callback(_task_done_callback)

            # Start periodic git sync task if configured
            sync_task = None
            if git_backend is not None and Config.GIT_SYNC_ENABLED:
                sync_task = asyncio.create_task(
                    _git_sync_loop(git_backend, search_engine, sync_event, transaction_manager),
                    name="git-sync",
                )
                sync_task.add_done_callback(_task_done_callback)

            yield

            # Shutdown: cancel the sync task gracefully
            if sync_task is not None and not sync_task.done():
                sync_task.cancel()
                try:
                    await sync_task
                except asyncio.CancelledError:
                    pass
            get_metrics().record_server_event("shutdown")
            get_metrics().close()

    app = create_api(
        filesystem, lifespan=_combined_lifespan, search_engine=search_engine,
        transaction_manager=transaction_manager,
    )
    ui_router = create_ui_router(
        filesystem, search_engine=search_engine, read_only=Config.READ_ONLY,
        transaction_manager=transaction_manager,
    )
    app.include_router(ui_router)

    # Serve vendored static assets (highlight.js, mermaid.js, etc.)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Mount FastMCP server onto FastAPI for streamable HTTP transport
    app.mount("/mcp", mcp_http_app)

    # Normalize /mcp → /mcp/ so the mounted sub-app handles requests to
    # both paths without a 307 redirect.  Redirects can break MCP clients
    # behind reverse proxies (e.g. Cloudflare) that drop the request body.
    class _MCPSlashMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http" and scope["path"] == "/mcp":
                scope = dict(scope)
                scope["path"] = "/mcp/"
            await self.app(scope, receive, send)

    app.add_middleware(_MCPSlashMiddleware)

    # Wire event bus: REST mutations emit MCP resource notifications
    def on_content_changed(event_type: str, path: str, **kwargs: str) -> None:
        logger.info(f"Content event: {event_type} {path} {kwargs}")

        # Record content lifecycle metric
        metric_event = _CONTENT_EVENT_METRIC_MAP.get(event_type)
        if metric_event:
            try:
                full_path = Config.CONTENT_DIR / path
                size = full_path.stat().st_size if full_path.is_file() else 0
            except Exception:
                size = 0
            get_metrics().record_content_event(metric_event, path, size_bytes=size)

        # Async bridge: trigger search index updates from sync event bus
        if search_engine is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("No running event loop; skipping search index update")
                return
            if event_type in ("content_created", "content_updated"):
                task = loop.create_task(
                    search_engine.index_file(path), name=f"index-{path}"
                )
                task.add_done_callback(_task_done_callback)
            elif event_type == "content_deleted":
                task = loop.create_task(
                    search_engine.remove_file(path), name=f"remove-{path}"
                )
                task.add_done_callback(_task_done_callback)
            elif event_type == "content_moved":
                source_path = kwargs.get("source_path", "")
                if source_path:
                    task = loop.create_task(
                        search_engine.move_file_index(source_path, path),
                        name=f"move-{source_path}->{path}",
                    )
                    task.add_done_callback(_task_done_callback)
                else:
                    task = loop.create_task(
                        search_engine.index_file(path), name=f"index-{path}"
                    )
                    task.add_done_callback(_task_done_callback)

    add_listener(on_content_changed)

    return app


def main():
    """Run the Stash-MCP web server."""
    logger.info("Starting Stash-MCP server...")
    logger.info(f"Server name: {Config.SERVER_NAME}")
    if Config.READ_ONLY:
        logger.info("Read-only mode enabled — write tools will not be registered")
    if Config.GIT_TRACKING:
        logger.info("Git tracking enabled")
    if Config.GIT_TRACKING and Config.GIT_AUTOCOMMIT:
        logger.info("Git autocommit enabled")
    if Config.GIT_SYNC_ENABLED:
        logger.info(
            f"Git sync enabled: remote={Config.GIT_SYNC_REMOTE} "
            f"branch={Config.GIT_SYNC_BRANCH} interval={Config.GIT_SYNC_INTERVAL}s"
        )

    app = create_app()

    logger.info(f"Server running at http://{Config.HOST}:{Config.PORT}")
    logger.info(f"API docs at http://{Config.HOST}:{Config.PORT}/docs")
    logger.info(f"UI at http://{Config.HOST}:{Config.PORT}/ui")
    transport_label = "Stateless HTTP" if Config.READ_ONLY else "Streamable HTTP"
    logger.info(f"MCP ({transport_label}) at http://{Config.HOST}:{Config.PORT}/mcp")
    if Config.READ_ONLY:
        logger.info("Stateless transport enabled — safe for horizontal scaling")

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
