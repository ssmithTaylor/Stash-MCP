"""FastAPI app entrypoint for Stash-MCP."""

import asyncio
import logging
import os
import subprocess
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


def _maybe_clone_repo() -> None:
    """Clone remote repo into content dir if a clone URL is configured.

    Checks ``STASH_GIT_CLONE_URL`` first (legacy), then falls back to
    ``STASH_GIT_SYNC_URL`` so that a single coherent set of sync env vars is
    sufficient to bootstrap an empty content directory.
    """
    # Determine which URL / parameters to use
    if Config.GIT_CLONE_URL:
        clone_url = Config.GIT_CLONE_URL
        clone_branch = Config.GIT_CLONE_BRANCH
        clone_token = Config.GIT_CLONE_TOKEN
        clone_remote = "origin"  # GitBackend.clone always uses "origin"
        url_env_var = "STASH_GIT_CLONE_URL"
    elif Config.GIT_SYNC_URL:
        clone_url = Config.GIT_SYNC_URL
        clone_branch = Config.GIT_SYNC_BRANCH
        clone_token = Config.GIT_SYNC_TOKEN
        clone_remote = Config.GIT_SYNC_REMOTE
        url_env_var = "STASH_GIT_SYNC_URL"
    else:
        return

    content_dir = Config.CONTENT_DIR

    if content_dir.exists() and any(content_dir.iterdir()):
        git_dir = content_dir / ".git"
        if git_dir.exists():
            logger.info("Content directory already contains a git repo, skipping clone")
            if clone_url == Config.GIT_SYNC_URL:
                # Auto-enable tracking so sync can proceed without requiring
                # STASH_GIT_TRACKING=true to be set explicitly in this case.
                Config.GIT_TRACKING = True
            return
        logger.error(
            "Content directory %s is non-empty but not a git repo. "
            "Cannot clone into it. Clear the directory or remove %s.",
            content_dir,
            url_env_var,
        )
        raise SystemExit(1)

    from .git_backend import GitBackend

    logger.info(
        "Cloning %s (branch=%s) into %s",
        clone_url,
        clone_branch,
        content_dir,
    )
    try:
        GitBackend.clone(
            url=clone_url,
            target_dir=content_dir,
            branch=clone_branch,
            token=clone_token,
            recursive=Config.GIT_SYNC_RECURSIVE,
        )
    except RuntimeError as exc:
        logger.error("Clone failed: %s", exc)
        raise SystemExit(1) from exc

    # If the desired sync remote name differs from the default "origin" set by
    # GitBackend.clone(), rename it so that subsequent pulls use the right name.
    if clone_remote != "origin":
        try:
            backend = GitBackend(content_dir)
            backend.rename_remote("origin", clone_remote)
        except RuntimeError as exc:
            logger.error("Failed to rename remote: %s", exc)
            raise SystemExit(1) from exc
        logger.info("Renamed remote 'origin' to '%s'", clone_remote)

    Config.GIT_TRACKING = True
    logger.info("Clone complete. Git tracking auto-enabled.")


def _nearest_existing_ancestor(path: Path) -> Path:
    """Return the closest directory in *path*'s own chain that exists on disk.

    Returns *path* itself (resolved) if it already exists. Otherwise walks up
    through its parents until it finds one that does. The filesystem root
    always exists, so this always terminates with a real, existing directory.
    """
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            break  # reached the filesystem root; defensive, should not happen
        current = parent
    return current


def _maybe_init_repo() -> None:
    """Initialise a local git repo in the content dir when tracking is on and
    no remote is configured.

    Precedence: a configured clone/sync URL always wins. ``_maybe_clone_repo()``
    owns that path — including its hard-fail on clone failure — so this
    function is a no-op whenever ``STASH_GIT_CLONE_URL``/``STASH_GIT_SYNC_URL``
    is set. It only bootstraps a repo for the case that's otherwise a hard
    startup failure: ``STASH_GIT_TRACKING=true`` with no remote and a content
    directory that isn't a git repo yet (e.g. a fresh container volume).

    A directory that already sits *inside* a different, pre-existing git
    repository is refused rather than silently nested: ``git rev-parse
    --git-dir`` would succeed from such a directory (it resolves upward to
    the parent repo's ``.git``), so ``--show-toplevel`` is used instead and
    compared against the content dir to detect this case correctly.

    This check must not be skipped just because the content dir doesn't exist
    yet — that's the common case, since this runs before
    ``Config.ensure_content_dir()``. So the ``--show-toplevel`` probe always
    runs, from the nearest ancestor of the content dir that actually exists
    (which may be the content dir itself, or may be several levels up). If
    that resolves to a repository at all, the content dir would end up nested
    underneath it — refused unconditionally — *unless* the content dir itself
    already exists and *is* that repository's root.
    """
    if Config.GIT_CLONE_URL or Config.GIT_SYNC_URL:
        return

    if not Config.GIT_TRACKING:
        return

    content_dir = Config.CONTENT_DIR
    nearest_existing = _nearest_existing_ancestor(content_dir)

    toplevel_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=nearest_existing,
        capture_output=True,
        text=True,
    )
    if toplevel_result.returncode == 0:
        toplevel = Path(toplevel_result.stdout.strip()).resolve()
        if content_dir.exists() and toplevel == content_dir.resolve():
            # content_dir itself already exists and is the repo root.
            return
        logger.error(
            "Content directory %s is inside an existing git repository "
            "rooted at %s. Refusing to create a nested repo. Point "
            "STASH_CONTENT_ROOT at the parent repo root, set "
            "STASH_GIT_TRACKING=false, or run 'git init' in the content "
            "directory yourself if a nested repo is genuinely intended.",
            content_dir,
            toplevel,
        )
        raise SystemExit(1)
    # Non-zero: no existing ancestor of content_dir is inside a git repository.

    content_dir.mkdir(parents=True, exist_ok=True)

    logger.info("No git repository found at %s; running 'git init'", content_dir)
    init_result = subprocess.run(
        ["git", "init", str(content_dir)], capture_output=True, text=True
    )
    if init_result.returncode != 0:
        logger.error("git init failed for %s: %s", content_dir, init_result.stderr.strip())
        raise SystemExit(1)

    from .git_backend import GitBackend

    backend = GitBackend(content_dir, author_default=Config.GIT_AUTHOR_DEFAULT)
    try:
        backend.validate()  # sets local user.name/user.email if not already configured
    except RuntimeError as exc:
        logger.error("Failed to configure committer identity after git init: %s", exc)
        raise SystemExit(1) from exc

    # Commit whatever is already present (respecting .gitignore) so a
    # pre-populated content dir doesn't leave existing files untracked — an
    # untracked file looks "restorable" to a transaction abort and would be
    # deleted by it. --allow-empty guarantees a commit even when there's
    # nothing to stage, since an unborn HEAD breaks GitBackend elsewhere
    # (reset_hard, and the HEAD-relative commands used during transactions).
    add_result = subprocess.run(
        ["git", "add", "-A"], cwd=content_dir, capture_output=True, text=True
    )
    if add_result.returncode != 0:
        logger.error("git add failed during auto-init: %s", add_result.stderr.strip())
        raise SystemExit(1)

    commit_result = subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "Initial commit (stash-mcp auto-init)"],
        cwd=content_dir,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        logger.error("git commit failed during auto-init: %s", commit_result.stderr.strip())
        raise SystemExit(1)

    logger.info("Initialised local git repository at %s with an initial commit.", content_dir)


def _create_search_engine():
    """Create a SearchEngine if search is enabled, or return None."""
    if not Config.SEARCH_ENABLED:
        return None

    try:
        from .search import SearchEngine

        engine = SearchEngine(
            content_dir=Config.CONTENT_DIR,
            index_dir=Config.SEARCH_INDEX_DIR,
            embedder_model=Config.SEARCH_EMBEDDER_MODEL,
            contextual_retrieval=Config.CONTEXTUAL_RETRIEVAL,
            contextual_model=Config.CONTEXTUAL_MODEL,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_cache_dir=Config.MODEL_CACHE_DIR,
            onnx_threads=Config.SEARCH_ONNX_THREADS,
            query_prefix=Config.SEARCH_QUERY_PREFIX,
            document_prefix=Config.SEARCH_DOCUMENT_PREFIX,
            chunk_size=Config.SEARCH_CHUNK_SIZE,
            chunk_overlap=Config.SEARCH_CHUNK_OVERLAP,
            heading_context=Config.SEARCH_HEADING_CONTEXT,
            mmr_enabled=Config.SEARCH_MMR_ENABLED,
            mmr_lambda=Config.SEARCH_MMR_LAMBDA,
            max_per_file=Config.SEARCH_MAX_PER_FILE,
            candidate_pool_multiplier=Config.SEARCH_CANDIDATE_POOL_MULTIPLIER,
            recency_weight=Config.SEARCH_RECENCY_WEIGHT,
            recency_half_life_days=Config.SEARCH_RECENCY_HALF_LIFE_DAYS,
            hybrid_enabled=Config.SEARCH_HYBRID_ENABLED,
            rrf_k=Config.SEARCH_RRF_K,
            bm25_candidate_pool=Config.SEARCH_BM25_CANDIDATE_POOL,
            rerank_enabled=Config.SEARCH_RERANK_ENABLED,
            rerank_model=Config.SEARCH_RERANK_MODEL,
            rerank_candidates=Config.SEARCH_RERANK_CANDIDATES,
            rerank_margin=Config.SEARCH_RERANK_MARGIN,
            default_exclude_patterns=Config.SEARCH_EXCLUDE_PATTERNS,
            default_boost_weight=Config.SEARCH_BOOST_WEIGHT,
        )
        logger.info(
            f"Search engine initialised (model={Config.SEARCH_EMBEDDER_MODEL}, "
            f"contextual={Config.CONTEXTUAL_RETRIEVAL}, "
            f"exclude={Config.SEARCH_EXCLUDE_PATTERNS} boost={Config.SEARCH_BOOST_WEIGHT})"
        )
        return engine
    except Exception as e:
        logger.error(f"Failed to create search engine: {e}")
        return None


def _create_git_backend():
    """Validate git config and return a GitBackend, or None if tracking is off.

    Raises SystemExit on misconfiguration so the server fails fast.
    """
    if Config.GIT_AUTOCOMMIT and not Config.GIT_TRACKING:
        logger.warning(
            "STASH_GIT_AUTOCOMMIT=true has no effect without STASH_GIT_TRACKING=true; ignoring."
        )

    if Config.GIT_SYNC_ENABLED and not Config.GIT_TRACKING:
        logger.error(
            "STASH_GIT_SYNC_ENABLED=true requires STASH_GIT_TRACKING=true. "
            "Set STASH_GIT_TRACKING=true or disable sync."
        )
        raise SystemExit(1)

    if not Config.GIT_TRACKING:
        return None

    from .git_backend import GitBackend

    backend = GitBackend(
        Config.CONTENT_DIR,
        sync_token=Config.GIT_SYNC_TOKEN,
        author_default=Config.GIT_AUTHOR_DEFAULT,
    )

    try:
        backend.validate()
    except RuntimeError as exc:
        logger.error("Git tracking enabled but validation failed: %s", exc)
        raise SystemExit(1) from exc

    # Nothing should be staged between operations; clear anything left by a
    # crashed sequence (mixed reset — the worktree is untouched).
    try:
        if backend.has_staged_changes():
            logger.warning("Found staged changes at startup; unstaging (worktree kept).")
            backend.unstage()
    except RuntimeError as exc:
        logger.warning("Could not inspect the git index at startup: %s", exc)

    logger.info("Git tracking active (content_dir=%s)", Config.CONTENT_DIR)

    if Config.GIT_SYNC_ENABLED:
        if not backend.validate_remote(Config.GIT_SYNC_REMOTE):
            logger.error(
                "Git sync remote '%s' is not configured in the repository.",
                Config.GIT_SYNC_REMOTE,
            )
            raise SystemExit(1)
        logger.info(
            "Git sync enabled (remote=%s branch=%s interval=%ds)",
            Config.GIT_SYNC_REMOTE,
            Config.GIT_SYNC_BRANCH,
            Config.GIT_SYNC_INTERVAL,
        )

    return backend


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

    transaction_manager = None
    fs_for_mcp = filesystem
    if not Config.READ_ONLY and git_backend is not None:
        from .transactions import TransactionManager

        transaction_manager = TransactionManager(
            filesystem,
            git_backend,
            autocommit=Config.GIT_AUTOCOMMIT,
            author_default=Config.GIT_AUTHOR_DEFAULT,
            lock_wait=Config.TRANSACTION_LOCK_WAIT,
        )
        # Gated mode installs the delegating wrapper so writes outside a
        # transaction are rejected; autocommit mode uses the raw filesystem
        # and the manager only for the write lock / commits. Load-bearing:
        # TransactionManager._resolve_txn_for_wrapper_write requires an open
        # transaction regardless of autocommit (unlike guard()'s gate, which
        # does understand autocommit) — reverting this to an unconditional
        # `transaction_manager` would make every autocommit-mode MCP write
        # fail with a generic "No active transaction" instead of committing.
        fs_for_mcp = filesystem if Config.GIT_AUTOCOMMIT else transaction_manager
        logger.info(
            "Git write mode: %s", "autocommit" if Config.GIT_AUTOCOMMIT else "transactions"
        )

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
