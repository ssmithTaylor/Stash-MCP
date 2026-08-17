"""Main server entry point for stdio MCP transport."""

import asyncio
import logging
import sys

from .config import Config
from .filesystem import FileSystem
from .mcp_server import create_mcp_server
from .metrics import get_metrics, init_metrics

# The startup helpers are shared with the FastAPI entrypoint (main.py) and live
# in startup.py. They are imported under private aliases and called through this
# module's namespace so tests can monkeypatch stash_mcp.server._create_* the same
# way they do for main. Do not reintroduce local copies: the copies these
# replaced had drifted, leaving stdio without auto-init, without the
# autocommit/sync config validation, without the search scoping defaults, and
# without a TransactionManager at all.
from .startup import build_write_stack
from .startup import create_git_backend as _create_git_backend
from .startup import create_search_engine as _create_search_engine
from .startup import maybe_clone_repo as _maybe_clone_repo
from .startup import maybe_init_repo as _maybe_init_repo

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run the Stash MCP server over stdio."""
    logger.info("Starting Stash-MCP server...")
    if Config.READ_ONLY:
        logger.info("Read-only mode enabled — write tools will not be registered")

    # Bootstrap the content repo before ensuring the content dir exists: clone
    # wins if a remote is configured, otherwise auto-init covers the fresh-volume
    # case. Same order as create_app() in main.py — maybe_init_repo() relies on
    # running before ensure_content_dir() to detect a nested repo correctly.
    _maybe_clone_repo()
    _maybe_init_repo()

    # Ensure content directory exists
    Config.ensure_content_dir()

    # Initialise metrics collector (no-op when disabled).
    # In read-only/stateless mode, metrics default to disabled to avoid file
    # corruption from multiple pods writing concurrently.  Users can still opt
    # in by setting STASH_METRICS_ENABLED=true explicitly.
    init_metrics(
        db_path=str(Config.METRICS_PATH),
        enabled=Config.get_effective_metrics_enabled(),
        retention_days=Config.METRICS_RETENTION_DAYS,
    )

    # Initialize filesystem
    filesystem = FileSystem(Config.CONTENT_DIR, include_patterns=Config.CONTENT_PATHS)

    # Optionally create search engine and git backend
    search_engine = _create_search_engine()
    if search_engine is not None:
        search_engine._filesystem = filesystem

    git_backend = _create_git_backend()
    if git_backend is not None and search_engine is not None:
        search_engine._git_backend = git_backend

    # Same write stack the HTTP entrypoint builds: gated transactions or
    # autocommit, depending on STASH_GIT_AUTOCOMMIT. Without this, git tracking
    # over stdio would register the write tools but leave every write
    # uncommitted and unlocked.
    transaction_manager, fs_for_mcp = build_write_stack(filesystem, git_backend)

    # Create FastMCP server and run with stdio transport
    mcp = create_mcp_server(
        fs_for_mcp,
        search_engine=search_engine,
        git_backend=git_backend,
        transaction_manager=transaction_manager,
    )

    if search_engine is not None:
        files = filesystem.list_all_files()
        await search_engine.build_index(files)
        logger.info("Search index built")

    get_metrics().record_server_event("startup")
    logger.info(f"Server running with content dir: {Config.CONTENT_DIR}")
    try:
        await mcp.run_stdio_async()
    finally:
        get_metrics().record_server_event("shutdown")
        get_metrics().close()


def run():
    """Synchronous entry point for stdio MCP transport (used by uvx/extensions)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run()
