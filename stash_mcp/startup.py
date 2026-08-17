"""Shared startup helpers for both server entrypoints.

``main.py`` (FastAPI/HTTP) and ``server.py`` (stdio) both have to bootstrap the
same things before they can serve: clone or initialise the content repo, build
the search engine, validate git config, and assemble the write stack. Those
steps used to be copy-pasted into both modules and drifted apart — the stdio
entrypoint silently lacked auto-init, the autocommit/sync config validation,
the scoping defaults passed to ``SearchEngine``, and the ``TransactionManager``
wiring entirely, so ``STASH_GIT_TRACKING=true`` behaved differently depending
on which transport you started. This module is the single home for that logic;
both entrypoints import from here and neither keeps a private copy.

Both entrypoints import these under their historical private aliases
(``maybe_clone_repo`` and friends) and call them through their own module
namespace, so monkeypatching ``stash_mcp.main.create_search_engine`` still
works the way the tests expect.
"""

import logging
import os
import subprocess
from pathlib import Path

from .config import Config
from .filesystem import FileSystem

logger = logging.getLogger(__name__)


def maybe_clone_repo() -> None:
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


def nearest_existing_ancestor(path: Path) -> Path:
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


def maybe_init_repo() -> None:
    """Initialise a local git repo in the content dir when tracking is on and
    no remote is configured.

    Precedence: a configured clone/sync URL always wins. ``maybe_clone_repo()``
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
    nearest_existing = nearest_existing_ancestor(content_dir)

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


def create_search_engine():
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


def create_git_backend():
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


def build_write_stack(filesystem: FileSystem, git_backend):
    """Assemble the write stack, returning ``(transaction_manager, fs_for_mcp)``.

    Returns ``(None, filesystem)`` when there is nothing to coordinate — no git
    backend, or read-only mode — so callers can use the result unconditionally.

    Gated mode installs the delegating wrapper so writes outside a transaction
    are rejected; autocommit mode uses the raw filesystem and the manager only
    for the write lock / commits. Load-bearing:
    ``TransactionManager._resolve_txn_for_wrapper_write`` requires an open
    transaction regardless of autocommit (unlike ``guard()``'s gate, which does
    understand autocommit) — reverting this to an unconditional
    ``transaction_manager`` would make every autocommit-mode MCP write fail with
    a generic "No active transaction" instead of committing.
    """
    if Config.READ_ONLY or git_backend is None:
        return None, filesystem

    from .transactions import TransactionManager

    transaction_manager = TransactionManager(
        filesystem,
        git_backend,
        autocommit=Config.GIT_AUTOCOMMIT,
        author_default=Config.GIT_AUTHOR_DEFAULT,
        lock_wait=Config.TRANSACTION_LOCK_WAIT,
    )
    fs_for_mcp = filesystem if Config.GIT_AUTOCOMMIT else transaction_manager
    logger.info(
        "Git write mode: %s", "autocommit" if Config.GIT_AUTOCOMMIT else "transactions"
    )
    return transaction_manager, fs_for_mcp
