"""MCP Server implementation for Stash using FastMCP."""

import asyncio
import functools
import hashlib
import inspect
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.resources import FunctionResource
from fastmcp.server.context import Context
from mcp.types import ToolAnnotations
from pydantic import AnyUrl, BaseModel, Field

from .config import Config
from .events import CONTENT_CREATED, CONTENT_DELETED, CONTENT_MOVED, CONTENT_UPDATED, emit
from .filesystem import (
    FileNotFoundError,
    FileSystem,
    InvalidPathError,
    glob_to_regex,
    normalize_glob,
)
from .frontmatter import extract_metadata, merge_frontmatter
from .headings import scan_headings
from .metrics import get_metrics
from .search import reject_path_traversal
from .transactions import TransactionError, TransactionManager, default_commit_message

logger = logging.getLogger(__name__)


class EditOperation(BaseModel):
    """A single string-replacement edit."""

    old_string: str = Field(description="The exact text to find in the file")
    new_string: str = Field(description="The text to replace it with")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence (True) or require exactly one match (False)",
    )


class FileEditOperation(BaseModel):
    """Edits targeting a single file, used by edit_content_batch."""

    file_path: str = Field(description="File path relative to content root")
    sha: str = Field(
        description="SHA-256 hex digest of the file's current full content, from read_content"
    )
    edits: list[EditOperation] = Field(description="Ordered list of edits to apply")


class MoveOperation(BaseModel):
    """A single file move operation."""

    source_path: str = Field(description="Current file path relative to content root")
    dest_path: str = Field(
        description="New file path relative to content root; must not already exist"
    )


# Shared parameter annotations so every tool's JSON schema carries
# per-parameter descriptions (clients surface these to agents).
ContentPath = Annotated[
    str,
    Field(description="File path relative to content root (POSIX-style, no leading slash)"),
]
FileSha = Annotated[
    str,
    Field(
        description="SHA-256 hex digest of the file's current full content, "
        "as returned by read_content"
    ),
]
MaxLines = Annotated[
    int | None,
    Field(
        ge=1,
        description="Maximum number of lines to return from the beginning of the "
        "file. Omit to return the full content — there is no offset parameter",
    ),
]
CommitMessage = Annotated[
    str | None,
    Field(
        description="Git commit message for this write (git-tracked servers only; ignored "
        "otherwise). Inside an open transaction it is recorded as a bullet in the "
        "transaction's commit body instead"
    ),
]
CommitAuthor = Annotated[
    str | None,
    Field(description='Git author as "Name <email>" (git-tracked servers only; ignored otherwise)'),
]

# Appended to write-tool descriptions when writes are transaction-gated.
_TXN_NOTE = (
    "\n\nNote: this server gates writes behind transactions. Call "
    "start_content_transaction before using this tool, then "
    "commit_content_transaction to persist the changes."
)

# Appended to write-tool descriptions when writes are auto-committed.
_AUTOCOMMIT_NOTE = (
    "\n\nNote: this server commits each write to git automatically; pass "
    "commit_message and author to attribute the change. To group several "
    "writes into one commit, call start_content_transaction first and "
    "commit_content_transaction when done."
)


def _build_instructions(
    *,
    read_only: bool,
    search_enabled: bool,
    git_enabled: bool,
    transactions_active: bool,
    autocommit: bool = False,
) -> str:
    """Assemble the server-level instructions string sent to MCP clients."""
    parts = [
        "Stash is a file-backed document store. All paths are POSIX-style and "
        "relative to the content root, with no leading slash (e.g. 'docs/guide.md'). "
        "Start with README.md at the content root (the store index) and the "
        "README.md of the root you are working in; use list_content(path=<root>) "
        "to browse it and find_content(pattern, path_prefix=<root>) to search "
        "within it. list_content(recursive=true) lists every file and is only useful for "
        "small stores."
    ]
    if read_only:
        parts.append(
            "This server is read-only: content can be listed and read but not modified."
        )
    else:
        parts.append(
            "Writing a file creates missing parent directories automatically; there "
            "is no separate mkdir step. To add knowledge to an existing markdown "
            "doc: list_content to find it, inspect_content_structure to see its "
            "heading outline, read_content for the current text and sha, then "
            "edit_content to change just the relevant part instead of rewriting "
            "the file. overwrite_content replaces the full file, update_metadata "
            "sets or removes frontmatter keys, create_content is only for files "
            "that do not exist yet, and delete_content also requires the sha."
        )
    if transactions_active and autocommit:
        parts.append(
            "Writes are committed to git automatically, one commit per write; pass "
            "commit_message (and optionally author) on create/edit/overwrite/move/"
            "delete to attribute the change. To land several writes as one commit, "
            "call start_content_transaction first, make the changes, then "
            "commit_content_transaction (or abort_content_transaction to revert "
            "them). Transactions from different sessions may be open at the same "
            "time; idle transactions are auto-aborted after a timeout."
        )
    elif transactions_active:
        parts.append(
            "Writes are gated behind transactions: call start_content_transaction "
            "before any create/edit/overwrite/move/delete, make the changes, then "
            "commit_content_transaction with a commit message to persist them (or "
            "abort_content_transaction to discard). Transactions from different "
            "sessions may be open at the same time; idle transactions are "
            "auto-aborted after a timeout."
        )
    if search_enabled:
        parts.append(
            "search_content finds content by meaning and returns ranked snippets "
            "with the Section (heading path) each came from; scope with "
            "path_prefix (hard — only these roots) or boost_prefix (soft — "
            "prefers without hiding the rest). The server may be configured to "
            "exclude certain paths (e.g. working directories) from results by "
            "default; pass include_excluded=true to search them too. The Section "
            "line names the heading a snippet came from, so follow up with "
            "read_content and jump straight to that part instead of scanning "
            "the whole file."
        )
    if git_enabled:
        parts.append(
            "log_content, diff_content, and blame_content expose the git history "
            "of any file."
        )
    return "\n\n".join(parts)


# Mime type mapping for common extensions
MIME_TYPES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".ts": "application/typescript",
    ".py": "text/x-python",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".rst": "text/x-rst",
    ".log": "text/plain",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".bmp": "image/bmp",
    ".mmd": "text/x-mermaid",
    ".mermaid": "text/x-mermaid",
    ".gantt": "text/x-gantt",
}

# Only files matching this name are exposed as MCP resources.
# All other files remain accessible via tools and the resource template.
RESOURCE_FILENAME = "README.md"


def _is_resource_file(path: str) -> bool:
    """Check if a file should be exposed as an MCP resource."""
    # Normalize to POSIX-style path and remove trailing slashes to handle
    # inputs with OS-native separators or accidental trailing separators.
    normalized = path.replace("\\", "/").rstrip("/")
    return PurePosixPath(normalized).name == RESOURCE_FILENAME


def _get_mime_type(path: str) -> str:
    """Get mime type for a file path based on extension."""
    suffix = PurePosixPath(path).suffix.lower()
    return MIME_TYPES.get(suffix, "text/plain")


# Mime types that find_content treats as text-searchable. Anything outside
# this set (images, binaries) is skipped without attempting to read.
_SEARCHABLE_MIMES: frozenset[str] = frozenset({
    "text/markdown",
    "text/plain",
    "application/json",
    "application/x-yaml",
    "application/xml",
    "text/html",
    "text/css",
    "application/javascript",
    "application/typescript",
    "text/x-python",
    "text/csv",
    "text/tab-separated-values",
    "application/toml",
    "text/x-rst",
    "text/x-mermaid",
    "text/x-gantt",
    # SVG is XML text — may legitimately contain searchable strings.
    "image/svg+xml",
})


def _is_searchable(path: str) -> bool:
    # Require a known extension. Falling back through _get_mime_type would
    # treat unknown suffixes (.pdf, .bin, no extension at all) as
    # "text/plain" and let them through the allowlist.
    suffix = PurePosixPath(path).suffix.lower()
    mime = MIME_TYPES.get(suffix)
    return mime is not None and mime in _SEARCHABLE_MIMES


def _get_description(fs: FileSystem, path: str) -> str:
    """Describe a file by its first content line.

    Skips YAML frontmatter, a leading provenance blockquote, blank lines and
    HTML comments so the description is the title, not the metadata.
    """
    try:
        content = fs.read_file(path)
        _, body = extract_metadata(content)
        in_comment = False
        after_delimiter = False
        for raw in body.splitlines():
            line = raw.strip()
            if in_comment:
                if "-->" in line:
                    in_comment = False
                continue
            if not line or line.startswith(">"):
                after_delimiter = False
                continue
            if line in ("---", "..."):
                after_delimiter = True
                continue
            if after_delimiter and ":" in line and not line.startswith("#"):
                continue
            after_delimiter = False
            if line.startswith("<!--"):
                in_comment = "-->" not in line
                continue
            if line.startswith("#"):
                line = line.lstrip("# ").strip()
            return line[:100] if line else f"Content file: {path}"
        return f"Content file: {path}"
    except Exception:
        return f"Content file: {path}"


def _count_lines(text: str) -> int:
    """Number of newline-delimited lines in *text* (0 for empty).

    Content read through the filesystem layer is universal-newline
    translated, so counting "\\n" matches line semantics without
    materializing a splitlines() list.
    """
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _truncate_lines(text: str, max_lines: int) -> tuple[str, bool]:
    """Return the first *max_lines* lines of *text* and whether it was cut.

    Equivalent to joining the first *max_lines* entries of
    ``text.splitlines(keepends=True)`` but without allocating the list.
    """
    pos = -1
    for _ in range(max_lines):
        pos = text.find("\n", pos + 1)
        if pos == -1:
            return text, False
    if pos + 1 >= len(text):
        return text, False
    return text[: pos + 1], True


def _apply_edits(content: str, edits: list[EditOperation], path: str) -> str:
    """Apply a sequence of string-replacement edits to *content*.

    Raises ``ValueError`` if any edit is invalid (empty old_string, not found,
    or ambiguous when replace_all is False).
    """
    for edit in edits:
        if not edit.old_string:
            raise ValueError(f"old_string must not be empty (file: {path})")
        if edit.old_string not in content:
            raise ValueError(
                f"old_string not found in '{path}'. The file content may have changed."
            )
        if not edit.replace_all and content.count(edit.old_string) > 1:
            raise ValueError(
                f"old_string appears {content.count(edit.old_string)} times in '{path}'. "
                "Set replace_all=True or provide a more specific old_string."
            )
        if edit.replace_all:
            content = content.replace(edit.old_string, edit.new_string)
        else:
            content = content.replace(edit.old_string, edit.new_string, 1)
    return content


def _build_heading_tree(flat: list[dict]) -> list[dict]:
    """Convert a flat list of headings into a nested tree."""
    root: list[dict] = []
    stack: list[dict] = []

    for heading in flat:
        while stack and stack[-1]["level"] >= heading["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(heading)
        else:
            root.append(heading)

        stack.append(heading)

    return root


def parse_markdown_structure(content: str, line_offset: int = 0) -> list[dict]:
    """Parse markdown content and return a nested heading structure.

    *line_offset* is added to every reported line number, so a caller that
    parses a document's *body* still reports lines relative to the original
    file (see ``_document_outline``).
    """
    flat_headings = [
        {
            "heading": h.text,
            "level": h.level,
            "line_number": h.line + line_offset,
            "children": [],
        }
        for h in scan_headings(content)
    ]
    return _build_heading_tree(flat_headings)


def _document_outline(content: str) -> tuple[list[dict], str | None, dict[str, str]]:
    """Return ``(sections, title, metadata)`` for a markdown document.

    Headings are scanned over the *body*, with any YAML frontmatter block
    stripped first: a ``#`` comment inside frontmatter matches the heading
    regex, so parsing the raw file reports it as a document heading and — if
    it is the first one — as the document ``title``, burying the real H1.
    That also contradicts the search indexer, which runs ``scan_headings``
    over the body. Line numbers are shifted back by the number of lines the
    frontmatter consumed, so they still point at the original file.
    """
    metadata, body = extract_metadata(content)
    line_offset = content[: len(content) - len(body)].count("\n")
    sections = parse_markdown_structure(body, line_offset)
    title = next((h["heading"] for h in sections if h["level"] == 1), None)
    return sections, title, metadata


def create_mcp_server(
    filesystem: FileSystem,
    search_engine=None,
    git_backend=None,
    transaction_manager: TransactionManager | None = None,
) -> FastMCP:
    """Create and configure the FastMCP server.

    Args:
        filesystem: Filesystem instance for content management
        search_engine: Optional SearchEngine instance for semantic search
        git_backend: Optional GitBackend instance for git tools
        transaction_manager: Optional TransactionManager providing the write
            lock, autocommit and transactions. When omitted and *filesystem*
            is itself a TransactionManager (gated mode), that instance is used.

    Returns:
        Configured FastMCP server
    """
    tm: TransactionManager | None = transaction_manager
    if tm is None and isinstance(filesystem, TransactionManager):
        tm = filesystem

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
        """Lifespan handler to inject filesystem into context."""
        yield {"fs": filesystem}

    transactions_active = not Config.READ_ONLY and git_backend is not None and tm is not None
    autocommit = bool(tm.autocommit) if tm is not None else False

    mcp = FastMCP(
        name=Config.SERVER_NAME,
        version=Config.SERVER_VERSION,
        lifespan=lifespan,
        instructions=_build_instructions(
            read_only=Config.READ_ONLY,
            search_enabled=search_engine is not None,
            git_enabled=git_backend is not None,
            transactions_active=transactions_active,
            autocommit=autocommit,
        ),
    )

    # Wrap mcp.tool() so every registered tool is automatically timed and
    # its outcome recorded in the metrics collector.  Using functools.wraps
    # preserves the original signature so FastMCP generates the correct schema.
    _original_mcp_tool = mcp.tool

    def _instrumented_tool(*deco_args, **deco_kwargs):
        orig_decorator = _original_mcp_tool(*deco_args, **deco_kwargs)

        def patching_decorator(fn):
            tool_name = fn.__name__

            @functools.wraps(fn)
            async def _tracked(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    get_metrics().record_tool_call(
                        tool_name, (time.perf_counter() - t0) * 1000, True
                    )
                    return result
                except Exception as exc:
                    get_metrics().record_tool_call(
                        tool_name,
                        (time.perf_counter() - t0) * 1000,
                        False,
                        type(exc).__name__,
                    )
                    raise

            return orig_decorator(_tracked)

        return patching_decorator

    mcp.tool = _instrumented_tool

    def _write_tool(**tool_kwargs):
        """mcp.tool wrapper for mutating tools.

        When writes are transaction-gated, appends a note to the tool
        description so agents learn the requirement up front instead of
        from a failed first call.
        """

        def decorate(fn):
            if transactions_active:
                fn.__doc__ = inspect.cleandoc(fn.__doc__ or "") + (
                    _AUTOCOMMIT_NOTE if autocommit else _TXN_NOTE
                )
            return mcp.tool(**tool_kwargs)(fn)

        return decorate

    def _session_id(ctx: Context | None) -> str | None:
        try:
            return str(id(ctx.session)) if ctx is not None else None
        except Exception:
            return None

    @asynccontextmanager
    async def _write_guard(ctx: Context | None):
        """Hold the write lock (and enforce the transaction gate) for one mutation."""
        if tm is None:
            yield
            return
        try:
            async with tm.guard(_session_id(ctx)):
                yield
        except TransactionError as exc:
            raise ValueError(str(exc)) from exc

    async def _after_write(
        ctx: Context | None,
        paths: list[str],
        commit_message: str | None,
        author: str | None,
        default_message: str,
    ) -> str | None:
        """Record in the caller's transaction or autocommit; None when git is off."""
        if tm is None:
            return None
        try:
            return await tm.after_write(
                paths,
                session_id=_session_id(ctx),
                message=commit_message,
                author=author,
                default_message=default_message,
            )
        except TransactionError as exc:
            raise ValueError(str(exc)) from exc

    def _with_commit(text: str, commit: str | None) -> str:
        return f"{text} (commit {commit})" if commit else text

    # --- Resources ---

    # Register only README.md files as resources (for resources/list).
    # All other files are accessible via tools and the resource template.
    for file_path in filesystem.list_all_files():
        if not _is_resource_file(file_path):
            continue
        uri = f"stash://{file_path}"
        mime = _get_mime_type(file_path)
        desc = _get_description(filesystem, file_path)
        fp = file_path  # capture for closure

        mcp.add_resource(
            FunctionResource(
                uri=AnyUrl(uri),
                name=file_path,
                description=desc,
                mime_type=mime,
                fn=lambda _fp=fp: filesystem.read_file(_fp),
            )
        )

    def _register_resource(path: str) -> bool:
        """Add a file to the MCP resource registry if it is a README.md.

        Returns:
            True if a resource was registered, False otherwise.
        """
        if not _is_resource_file(path):
            return False
        uri = f"stash://{path}"
        mcp.add_resource(FunctionResource(
            uri=AnyUrl(uri), name=path,
            description=_get_description(filesystem, path),
            mime_type=_get_mime_type(path),
            fn=lambda _fp=path: filesystem.read_file(_fp),
        ))
        return True

    def _unregister_resource(path: str) -> bool:
        """Remove a file from the MCP resource registry.

        Returns:
            True if a resource was removed, False otherwise.
        """
        if not _is_resource_file(path):
            return False
        uri_key = f"stash://{path}"
        try:
            # fastmcp 3.x: use public local_provider API
            mcp.local_provider.remove_resource(uri_key)
            return True
        except AttributeError:
            # fastmcp 2.x: ResourceManager exposes _resources dict directly
            return mcp._resource_manager._resources.pop(uri_key, None) is not None
        except KeyError:
            return False

    # Resource template for dynamic access (resources/templates/list)
    @mcp.resource("stash://{path}", mime_type="text/plain", description="Read any file by path")
    def read_resource(path: str) -> str:
        """Read a file by its path."""
        try:
            return filesystem.read_file(path)
        except FileNotFoundError:
            raise ValueError(f"Resource not found: stash://{path}")
        except InvalidPathError as e:
            raise ValueError(f"Invalid resource path: {e}")

    # --- Tools ---

    # --- Write tools (only registered when not in read-only mode) ---

    if not Config.READ_ONLY:

        @_write_tool(
            annotations=ToolAnnotations(
                title="Create file",
                readOnlyHint=False,
                destructiveHint=False,
                openWorldHint=False,
            )
        )
        async def create_content(
            path: ContentPath,
            content: Annotated[str, Field(description="File content (UTF-8 text)")],
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> str:
            """
            Create a new file. Errors if the file already exists.

            Missing parent directories are created automatically — writing a
            file is also the only way to create a directory.

            Args:
                path: File path relative to content root
                content: File content
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            """
            async with _write_guard(ctx):
                if filesystem.file_exists(path):
                    raise ValueError(
                        f"File already exists: {path}. Use overwrite_content to replace "
                        "it or edit_content for targeted edits."
                    )
                filesystem.write_file(path, content)
                commit = await _after_write(ctx, [path], commit_message, author, f"Create {path}")
            if _register_resource(path):
                await ctx.send_resource_list_changed()
            emit(CONTENT_CREATED, path)
            logger.info(f"Created: {path}")
            return _with_commit(f"Created: {path}", commit)

        @_write_tool(
            annotations=ToolAnnotations(
                title="Overwrite file content",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            )
        )
        async def overwrite_content(
            path: ContentPath,
            content: Annotated[
                str, Field(description="New full file content (replaces everything)")
            ],
            sha: FileSha,
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> str:
            """
            Replace the full content of an existing file.

            The file must already exist (use create_content for new files).
            For small changes prefer edit_content, which replaces exact
            strings instead of the whole file.

            Args:
                path: File path relative to content root
                content: New file content
                sha: SHA-256 hex digest of the current file content (from read_content)
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            """
            async with _write_guard(ctx):
                if filesystem.file_exists(path):
                    current = filesystem.read_file(path)
                    current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
                    if sha != current_sha:
                        raise ValueError(
                            f"SHA mismatch for '{path}': expected {current_sha}, got {sha}. "
                            "The file may have changed since it was last read."
                        )
                else:
                    raise FileNotFoundError(
                        f"File '{path}' does not exist. Use create_content for new files."
                    )
                filesystem.write_file(path, content)
                commit = await _after_write(ctx, [path], commit_message, author, f"Update {path}")
            if _is_resource_file(path):
                uri = AnyUrl(f"stash://{path}")
                await ctx.session.send_resource_updated(uri=uri)
            emit(CONTENT_UPDATED, path)
            logger.info(f"Updated: {path}")
            return _with_commit(f"Updated: {path}", commit)

        @_write_tool(
            annotations=ToolAnnotations(
                title="Edit file",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            )
        )
        async def edit_content(
            file_path: ContentPath,
            sha: FileSha,
            edits: Annotated[
                list[EditOperation],
                Field(min_length=1, description="Ordered list of edits to apply"),
            ],
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> dict:
            """
            Apply targeted string-replacement edits to an existing file.

            Each edit replaces an exact occurrence of old_string with new_string.
            Edits are applied sequentially — later edits see the result of earlier
            ones. All edits are validated in memory and the file is written once;
            if any edit fails, nothing is written.

            To change only YAML frontmatter keys, prefer update_metadata.

            Args:
                file_path: File path relative to content root
                sha: SHA-256 hex digest of the current file content (from read_content)
                edits: Ordered list of edit operations to apply
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            Returns:
                A dict with path, result status, new_sha, and commit (git hash,
                or null when nothing was committed — e.g. inside a transaction)
            """
            async with _write_guard(ctx):
                current = filesystem.read_file(file_path)
                current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
                if sha != current_sha:
                    raise ValueError(
                        f"SHA mismatch for '{file_path}': expected {current_sha}, got {sha}. "
                        "The file may have changed since it was last read."
                    )
                new_content = _apply_edits(current, edits, file_path)
                filesystem.write_file(file_path, new_content)
                commit = await _after_write(
                    ctx, [file_path], commit_message, author, f"Update {file_path}"
                )
            if _is_resource_file(file_path):
                uri = AnyUrl(f"stash://{file_path}")
                await ctx.session.send_resource_updated(uri=uri)
            emit(CONTENT_UPDATED, file_path)
            logger.info(f"Edited: {file_path}")
            new_sha = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            return {"path": file_path, "result": "ok", "new_sha": new_sha, "commit": commit}

        @_write_tool(
            annotations=ToolAnnotations(
                title="Edit multiple files",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            )
        )
        async def edit_content_batch(
            edit_operations: Annotated[
                list[FileEditOperation],
                Field(
                    min_length=1,
                    max_length=10,
                    description="Per-file edit operations (max 10 files, no duplicates)",
                ),
            ],
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> dict:
            """
            Apply string-replacement edits to up to 10 files in one call;
            every file is validated before any is written.

            All validations run before any writes — if any file fails validation
            the entire operation is aborted and no files are modified.

            Args:
                edit_operations: List of per-file edit operations (max 10)
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            Returns:
                A dict with a results list (path, result status, and new_sha per
                file) and a top-level commit (git hash for the single commit
                covering every file, or null when nothing was committed — e.g.
                inside a transaction)
            """
            if len(edit_operations) == 0:
                raise ValueError("At least one edit operation is required.")
            if len(edit_operations) > 10:
                raise ValueError(
                    f"Maximum 10 files per batch edit. Got {len(edit_operations)}."
                )
            # Reject duplicate file paths
            paths = [op.file_path for op in edit_operations]
            if len(paths) != len(set(paths)):
                raise ValueError("Duplicate file_path entries are not allowed in a single edit_content_batch call.")

            async with _write_guard(ctx):
                # Phase 1: read all files and validate SHAs
                originals: dict[str, str] = {}
                for op in edit_operations:
                    current = filesystem.read_file(op.file_path)
                    current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
                    if op.sha != current_sha:
                        raise ValueError(
                            f"SHA mismatch for '{op.file_path}': expected {current_sha}, "
                            f"got {op.sha}. The file may have changed since it was last read."
                        )
                    originals[op.file_path] = current

                # Phase 2: apply all edits in memory
                new_contents: dict[str, str] = {}
                for op in edit_operations:
                    new_contents[op.file_path] = _apply_edits(
                        originals[op.file_path], op.edits, op.file_path
                    )

                # Phase 3: write all files, collecting what to notify about.
                # Notifications are deliberately deferred until the guard has
                # closed (see below).
                results = []
                updated_uris: list[AnyUrl] = []
                for op in edit_operations:
                    filesystem.write_file(op.file_path, new_contents[op.file_path])
                    if _is_resource_file(op.file_path):
                        updated_uris.append(AnyUrl(f"stash://{op.file_path}"))
                    logger.info(f"Edited: {op.file_path}")
                    new_sha = hashlib.sha256(new_contents[op.file_path].encode("utf-8")).hexdigest()
                    results.append({"path": op.file_path, "result": "ok", "new_sha": new_sha})

                commit = await _after_write(
                    ctx,
                    [op.file_path for op in edit_operations],
                    commit_message,
                    author,
                    default_commit_message([op.file_path for op in edit_operations]),
                )

            # send_resource_updated writes to this session's SSE stream, so a
            # slow (not disconnected) client applies backpressure here. Doing
            # it inside the guard would hold the process-wide write lock
            # across a client round-trip and stall every other writer.
            for uri in updated_uris:
                await ctx.session.send_resource_updated(uri=uri)
            for op in edit_operations:
                emit(CONTENT_UPDATED, op.file_path)

            return {"results": results, "commit": commit}

        @_write_tool(
            annotations=ToolAnnotations(
                title="Delete file",
                readOnlyHint=False,
                destructiveHint=True,
                openWorldHint=False,
            )
        )
        async def delete_content(
            path: ContentPath,
            sha: FileSha,
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> str:
            """
            Delete a content file.

            On git-tracked servers the deletion is recoverable from git
            history; without git tracking it is permanent.

            Args:
                path: File path relative to content root
                sha: SHA-256 hex digest of the current file content (from read_content)
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            Returns:
                Confirmation message
            """
            async with _write_guard(ctx):
                current = filesystem.read_file(path)
                current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
                if sha != current_sha:
                    raise ValueError(
                        f"SHA mismatch for '{path}': expected {current_sha}, got {sha}. "
                        "The file may have changed since it was last read."
                    )
                filesystem.delete_file(path)
                commit = await _after_write(ctx, [path], commit_message, author, f"Delete {path}")
            if _unregister_resource(path):
                await ctx.send_resource_list_changed()
            emit(CONTENT_DELETED, path)
            logger.info(f"Deleted: {path}")
            return _with_commit(f"Deleted: {path}", commit)

        @_write_tool(
            annotations=ToolAnnotations(
                title="Update document metadata",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            )
        )
        async def update_metadata(
            path: ContentPath,
            values: Annotated[
                dict[str, str],
                Field(description="Frontmatter keys to set (values are written as strings)"),
            ],
            ctx: Context,
            unset: Annotated[
                list[str], Field(description="Frontmatter keys to remove"),
            ] = [],
            sha: Annotated[
                str | None,
                Field(
                    description="Optional staleness check: SHA-256 of the current "
                    "full content, from read_content. Omit to apply the merge "
                    "without checking whether the file changed since you read it "
                    "(the body is preserved either way)."
                ),
            ] = None,
        ) -> dict:
            """Set or remove keys in a file's YAML frontmatter without touching the body.

            Markdown files only (.md, .markdown) — a YAML block prepended to a
            .py/.json/.yaml/.csv file would silently corrupt it.

            Creates the frontmatter block when the file has none, preserves the
            other keys and their order, and leaves the body byte-for-byte
            unchanged. Use this for provenance/freshness fields (verified,
            review_every, owner, describes, layer, status, stale_anchors)
            instead of string-editing the YAML.

            Args:
                path: Markdown file path (.md or .markdown) relative to content root
                values: Keys to set
                unset: Keys to remove
                sha: Optional current content sha (as returned by read_content)
            Returns:
                A dict with 'path', 'metadata' and 'new_sha'. 'metadata' is the
                document's full metadata after the write — frontmatter merged
                over the leading blockquote, exactly what read_content and
                search results report — not just the frontmatter keys.
            """
            if not values and not unset:
                raise ValueError("Provide at least one key in values or unset.")
            suffix = PurePosixPath(path).suffix.lower()
            if suffix not in {".md", ".markdown"}:
                raise ValueError(
                    f"update_metadata only supports markdown files (.md, .markdown). Got: {path}"
                )
            current = filesystem.read_file(path)
            if sha is not None:
                current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
                if sha != current_sha:
                    raise ValueError(
                        f"SHA mismatch for '{path}': expected {current_sha}, got {sha}. "
                        "The file may have changed since it was last read."
                    )
            # merge_frontmatter's own second return value is frontmatter-only
            # (the truthfulness guard in that module is defined against it, so
            # it must stay that way). Report the same shape every other
            # surface does — frontmatter over leading blockquote — or an agent
            # that sets one key and reads the answer back concludes it just
            # wiped the blockquote-provided keys.
            new_content, _ = merge_frontmatter(current, values, unset)
            metadata = extract_metadata(new_content)[0]
            filesystem.write_file(path, new_content)
            if _is_resource_file(path):
                uri = AnyUrl(f"stash://{path}")
                await ctx.session.send_resource_updated(uri=uri)
            emit(CONTENT_UPDATED, path)
            logger.info(f"Metadata updated: {path}")
            new_sha = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            return {"path": path, "metadata": metadata, "new_sha": new_sha}

    # --- Read-only tools (always registered) ---

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Read file content",
            readOnlyHint=True,
            openWorldHint=False,
        )
    )
    async def read_content(
        path: ContentPath,
        max_lines: MaxLines = None,
    ) -> dict:
        """
        Read and return the contents of a file along with its SHA-256 hash.
        The sha is the concurrency token required by the write tools
        (overwrite_content, edit_content, edit_content_batch, delete_content)
        when they are registered.

        Args:
            path: File path relative to content root
            max_lines: Optional maximum number of lines to return from the
                beginning of the file. If omitted, returns the full file.
                There is no offset parameter — to read past the truncation
                point, call again without max_lines.
        Returns:
            A dict with 'content' (file text), 'sha' (SHA-256 hex digest of
            the FULL file, even when truncated), 'truncated' (bool),
            'total_lines' (line count of the full file), and 'metadata'
            (frontmatter/blockquote key-values)
        """
        content = await asyncio.to_thread(filesystem.read_file, path)
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        total_lines = _count_lines(content)
        metadata = extract_metadata(content)[0]
        truncated = False
        if max_lines is not None:
            if max_lines < 1:
                raise ValueError("max_lines must be a positive integer.")
            content, truncated = _truncate_lines(content, max_lines)
        return {
            "content": content,
            "sha": sha,
            "truncated": truncated,
            "total_lines": total_lines,
            "metadata": metadata,
        }

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Read multiple files",
            readOnlyHint=True,
            openWorldHint=False,
        )
    )
    async def read_content_batch(
        paths: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=10,
                description="File paths relative to content root (max 10, no duplicates)",
            ),
        ],
        max_lines: MaxLines = None,
    ) -> dict:
        """Read multiple files and return their contents with SHA-256 hashes.

        Reads up to 10 files in a single call. Each file's sha is the
        concurrency token required by the write tools (overwrite_content,
        edit_content, edit_content_batch, delete_content) when they are
        registered.

        Args:
            paths: List of file paths relative to content root (max 10)
            max_lines: Optional maximum number of lines to return from the
                beginning of each file. If omitted, returns full content.
        Returns:
            A dict with 'results' list, each containing 'path', 'content',
            'sha', 'truncated', 'total_lines', 'metadata' (frontmatter/blockquote
            key-values; null on error), and 'error' (null on success; per-file
            failures set 'error' without failing the whole call)
        """
        if not paths:
            raise ValueError("At least one path is required.")
        if len(paths) > 10:
            raise ValueError(f"Maximum 10 files per batch read. Got {len(paths)}.")
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate paths are not allowed in a single batch read.")
        if max_lines is not None and max_lines < 1:
            raise ValueError("max_lines must be a positive integer.")

        results = []
        for path in paths:
            try:
                content = await asyncio.to_thread(filesystem.read_file, path)
                sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                total_lines = _count_lines(content)
                metadata = extract_metadata(content)[0]
                truncated = False
                if max_lines is not None:
                    content, truncated = _truncate_lines(content, max_lines)
                results.append({
                    "path": path, "content": content, "sha": sha,
                    "truncated": truncated, "total_lines": total_lines,
                    "metadata": metadata, "error": None,
                })
            except (FileNotFoundError, InvalidPathError) as exc:
                results.append({
                    "path": path, "content": None, "sha": None,
                    "truncated": False, "total_lines": None,
                    "metadata": None, "error": str(exc),
                })
        return {"results": results}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="List files and directories",
            readOnlyHint=True,
            openWorldHint=False,
        )
    )
    async def list_content(
        path: Annotated[
            str,
            Field(description="Directory path relative to content root; empty string is the root"),
        ] = "",
        recursive: Annotated[
            bool,
            Field(description="If true, list every file under path as full relative paths"),
        ] = False,
        max_depth: Annotated[
            int | None,
            Field(ge=1, description="Bound the recursive listing to this many levels below "
                  "path (1 = direct children only); implies recursive"),
        ] = None,
        glob: Annotated[
            str | None,
            Field(description="Glob over full relative paths (STASH_CONTENT_PATHS dialect: *, ?, "
                  "**), e.g. 'projects/*/services/*.md'; implies recursive. Matched from the "
                  "content root, NOT relative to path, so put the directory in the glob "
                  "itself: glob='projects/x/*.md', not path='projects/x' + glob='*.md'"),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, description="Maximum entries returned (default 500)"),
        ] = 500,
        with_metadata: Annotated[
            bool,
            Field(description="Return JSON rows for markdown files with size and frontmatter "
                  "metadata instead of the text listing; implies recursive"),
        ] = False,
    ) -> str | dict:
        """List files and directories in the content store.

        Non-recursive listings show one entry per line with a 📁 prefix for
        directories and 📄 for files; entries are names only, so join them
        with *path* to build full paths. Recursive listings return full
        relative file paths, one per line, with no prefixes. Hidden files
        (dotfiles) are excluded. Large stores: prefer max_depth/glob over a
        bare recursive listing, which is capped by limit.

        Args:
            path: Path relative to content root (defaults to root)
            recursive: If true, list all files recursively
            max_depth: Bound recursion depth (implies recursive)
            glob: Filter full paths with a glob, root-anchored rather than
                relative to *path* (implies recursive)
            limit: Cap on entries; a trailing "… truncated" line marks a cut
            with_metadata: Return {"items": [{path, size, metadata}], "truncated"}
                for markdown files (implies recursive)
        Returns:
            A formatted string listing, or a dict when with_metadata is true
        """
        base = path.strip("/")
        base_depth = len(base.split("/")) if base else 0
        # bool(glob), not `glob is not None`: an explicit glob="" (the likeliest
        # way a caller spells "no glob") would otherwise flip this into a full
        # recursive listing while the `if glob:` below applies no filter at all.
        wants_files = recursive or max_depth is not None or bool(glob) or with_metadata
        if wants_files:
            files = await asyncio.to_thread(filesystem.list_all_files, base)
            if max_depth is not None:
                files = [f for f in files if len(f.split("/")) - base_depth <= max_depth]
            if glob:
                rx = glob_to_regex(normalize_glob(glob))
                files = [f for f in files if rx.match(f)]
            if with_metadata:
                files = [f for f in files if f.lower().endswith((".md", ".markdown"))]
            truncated = len(files) > limit
            files = files[:limit]
            if with_metadata:
                items = []
                for f in files:
                    text = await asyncio.to_thread(filesystem.try_read_text, f)
                    if text is None:
                        continue
                    meta, _ = extract_metadata(text)
                    items.append({
                        "path": f, "size": len(text.encode("utf-8")), "metadata": meta,
                    })
                return {"items": items, "truncated": truncated}
            if not files:
                return f"No files found under '{base or '/'}'"
            out = "\n".join(files)
            if truncated:
                out += f"\n… truncated ({len(files)} shown; raise limit or narrow with path/glob)"
            return out
        items = filesystem.list_files(base)
        lines = []
        for name, is_dir in items[:limit]:
            prefix = "📁 " if is_dir else "📄 "
            lines.append(f"{prefix}{name}")
        if not lines:
            return f"Empty directory: '{base or '/'}'"
        if len(items) > limit:
            lines.append(f"… truncated ({limit} shown)")
        return "\n".join(lines)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Inspect content structure",
            readOnlyHint=True,
            openWorldHint=False,
        )
    )
    async def inspect_content_structure(
        path: Annotated[
            str,
            Field(description="Markdown file path (.md or .markdown) relative to content root"),
        ],
    ) -> dict:
        """Return a markdown file's heading outline (title + nested sections
        with line numbers) without its body text.

        Parses the heading hierarchy (h1-h6) and returns a nested outline of
        the document. Useful for understanding document organization before
        reading full content. A YAML frontmatter block is not scanned for
        headings — a '#' comment inside it is a comment, not an h1 — but
        line numbers stay relative to the whole file.

        Args:
            path: File path relative to content root (must be a .md or .markdown file)
        Returns:
            A dict with 'path', 'title' (first h1 if present), 'sections'
            (nested list of {heading, level, line_number, children} entries),
            and 'metadata' (frontmatter/blockquote key-values)
        """
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in {".md", ".markdown"}:
            raise ValueError(
                f"inspect_content_structure only supports markdown files (.md, .markdown). Got: {path}"
            )
        content = filesystem.read_file(path)
        sections, title, metadata = _document_outline(content)
        return {"path": path, "title": title, "sections": sections, "metadata": metadata}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Inspect content structure (batch)",
            readOnlyHint=True,
            openWorldHint=False,
        )
    )
    async def inspect_content_structure_batch(
        paths: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=10,
                description="Markdown file paths (.md or .markdown) relative to "
                "content root (max 10, no duplicates)",
            ),
        ],
    ) -> dict:
        """Return the heading structure of multiple markdown files.

        Parses up to 10 markdown files and returns their heading hierarchies.
        Useful for scanning a documentation tree to understand content organization
        across multiple files. A YAML frontmatter block is not scanned for
        headings — a '#' comment inside it is a comment, not an h1 — but line
        numbers stay relative to the whole file.

        Args:
            paths: List of markdown file paths relative to content root (max 10)
        Returns:
            A dict with 'results' list, each containing the path, title, sections
            (nested {heading, level, line_number, children} entries), metadata
            (frontmatter/blockquote key-values; null on error), and error
            (null on success; per-file failures set 'error' without failing
            the whole call)
        """
        if len(paths) == 0:
            raise ValueError("At least one path is required.")
        if len(paths) > 10:
            raise ValueError(f"Maximum 10 files per batch. Got {len(paths)}.")
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate paths are not allowed in a single batch call.")

        results = []
        for path in paths:
            try:
                suffix = PurePosixPath(path).suffix.lower()
                if suffix not in (".md", ".markdown"):
                    raise ValueError(
                        f"inspect_content_structure only supports markdown files "
                        f"(.md, .markdown). Got: {path}"
                    )
                content = filesystem.read_file(path)
                sections, title, metadata = _document_outline(content)
                results.append({
                    "path": path,
                    "title": title,
                    "sections": sections,
                    "metadata": metadata,
                    "error": None,
                })
            except (FileNotFoundError, InvalidPathError, ValueError) as exc:
                results.append({
                    "path": path,
                    "title": None,
                    "sections": None,
                    "metadata": None,
                    "error": str(exc),
                })
        return {"results": results}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Find literal matches in content",
            readOnlyHint=True,
            openWorldHint=False,
        )
    )
    async def find_content(
        pattern: str,
        is_regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 50,
        file_types: str | None = None,
        path_prefix: str | None = None,
        context_lines: int = 0,
        exclude_patterns: str | None = None,
    ) -> dict:
        """Find every line matching a literal string or regex.

        Exhaustive enumeration, not ranked retrieval. Use this for
        completeness queries: every reference to a symbol, every file
        containing an env var, every match of an error string.
        Only files with recognized text extensions are scanned; binaries
        and unknown extensions are skipped.

        When this server has search_content registered, prefer it for
        conceptual queries ("how does X work") — it ranks by meaning
        instead of enumerating every literal match.

        Args:
            pattern: Literal substring (default) or regex (when is_regex=True).
            is_regex: Treat pattern as a Python regex, matched line by line —
                a pattern can never span lines.
            case_sensitive: Match case-sensitively. Default false.
            max_results: Hard cap on total matches returned. Default 50.
            file_types: Optional comma-separated file extensions
                (e.g. ".md,.py").
            path_prefix: Single subtree to scan, e.g. "docs/". One prefix
                only — unlike search_content's path_prefix, this is NOT
                comma-separated.
            context_lines: Lines of context to include before and after
                each match. Default 0, max 10.
            exclude_patterns: Optional comma-separated globs to skip (e.g. "**/_reports/**");
                the server's default search-exclusion patterns are NOT
                applied here — find_content scans everything text-like
                unless you exclude explicitly.
        Returns:
            A dict with 'matches' (list of {file_path, line_number, line,
            context_before, context_after}), 'truncated' (bool), and
            'files_scanned' (int).
        """
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        if max_results < 1:
            raise ValueError("max_results must be a positive integer")
        if max_results > Config.FIND_MAX_RESULTS_CEILING:
            raise ValueError(
                f"max_results exceeds ceiling of {Config.FIND_MAX_RESULTS_CEILING}"
            )
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        if context_lines > 10:
            raise ValueError("context_lines capped at 10")

        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex: {e}") from e
        else:
            compiled = re.compile(re.escape(pattern), flags)

        types_list = None
        if file_types:
            types_list = [t.strip() for t in file_types.split(",") if t.strip()]

        exclude_res = []
        if exclude_patterns:
            exclude_res = [
                glob_to_regex(normalize_glob(p))
                for p in exclude_patterns.split(",") if p.strip()
            ]

        try:
            all_files = await asyncio.to_thread(
                filesystem.list_all_files, path_prefix or ""
            )
        except InvalidPathError as exc:
            raise ValueError(str(exc)) from exc

        matches: list[dict] = []
        files_scanned = 0
        truncated = False

        for fp in all_files:
            if not _is_searchable(fp):
                continue
            if exclude_res and any(rx.match(fp) for rx in exclude_res):
                continue
            if types_list and not any(fp.endswith(ext) for ext in types_list):
                continue
            content = await asyncio.to_thread(filesystem.try_read_text, fp)
            if content is None:
                continue
            files_scanned += 1
            lines = content.splitlines()
            for i, line in enumerate(lines, start=1):
                if compiled.search(line):
                    ctx_before = (
                        lines[max(0, i - 1 - context_lines):i - 1]
                        if context_lines else []
                    )
                    ctx_after = (
                        lines[i:i + context_lines]
                        if context_lines else []
                    )
                    matches.append({
                        "file_path": fp,
                        "line_number": i,
                        "line": line,
                        "context_before": ctx_before,
                        "context_after": ctx_after,
                    })
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        return {
            "matches": matches,
            "truncated": truncated,
            "files_scanned": files_scanned,
        }

    if not Config.READ_ONLY:

        @_write_tool(
            annotations=ToolAnnotations(
                title="Move or rename file",
                readOnlyHint=False,
                destructiveHint=False,
                openWorldHint=False,
            )
        )
        async def move_content(
            source_path: Annotated[
                str, Field(description="Current file path relative to content root")
            ],
            dest_path: Annotated[
                str,
                Field(
                    description="New file path relative to content root; "
                    "must not already exist"
                ),
            ],
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> str:
            """Move or rename a content file.

            The destination must not already exist (files are never
            overwritten by a move); missing parent directories are created
            automatically. Use move_content_directory for whole directories.

            Args:
                source_path: Current file path relative to content root
                dest_path: New file path relative to content root
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            Returns:
                Confirmation message
            """
            async with _write_guard(ctx):
                filesystem.move_file(source_path, dest_path)
                commit = await _after_write(
                    ctx, [source_path, dest_path], commit_message, author,
                    f"Move {source_path} -> {dest_path}",
                )
            source_was_resource = _unregister_resource(source_path)
            dest_is_resource = _register_resource(dest_path)
            if source_was_resource or dest_is_resource:
                await ctx.send_resource_list_changed()
            emit(CONTENT_MOVED, dest_path, source_path=source_path)
            logger.info(f"Moved: {source_path} -> {dest_path}")
            return _with_commit(f"Moved: {source_path} -> {dest_path}", commit)

        @_write_tool(
            annotations=ToolAnnotations(
                title="Move content directory",
                readOnlyHint=False,
                destructiveHint=False,
                openWorldHint=False,
            )
        )
        async def move_content_directory(
            source_path: Annotated[
                str, Field(description="Current directory path relative to content root")
            ],
            dest_path: Annotated[
                str,
                Field(
                    description="New directory path relative to content root; "
                    "must not already exist"
                ),
            ],
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> dict:
            """Move or rename an entire directory tree.

            Moves all files and subdirectories from source_path to dest_path.
            The destination must not already exist.

            Args:
                source_path: Current directory path relative to content root
                dest_path: New directory path relative to content root
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            Returns:
                A dict with 'source', 'destination', 'files_moved' count, and
                'commit' (git hash, or null when nothing was committed — e.g.
                inside a transaction)
            """
            async with _write_guard(ctx):
                moved_files = filesystem.move_directory(source_path, dest_path)
                commit = await _after_write(
                    ctx,
                    [p for pair in moved_files for p in pair],
                    commit_message,
                    author,
                    f"Move {source_path} -> {dest_path}",
                )

            # Handle resource registration changes for any README.md files
            resources_changed = False
            for old_path, new_path in moved_files:
                if _unregister_resource(old_path):
                    resources_changed = True
                if _register_resource(new_path):
                    resources_changed = True
                emit(CONTENT_MOVED, new_path, source_path=old_path)

            if resources_changed:
                await ctx.send_resource_list_changed()

            logger.info(f"Moved directory: {source_path} -> {dest_path} ({len(moved_files)} files)")
            return {
                "source": source_path,
                "destination": dest_path,
                "files_moved": len(moved_files),
                "commit": commit,
            }

        @_write_tool(
            annotations=ToolAnnotations(
                title="Move multiple files",
                readOnlyHint=False,
                destructiveHint=False,
                openWorldHint=False,
            )
        )
        async def move_content_batch(
            moves: Annotated[
                list[MoveOperation],
                Field(
                    min_length=1,
                    max_length=10,
                    description="Move operations (max 10); no duplicate sources or "
                    "destinations, and no path may be both",
                ),
            ],
            ctx: Context,
            commit_message: CommitMessage = None,
            author: CommitAuthor = None,
        ) -> dict:
            """Move or rename multiple files in a single operation.

            All validations run before any moves — if any move fails validation
            the entire operation is aborted and no files are moved. Destinations
            must not already exist.

            Args:
                moves: List of move operations (max 10), each with source_path and dest_path
                commit_message: Optional git commit message (git-tracked servers only)
                author: Optional git author "Name <email>" (git-tracked servers only)
            Returns:
                A dict with a 'results' list (source, destination, and status
                per file) and a top-level 'commit' (git hash for the single
                commit covering every move, or null when nothing was
                committed — e.g. inside a transaction)
            """
            if len(moves) == 0:
                raise ValueError("At least one move operation is required.")
            if len(moves) > 10:
                raise ValueError(f"Maximum 10 moves per batch. Got {len(moves)}.")

            sources = [m.source_path for m in moves]
            if len(sources) != len(set(sources)):
                raise ValueError("Duplicate source paths are not allowed in a single batch move.")

            dests = [m.dest_path for m in moves]
            if len(dests) != len(set(dests)):
                raise ValueError(
                    "Duplicate destination paths are not allowed in a single batch move."
                )

            source_set = set(sources)
            dest_set = set(dests)
            overlap = source_set & dest_set
            if overlap:
                raise ValueError(
                    f"Paths cannot appear as both source and destination: {overlap}. "
                    "Use intermediate paths for swap operations."
                )

            async with _write_guard(ctx):
                for m in moves:
                    if not filesystem.file_exists(m.source_path):
                        raise ValueError(f"Source file not found: {m.source_path}")
                    dst = filesystem._resolve_path(m.dest_path)
                    if dst.exists():
                        raise ValueError(f"Destination already exists: {m.dest_path}")

                results = []
                resources_changed = False

                for m in moves:
                    filesystem.move_file(m.source_path, m.dest_path)
                    if _unregister_resource(m.source_path):
                        resources_changed = True
                    if _register_resource(m.dest_path):
                        resources_changed = True
                    emit(CONTENT_MOVED, m.dest_path, source_path=m.source_path)
                    logger.info(f"Moved: {m.source_path} -> {m.dest_path}")
                    results.append({
                        "source": m.source_path,
                        "destination": m.dest_path,
                        "result": "ok",
                    })

                # A one-move batch reads as a plain move, so it gets the same
                # "Move <src> -> <dst>" subject move_content produces.
                default_message = (
                    f"Move {moves[0].source_path} -> {moves[0].dest_path}"
                    if len(moves) == 1
                    else default_commit_message([m.dest_path for m in moves], "Move")
                )
                commit = await _after_write(
                    ctx,
                    [p for m in moves for p in (m.source_path, m.dest_path)],
                    commit_message,
                    author,
                    default_message,
                )

            if resources_changed:
                await ctx.send_resource_list_changed()

            return {"results": results, "commit": commit}

    # --- Search tool (conditional) ---

    if search_engine is not None:

        @mcp.tool(
            annotations=ToolAnnotations(
                title="Search content by meaning",
                readOnlyHint=True,
                openWorldHint=False,
            )
        )
        async def search_content(
            query: str,
            max_results: int = 5,
            file_types: str | None = None,
            path_prefix: str | None = None,
            boost_prefix: str | None = None,
            exclude_patterns: str | None = None,
            include_excluded: bool = False,
            metadata_filters: dict[str, str] | None = None,
        ) -> str:
            """Search for content by meaning using semantic similarity.

            Best for conceptual or open-ended queries where the most relevant
            chunks matter even if the exact wording differs — e.g. "how does
            the transaction system handle rollback", "what's the indexing
            strategy for embeddings". Returns the top-k ranked results.

            For literal substring or regex matching with exhaustive results
            (every file mentioning STASH_FOO, every reference to an old
            symbol, every occurrence of an error string), use find_content
            instead — it returns every match up to max_results (see the
            truncated flag) rather than a ranked top-k.

            Scoping: when you work inside one root of a multi-project store,
            pass boost_prefix=<root> to rank that root first without hiding
            the rest; pass path_prefix=<root> only when you want nothing
            else. The server may be configured to exclude working
            directories (e.g. report/scan output) from results by default;
            pass include_excluded=true to search them too.

            Args:
                query: Natural language search query
                max_results: Maximum number of results (default 5)
                file_types: Optional comma-separated file extensions
                    (e.g. ".md,.py")
                path_prefix: Optional comma-separated subtree(s) to search;
                    results must lie under one of them (e.g. "projects/x/").
                    A subtree whose directory name contains a literal ","
                    cannot be expressed this way — it would be read as two.
                boost_prefix: Optional comma-separated subtree(s) to prefer;
                    results under them rank first, others still appear
                exclude_patterns: Optional comma-separated glob patterns to
                    drop (root-anchored; "**/_reports/**" matches at any
                    depth). Applied even when include_excluded=true.
                include_excluded: Also return files matched by the server's
                    default exclusion patterns (default false)
                metadata_filters: Optional {key: value} equality filters on
                    document metadata (frontmatter keys such as layer,
                    describes, verified); keys are format-insensitive;
                    values must match exactly as strings.
            Returns:
                Search results formatted as a string; each result shows the
                path, score and a snippet, plus — only when the underlying
                value exists — the Section (heading path) the chunk came
                from and Meta/Context/Last changed lines. The Section line
                names the heading the snippet came from, so after
                read_content you can jump straight to that part of the file
                instead of scanning the whole thing. A Meta value cut at 60
                characters ends in "…".
            """
            types_list = None
            if file_types:
                types_list = [
                    t.strip() for t in file_types.split(",") if t.strip()
                ]
            excludes = None
            if exclude_patterns:
                excludes = [
                    p.strip() for p in exclude_patterns.split(",") if p.strip()
                ]
            for label, value in (("path_prefix", path_prefix), ("boost_prefix", boost_prefix)):
                reject_path_traversal(label, value)

            t0 = time.perf_counter()
            results = await search_engine.search(
                query,
                max_results=max_results,
                file_types=types_list,
                path_prefix=path_prefix,
                exclude_patterns=excludes,
                metadata_filters=metadata_filters,
                include_excluded=include_excluded,
                boost_prefixes=boost_prefix,
            )
            get_metrics().record_search_query(
                query=query,
                provider=Config.SEARCH_EMBEDDER_MODEL,
                result_count=len(results),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

            if not results:
                return "No results found."

            lines = []
            for r in results:
                lines.append(f"📄 {r.file_path} (score: {r.score:.2f})")
                if r.heading_path:
                    lines.append(f"   Section: {' > '.join(r.heading_path)}")
                if r.metadata:
                    # Mark the cut: without it an agent can't tell a truncated
                    # `describes:` from a complete one.
                    meta_str = " ".join(
                        f"{k}={v[:60]}…" if len(v) > 60 else f"{k}={v}"
                        for k, v in sorted(r.metadata.items())
                    )
                    lines.append(f"   Meta: {meta_str}")
                if r.context:
                    lines.append(f"   Context: {r.context}")
                if r.last_changed_at:
                    lines.append(f"   Last changed: {r.last_changed_at} by {r.changed_by}")
                if r.commit_message:
                    lines.append(f"   Commit: {r.commit_message}")
                snippet = r.content[:200]
                if len(r.content) > 200:
                    snippet += "..."
                lines.append(f"   {snippet}")
                lines.append("")
            return "\n".join(lines)

    # --- Git tools (registered only when GIT_TRACKING is enabled) ---

    if git_backend is not None:

        @mcp.tool(
            annotations=ToolAnnotations(
                title="Log file history",
                readOnlyHint=True,
                openWorldHint=False,
            )
        )
        async def log_content(
            path: ContentPath,
            max_count: Annotated[
                int, Field(ge=1, description="Maximum number of commits to return")
            ] = 20,
        ) -> str:
            """Return recent git commits touching a file.

            Each line is: short hash, ISO timestamp, author, commit message.
            Uncommitted changes (e.g. writes in an open transaction) do not
            appear until committed.

            Args:
                path: File path relative to content root
                max_count: Maximum number of commits to return (default 20)
            Returns:
                Commit history formatted as a string, or a "No git history
                found" message for files with no commits
            """
            import asyncio

            entries = await asyncio.to_thread(git_backend.log, path, max_count)
            if not entries:
                return f"No git history found for '{path}'."
            lines = []
            for e in entries:
                lines.append(
                    f"{e.commit_hash[:8]}  {e.timestamp.isoformat()}  {e.author}  {e.message}"
                )
            return "\n".join(lines)

        @mcp.tool(
            annotations=ToolAnnotations(
                title="View file changes",
                readOnlyHint=True,
                openWorldHint=False,
            )
        )
        async def diff_content(
            path: ContentPath,
            ref: Annotated[
                str | None,
                Field(
                    description="Git ref to diff against (default HEAD~1). "
                    "Use 'HEAD' to see uncommitted changes"
                ),
            ] = None,
        ) -> str:
            """Show what changed in a file since a given git ref.

            Compares the file's current on-disk content (working tree)
            against *ref*. The default HEAD~1 shows changes since the
            previous commit; pass ref='HEAD' to see only uncommitted
            changes, such as pending edits in an open transaction.

            Args:
                path: File path relative to content root
                ref: Git ref to diff against (default: HEAD~1)
            Returns:
                Unified diff as a string; an empty string means no changes.
                Invalid refs return git's error text instead of a diff.
            """
            import asyncio

            return await asyncio.to_thread(git_backend.diff, path, ref)

        @mcp.tool(
            annotations=ToolAnnotations(
                title="View file blame",
                readOnlyHint=True,
                openWorldHint=False,
            )
        )
        async def blame_content(
            path: ContentPath,
            start_line: Annotated[
                int | None,
                Field(
                    ge=1,
                    description="1-based start line (inclusive); requires end_line "
                    "to also be set, otherwise the whole file is blamed",
                ),
            ] = None,
            end_line: Annotated[
                int | None,
                Field(
                    ge=1,
                    description="1-based end line (inclusive); requires start_line "
                    "to also be set, otherwise the whole file is blamed",
                ),
            ] = None,
        ) -> str:
            """Return line-level authorship and timestamps for a file.

            The line range applies only when BOTH start_line and end_line are
            provided; if either is omitted, the entire file is blamed.

            Args:
                path: File path relative to content root
                start_line: Optional 1-based start line (requires end_line)
                end_line: Optional 1-based end line (requires start_line)
            Returns:
                Blame information formatted as a string
            """
            import asyncio

            blame_lines = await asyncio.to_thread(
                git_backend.blame, path, start_line, end_line
            )
            if not blame_lines:
                return f"No blame data available for '{path}'."
            lines = []
            for bl in blame_lines:
                lines.append(
                    f"{bl.line_number:4d}  {bl.commit_hash[:8]}  "
                    f"{bl.timestamp.isoformat()}  {bl.author}  {bl.content}"
                )
            return "\n".join(lines)

    # --- Transaction tools (only when write mode + a TransactionManager are active) ---

    if not Config.READ_ONLY and tm is not None:

        @mcp.tool(
            annotations=ToolAnnotations(
                title="Start transaction",
                readOnlyHint=False,
                destructiveHint=False,
                openWorldHint=False,
            )
        )
        async def start_content_transaction(ctx: Context) -> str:
            """Begin a write transaction for this session and return its UUID.

            Subsequent writes from this session are grouped into one commit
            made by commit_content_transaction (abort_content_transaction
            reverts exactly those files). Transactions from other sessions
            may be open at the same time and do not block this one. Fails
            if this session already has an open transaction — commit or
            abort it first (list_content_transactions shows it). Idle
            transactions are auto-aborted after a timeout, discarding their
            changes.

            Returns:
                Confirmation string containing the transaction UUID.
            """
            session_id = str(id(ctx.session))
            try:
                txn_id = await tm.start_transaction(
                    session_id,
                    Config.TRANSACTION_TIMEOUT,
                    Config.TRANSACTION_LOCK_WAIT,
                )
            except TransactionError as exc:
                raise ValueError(str(exc))
            return (
                f"Transaction started: {txn_id}\n\n"
                "IMPORTANT: When you are finished making changes, call "
                "`commit_content_transaction` to save them. If you want to "
                "discard all changes, call `abort_content_transaction` instead."
            )

        @mcp.tool(
            annotations=ToolAnnotations(
                title="Commit transaction",
                readOnlyHint=False,
                destructiveHint=False,
                openWorldHint=False,
            )
        )
        async def commit_content_transaction(
            message: Annotated[
                str,
                Field(description="Git commit message describing the changes"),
            ],
            ctx: Context,
            author: Annotated[
                str | None,
                Field(
                    description='Optional commit author as "Name <email>"; '
                    "defaults to the repository's configured identity"
                ),
            ] = None,
        ) -> str:
            """Commit the files written in this session's transaction (only those files).

            Per-write commit_message values passed to individual write calls
            during the transaction are appended to this commit's body as
            bullets. When GIT_SYNC_ENABLED is true the commit is pushed to the
            configured remote by the periodic sync task, not by this call.
            Releases the transaction lock so other sessions may proceed.

            Args:
                message: Commit message describing the changes
                author: Optional commit author in ``"Name <email>"`` format.
                    Defaults to the repository's configured identity.
            Returns:
                Confirmation string, including the commit hash when one was made
            """
            session_id = str(id(ctx.session))
            try:
                commit = await tm.end_transaction(session_id, message, author)
            except TransactionError as exc:
                raise ValueError(str(exc))
            except RuntimeError as exc:
                raise ValueError(
                    f"Transaction closed, but the commit failed: {exc}. The "
                    "touched files were NOT committed — they remain on disk, "
                    "dirty and unstaged, and are no longer part of any "
                    "transaction; inspect them manually."
                ) from exc
            return _with_commit(f"Transaction committed: {message}", commit)

        @mcp.tool(
            annotations=ToolAnnotations(
                title="Abort transaction",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            )
        )
        async def abort_content_transaction(ctx: Context) -> str:
            """Abort this session's transaction, restoring the files it
            touched to their last committed state.

            Returns:
                Confirmation string
            """
            session_id = str(id(ctx.session))
            try:
                await tm.abort_transaction(session_id)
            except TransactionError as exc:
                raise ValueError(str(exc))
            except RuntimeError as exc:
                raise ValueError(
                    f"Transaction closed, but restoring its files failed: {exc}. "
                    "The working tree was NOT reverted for the paths this "
                    "transaction touched — inspect them manually."
                ) from exc
            return "Transaction aborted."

        @mcp.tool(
            annotations=ToolAnnotations(
                title="List transactions",
                readOnlyHint=True,
                openWorldHint=False,
            )
        )
        async def list_content_transactions(ctx: Context) -> dict:
            """List active content transactions.

            Returns the current transaction state including whether a
            transaction is active, its ID, which session owns it, and
            whether this session is the owner.  Useful for agent
            retry/recovery scenarios where the agent needs to know if a
            transaction is still open before attempting to start a new one.

            Returns:
                A dict with 'has_active_transaction' (bool) and, when true,
                'transaction_id', 'session_id', 'owned_by_current_session',
                'count' (number of transactions open across all sessions),
                and 'transactions' (a list with every open transaction's own
                transaction_id, session_id, started_at, touched_paths, and
                owned_by_current_session).

                Top-level transaction_id/session_id describe this session's
                transaction when it has one, otherwise the oldest open one —
                check owned_by_current_session. Another session's open
                transaction does not block you from starting your own.
            """
            session_id = str(id(ctx.session))
            return tm.get_transaction_status(session_id)

    return mcp
