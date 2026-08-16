"""Git backend for Stash-MCP: blame, log, diff, and periodic pull operations."""

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BlameLine:
    """Blame information for a single file line."""

    line_number: int
    commit_hash: str
    author: str
    timestamp: datetime
    summary: str
    content: str


@dataclass
class LogEntry:
    """A single git log entry."""

    commit_hash: str
    author: str
    timestamp: datetime
    message: str


@dataclass
class PullResult:
    """Result of a git pull operation."""

    success: bool
    changed_files: list[str] = field(default_factory=list)
    added_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    message: str = ""


def _parse_blame_porcelain(output: str) -> list[BlameLine]:
    """Parse ``git blame --porcelain`` output into BlameLine objects."""
    lines: list[BlameLine] = []
    commits: dict[str, dict] = {}  # commit hash → metadata accumulated so far

    current_hash: str | None = None
    current_meta: dict = {}
    line_number: int = 0

    for raw_line in output.splitlines():
        if raw_line.startswith("\t"):
            # Tab-prefixed lines are the actual file content — this closes the entry
            if current_hash is not None:
                content = raw_line[1:]
                commits[current_hash] = dict(current_meta)
                lines.append(
                    BlameLine(
                        line_number=line_number,
                        commit_hash=current_hash,
                        author=current_meta.get("author", ""),
                        timestamp=current_meta.get(
                            "timestamp", datetime.now(UTC)
                        ),
                        summary=current_meta.get("summary", ""),
                        content=content,
                    )
                )
        else:
            # Try to detect the header line: "<40-hex-hash> <orig> <final> [<n>]"
            parts = raw_line.split()
            if (
                len(parts) >= 3
                and len(parts[0]) == 40
                and all(c in "0123456789abcdef" for c in parts[0])
            ):
                current_hash = parts[0]
                try:
                    line_number = int(parts[2])
                except (IndexError, ValueError):
                    line_number = 0
                # Reuse cached metadata if we've seen this commit before
                if current_hash in commits:
                    current_meta = dict(commits[current_hash])
                else:
                    current_meta = {}
            elif current_hash is not None:
                # Metadata lines for the current commit
                if raw_line.startswith("author ") and not raw_line.startswith("author-"):
                    current_meta["author"] = raw_line[7:]
                elif raw_line.startswith("author-time "):
                    try:
                        ts = int(raw_line[12:])
                        current_meta["timestamp"] = datetime.fromtimestamp(
                            ts, tz=UTC
                        )
                    except ValueError:
                        current_meta["timestamp"] = datetime.now(UTC)
                elif raw_line.startswith("summary "):
                    current_meta["summary"] = raw_line[8:]

    return lines


def _parse_pull_file_statuses(
    git_diff_output: str,
) -> tuple[list[str], list[str], list[str]]:
    """Parse ``git diff --name-status`` output into added/modified/deleted lists."""
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []

    for line in git_diff_output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        status, path = parts[0].strip(), parts[1].strip()
        if status.startswith("A"):
            added.append(path)
        elif status.startswith("D"):
            deleted.append(path)
        elif status.startswith("M") or status.startswith("R") or status.startswith("C"):
            modified.append(path)

    return added, modified, deleted


def _parse_author_string(author: str) -> tuple[str, str]:
    """Parse a ``"Name <email>"`` string into ``(name, email)``.

    Returns the original string as the name and an empty email if the
    format does not match.
    """
    match = re.match(r"^(.*?)\s*<([^>]*)>\s*$", author)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return author.strip(), ""


class GitBackend:
    """Wraps git CLI operations used by Stash-MCP."""

    def __init__(
        self,
        content_dir: Path,
        sync_token: str | None = None,
        author_default: str = "stash-mcp <stash@local>",
    ) -> None:
        self.content_dir = content_dir
        self.author_default = author_default
        if sync_token:
            self._configure_credentials(sync_token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(
        self, args: list[str], check: bool = False, **kwargs
    ) -> subprocess.CompletedProcess:
        """Run a git command in *content_dir*, capturing stdout/stderr."""
        return subprocess.run(
            args,
            cwd=self.content_dir,
            capture_output=True,
            text=True,
            check=check,
            **kwargs,
        )

    def _configure_credentials(self, token: str) -> None:
        """Configure a git credential helper to inject *token* for HTTPS pulls.

        Writes a small shell script into the repository's .git directory so
        the token is not embedded on the git config command line (which would
        be visible in process listings).

        Security note: The token is stored in plaintext in the .git directory.
        Ensure the repository directory has appropriate filesystem permissions.
        """
        result = self._run(["git", "rev-parse", "--git-dir"])
        if result.returncode != 0:
            logger.warning("Cannot configure git credentials: not a git repository")
            return

        git_dir_raw = result.stdout.strip()
        git_dir = (
            Path(git_dir_raw)
            if Path(git_dir_raw).is_absolute()
            else self.content_dir / git_dir_raw
        )

        helper_path = git_dir / "stash-credential-helper.sh"
        # shlex.quote ensures the token value is safely escaped for the shell
        safe_token = shlex.quote(token)
        helper_path.write_text(
            f"#!/bin/sh\necho username=x-access-token\necho password={safe_token}\n"
        )
        helper_path.chmod(0o700)

        subprocess.run(
            ["git", "config", "credential.helper", str(helper_path)],
            cwd=self.content_dir,
            check=True,
            capture_output=True,
        )
        logger.debug("Git credential helper configured at %s", helper_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Verify that *content_dir* is a git repository and ensure a
        committer identity is configured.

        When no local ``user.name`` is set (typical in container environments
        with no global git config), the identity is set from
        :attr:`author_default` so that commits never fail with "Please tell me
        who you are".

        Raises:
            RuntimeError: If the directory is not a git repository.
        """
        result = self._run(["git", "rev-parse", "--git-dir"])
        if result.returncode != 0:
            raise RuntimeError(
                f"Content directory '{self.content_dir}' is not a git repository. "
                "Initialise it with 'git init' or set STASH_GIT_TRACKING=false."
            )

        # Ensure a valid committer identity is present.  Check only the local
        # repo config so we don't override a developer's global identity.
        name_result = self._run(["git", "config", "--local", "user.name"])
        if not name_result.stdout.strip():
            name, email = _parse_author_string(self.author_default)
            self._run(["git", "config", "user.name", name])
            self._run(["git", "config", "user.email", email])
            logger.info(
                "Set git committer identity from STASH_GIT_AUTHOR_DEFAULT: %s <%s>",
                name,
                email,
            )

    def validate_remote(self, remote: str) -> bool:
        """Return True if *remote* is configured in this repository."""
        result = self._run(["git", "remote", "get-url", remote])
        return result.returncode == 0

    def rename_remote(self, old_name: str, new_name: str) -> None:
        """Rename a git remote from *old_name* to *new_name*.

        Args:
            old_name: Current remote name.
            new_name: Desired remote name.

        Raises:
            RuntimeError: If the rename fails.
        """
        result = self._run(["git", "remote", "rename", old_name, new_name])
        if result.returncode != 0:
            raise RuntimeError(
                f"git remote rename {old_name!r} {new_name!r} failed: {result.stderr.strip()}"
            )
        logger.debug("Renamed remote '%s' to '%s'", old_name, new_name)

    def blame(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[BlameLine]:
        """Return per-line blame data for *path*.

        Args:
            path: File path relative to the content directory.
            start_line: Optional 1-based start line (inclusive).
            end_line: Optional 1-based end line (inclusive).

        Returns:
            List of :class:`BlameLine` objects, one per file line.
        """
        args = ["git", "blame", "--porcelain"]
        if start_line is not None and end_line is not None:
            args.extend(["-L", f"{start_line},{end_line}"])
        args.append(path)

        result = self._run(args)
        if result.returncode != 0:
            logger.warning("git blame failed for %s: %s", path, result.stderr.strip())
            return []

        return _parse_blame_porcelain(result.stdout)

    def log(self, path: str | None = None, max_count: int = 20) -> list[LogEntry]:
        """Return recent commits, optionally filtered to *path*.

        Args:
            path: Optional file path to scope the log.
            max_count: Maximum number of commits to return.

        Returns:
            List of :class:`LogEntry` objects.
        """
        args = [
            "git",
            "log",
            f"--max-count={max_count}",
            "--format=%H%x00%an%x00%aI%x00%s",
        ]
        if path:
            args.extend(["--", path])

        result = self._run(args)
        if result.returncode != 0:
            logger.warning("git log failed: %s", result.stderr.strip())
            return []

        entries: list[LogEntry] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("\x00")
            if len(parts) < 4:
                continue
            commit_hash, author, timestamp_str, message = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
            )
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now(UTC)
            entries.append(
                LogEntry(
                    commit_hash=commit_hash,
                    author=author,
                    timestamp=timestamp,
                    message=message,
                )
            )
        return entries

    def diff(self, path: str, ref: str | None = None) -> str:
        """Return the diff for *path* against *ref* (default ``HEAD~1``).

        Args:
            path: File path relative to the content directory.
            ref: Git ref to diff against.  Defaults to ``HEAD~1``.

        Returns:
            Unified diff as a string, or an error message.
        """
        ref = ref or "HEAD~1"
        result = self._run(["git", "diff", ref, "--", path])
        if result.returncode != 0:
            logger.warning("git diff failed for %s: %s", path, result.stderr.strip())
            return result.stderr or "diff unavailable"
        return result.stdout

    def commit(self, message: str, author: str | None = None) -> None:
        """Stage all changes and create a commit with *message*.

        Args:
            message: Commit message.
            author: Optional author string in ``"Name <email>"`` format.
                When provided, the commit is recorded with this author identity
                instead of the repository's configured ``user.name``/``user.email``.

        Raises:
            RuntimeError: If staging or committing fails.
        """
        add_result = self._run(["git", "add", "-A"])
        if add_result.returncode != 0:
            raise RuntimeError(f"git add -A failed: {add_result.stderr.strip()}")

        commit_args = ["git", "commit", "-m", message]
        if author:
            commit_args.extend(["--author", author])
        commit_result = self._run(commit_args)
        if commit_result.returncode != 0:
            raise RuntimeError(f"git commit failed: {commit_result.stderr.strip()}")

        logger.info("Committed: %s", message)

    # ------------------------------------------------------------------
    # Path-scoped operations (autocommit / concurrent transactions)
    # ------------------------------------------------------------------

    def _normalize_paths(self, paths: list[str] | None) -> list[str]:
        """Strip leading/trailing ``/`` from each path and drop empties.

        Shared by every path-taking method in this section so pathspecs
        handed to git are always POSIX-relative to ``content_dir`` with no
        leading slash.
        """
        if not paths:
            return []
        return [p.strip("/") for p in paths if p and p.strip("/")]

    def _stageable(self, paths: list[str]) -> list[str]:
        """Return the subset of *paths* that exist on disk or are tracked.

        ``git add -A -- <path>`` errors on a pathspec that matches nothing
        (neither worktree nor index), e.g. a file created and deleted within
        one transaction. Filtering keeps the commit robust.
        """
        clean = self._normalize_paths(paths)
        if not clean:
            return []
        existing = [p for p in clean if (self.content_dir / p).exists()]
        missing = [p for p in clean if p not in existing]
        tracked: set[str] = set()
        if missing:
            result = self._run(["git", "ls-files", "-z", "--", *missing])
            if result.returncode == 0:
                tracked = {p for p in result.stdout.split("\0") if p}
        ordered = []
        for p in clean:
            if (p in existing or p in tracked) and p not in ordered:
                ordered.append(p)
        return ordered

    def has_staged_changes(self, paths: list[str] | None = None) -> bool:
        """True when the index differs from HEAD (optionally only for *paths*).

        A non-empty *paths* that normalizes to nothing (e.g. ``["/"]``) is a
        no-op and reports ``False`` — it must never silently widen to the
        whole index. Only ``paths=None`` (or an empty list) means "check the
        whole index".
        """
        args = ["git", "diff", "--cached", "--quiet"]
        if paths:
            normalized = self._normalize_paths(paths)
            if not normalized:
                return False
            args += ["--", *normalized]
        result = self._run(args)
        if result.returncode == 0:
            return False
        if result.returncode == 1:
            return True
        raise RuntimeError(f"git diff --cached failed: {result.stderr.strip()}")

    def unstage(self, paths: list[str] | None = None) -> None:
        """``git reset -q [-- paths]`` — clears the index, leaves the worktree alone.

        A non-empty *paths* that normalizes to nothing (e.g. ``["/"]``) is a
        no-op — it must never silently widen to a whole-index reset. Only
        ``paths=None`` (or an empty list) resets the whole index.
        """
        args = ["git", "reset", "-q"]
        if paths:
            normalized = self._normalize_paths(paths)
            if not normalized:
                return
            args += ["--", *normalized]
        result = self._run(args)
        if result.returncode != 0:
            logger.warning("git reset failed: %s", result.stderr.strip())

    def commit_paths(
        self, paths: list[str], message: str, author: str | None = None
    ) -> str | None:
        """Stage exactly *paths* and commit them.

        Returns the short commit hash, or ``None`` when none of the paths
        changed relative to HEAD. On commit failure the paths are unstaged
        again and ``RuntimeError`` is raised.
        """
        stageable = self._stageable(paths)
        if not stageable:
            return None
        add_result = self._run(["git", "add", "-A", "--", *stageable])
        if add_result.returncode != 0:
            raise RuntimeError(f"git add failed: {add_result.stderr.strip()}")
        if not self.has_staged_changes(stageable):
            return None
        commit_args = ["git", "commit", "-q", "-m", message]
        if author:
            commit_args.extend(["--author", author])
        commit_args.extend(["--", *stageable])
        commit_result = self._run(commit_args)
        if commit_result.returncode != 0:
            self.unstage(stageable)
            raise RuntimeError(f"git commit failed: {commit_result.stderr.strip()}")
        head = self._run(["git", "rev-parse", "--short", "HEAD"])
        short = head.stdout.strip() if head.returncode == 0 else ""
        logger.info("Committed %s: %s", short or "?", message.splitlines()[0] if message else "")
        return short or None

    def restore_paths(self, paths: list[str]) -> tuple[list[str], list[str]]:
        """Revert *paths* to HEAD without touching anything else.

        Paths present in HEAD are checked out (restoring deletions too);
        paths unknown to HEAD are removed from the worktree.

        Returns:
            ``(restored, deleted)`` lists of relative paths.
        """
        clean = self._normalize_paths(paths)
        if not clean:
            return [], []
        ls = self._run(["git", "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", *clean])
        in_head = {p for p in ls.stdout.split("\0") if p} if ls.returncode == 0 else set()
        to_checkout = [p for p in clean if p in in_head]
        to_delete = [p for p in clean if p not in in_head]
        restored: list[str] = []
        if to_checkout:
            result = self._run(["git", "checkout", "-q", "HEAD", "--", *to_checkout])
            if result.returncode != 0:
                raise RuntimeError(f"git checkout failed: {result.stderr.strip()}")
            restored = to_checkout
        deleted: list[str] = []
        for p in to_delete:
            full = self.content_dir / p
            if full.is_file():
                full.unlink()
                deleted.append(p)
        return restored, deleted

    def ahead_count(self, remote: str, branch: str) -> int:
        """Number of local commits not on ``remote/branch`` (0 if that ref is unknown)."""
        result = self._run(["git", "rev-list", "--count", f"{remote}/{branch}..HEAD"])
        if result.returncode != 0:
            return 0
        try:
            return int(result.stdout.strip() or "0")
        except ValueError:
            return 0

    def reset_hard(self) -> None:
        """Discard all uncommitted changes with ``git reset --hard HEAD``.

        Raises:
            RuntimeError: If the reset fails.
        """
        result = self._run(["git", "reset", "--hard", "HEAD"])
        if result.returncode != 0:
            raise RuntimeError(f"git reset --hard failed: {result.stderr.strip()}")
        logger.info("Hard reset to HEAD completed.")

    def push(self, remote: str, branch: str) -> None:
        """Push *branch* to *remote*.

        Args:
            remote: Remote name (e.g. ``"origin"``).
            branch: Branch name (e.g. ``"main"``).

        Raises:
            RuntimeError: If the push fails.
        """
        result = self._run(["git", "push", remote, branch])
        if result.returncode != 0:
            raise RuntimeError(f"git push failed: {result.stderr.strip()}")
        logger.info("Pushed %s to %s/%s.", branch, remote, branch)

    @classmethod
    def clone(
        cls,
        url: str,
        target_dir: Path,
        branch: str = "main",
        token: str | None = None,
        recursive: bool = False,
    ) -> "GitBackend":
        """Clone a remote repository into *target_dir*.

        If *token* is provided it is injected into the HTTPS URL for
        authentication.  After the clone succeeds the token is removed from
        the remote URL and re-applied via the credential-helper script so it
        is not stored in plain text inside ``.git/config``.

        Args:
            url: HTTPS URL of the repository to clone.
            target_dir: Directory to clone into (must not exist or be empty).
            branch: Branch to checkout after cloning.
            token: Optional personal-access token for private repos.
            recursive: When True, clone with ``--recurse-submodules``.

        Returns:
            A :class:`GitBackend` instance pointed at the cloned repository.

        Raises:
            RuntimeError: If the clone fails (with an actionable message).
        """
        # Inject token into the URL only for the clone operation
        if token:
            if url.startswith("https://"):
                clone_url = "https://x-access-token:" + token + "@" + url[len("https://"):]
            else:
                clone_url = url
        else:
            clone_url = url

        target_dir.parent.mkdir(parents=True, exist_ok=True)

        clone_args = ["git", "clone", "--branch", branch]
        if recursive:
            clone_args.append("--recurse-submodules")
        clone_args.extend([clone_url, str(target_dir)])

        result = subprocess.run(clone_args, capture_output=True, text=True)

        if result.returncode != 0:
            stderr = result.stderr or ""
            if any(
                phrase in stderr.lower()
                for phrase in [
                    "authentication",
                    "permission denied",
                    "could not read username",
                    "403",
                    "401",
                    "invalid credentials",
                ]
            ):
                raise RuntimeError(
                    f"Git clone authentication failure: {stderr.strip()}. "
                    "Check your STASH_GIT_CLONE_TOKEN."
                )
            raise RuntimeError(f"Git clone failed: {stderr.strip()}")

        instance = cls(target_dir)

        if token:
            # Reset the remote URL to the original (without embedded token) …
            subprocess.run(
                ["git", "remote", "set-url", "origin", url],
                cwd=target_dir,
                check=True,
                capture_output=True,
            )
            # … then configure the credential helper so pulls still work.
            instance._configure_credentials(token)

        logger.info("Cloned %s (branch=%s) into %s", url, branch, target_dir)
        return instance

    def pull(self, remote: str, branch: str, recursive: bool = False) -> PullResult:
        """Pull from *remote*/*branch* and return a :class:`PullResult`.

        Determines which files changed by comparing HEAD before and after the
        pull, so callers can emit precise content-change events.

        Args:
            remote: Remote name (e.g. ``"origin"``).
            branch: Branch name (e.g. ``"main"``).
            recursive: If True, pass ``--recurse-submodules``.

        Returns:
            :class:`PullResult` with success flag, categorised file lists,
            and the raw git output message.
        """
        # Snapshot HEAD before pulling so we can diff afterwards
        head_before_result = self._run(["git", "rev-parse", "HEAD"])
        old_head = (
            head_before_result.stdout.strip()
            if head_before_result.returncode == 0
            else None
        )

        pull_args = ["git", "pull", remote, branch]
        if recursive:
            pull_args.append("--recurse-submodules")

        result = self._run(pull_args)
        if result.returncode != 0:
            stderr = result.stderr or ""
            if any(
                phrase in stderr.lower()
                for phrase in [
                    "authentication",
                    "permission denied",
                    "could not read username",
                    "403",
                    "401",
                    "invalid credentials",
                ]
            ):
                logger.warning(
                    "Git pull authentication failure: %s. "
                    "Set STASH_GIT_SYNC_TOKEN for HTTPS authentication.",
                    stderr.strip(),
                )
            else:
                logger.warning("Git pull failed: %s", stderr.strip())
            return PullResult(success=False, message=stderr.strip())

        # Determine which files changed
        head_after_result = self._run(["git", "rev-parse", "HEAD"])
        new_head = (
            head_after_result.stdout.strip()
            if head_after_result.returncode == 0
            else None
        )

        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []

        if old_head and new_head and old_head != new_head:
            diff_result = self._run(
                ["git", "diff", "--name-status", old_head, new_head]
            )
            if diff_result.returncode == 0:
                added, modified, deleted = _parse_pull_file_statuses(diff_result.stdout)

        all_changed = added + modified + deleted
        return PullResult(
            success=True,
            changed_files=all_changed,
            added_files=added,
            modified_files=modified,
            deleted_files=deleted,
            message=result.stdout.strip(),
        )
