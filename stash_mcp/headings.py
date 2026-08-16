"""Fence-aware markdown heading scanner with character offsets.

Shared by the search indexer (chunk heading paths), ``parse_markdown_structure``
and the section tools. Pure functions, no I/O.
"""

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
# A CommonMark ATX closing sequence is a run of "#" preceded by whitespace;
# "C#" is not one, so an unconditional rstrip("#") would corrupt it to "C".
_CLOSING_HASHES_RE = re.compile(r"(?<=\s)#+\s*$")


@dataclass(frozen=True)
class Heading:
    text: str
    level: int
    line: int      # 1-based
    offset: int    # character index of the line start in the scanned text


def scan_headings(text: str) -> list[Heading]:
    """Return ATX headings outside fenced code blocks, in document order.

    A fence opened with ``` is only closed by another ``` line (likewise for
    ~~~) — a line using the other delimiter while a fence is open is just
    content, not a close, matching CommonMark.
    """
    headings: list[Heading] = []
    fence_delim: str | None = None
    offset = 0
    for line_no, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            delim = fence_match.group(1)
            if fence_delim is None:
                fence_delim = delim
            elif delim == fence_delim:
                fence_delim = None
        elif fence_delim is None:
            match = _HEADING_RE.match(stripped)
            if match:
                title = _CLOSING_HASHES_RE.sub("", match.group(2)).strip()
                headings.append(
                    Heading(text=title, level=len(match.group(1)), line=line_no, offset=offset)
                )
        offset += len(raw) + 1  # +1 for the "\n" consumed by split
    return headings


def heading_path_at(headings: list[Heading], offset: int) -> list[str]:
    """Texts of the heading stack in force at *offset* (headings at or before it)."""
    stack: list[Heading] = []
    for h in headings:
        if h.offset > offset:
            break
        while stack and stack[-1].level >= h.level:
            stack.pop()
        stack.append(h)
    return [h.text for h in stack]
