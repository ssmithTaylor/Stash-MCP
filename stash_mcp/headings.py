"""Fence-aware markdown heading scanner with character offsets.

Shared by the search indexer (chunk heading paths), ``parse_markdown_structure``
and the section tools. Pure functions, no I/O.
"""

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass(frozen=True)
class Heading:
    text: str
    level: int
    line: int      # 1-based
    offset: int    # character index of the line start in the scanned text


def scan_headings(text: str) -> list[Heading]:
    """Return ATX headings outside fenced code blocks, in document order."""
    headings: list[Heading] = []
    in_fence = False
    offset = 0
    for line_no, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_RE.match(stripped)
            if match:
                title = match.group(2).rstrip("#").strip()
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
