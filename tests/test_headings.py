"""Tests for the fence-aware heading scanner shared by the indexer and structure tools."""

from stash_mcp.headings import Heading, heading_path_at, scan_headings

DOC = (
    "intro line\n"
    "# Title\n"
    "text\n"
    "## Config\n"
    "```\n"
    "# not a heading\n"
    "```\n"
    "### Env vars\n"
    "body\n"
    "## Running\n"
)


def test_scan_headings_levels_lines_offsets_and_fences():
    hs = scan_headings(DOC)
    assert [(h.text, h.level, h.line) for h in hs] == [
        ("Title", 1, 2), ("Config", 2, 4), ("Env vars", 3, 8), ("Running", 2, 10),
    ]
    assert DOC[hs[0].offset:].startswith("# Title")
    assert DOC[hs[2].offset:].startswith("### Env vars")


def test_heading_path_at_walks_the_stack():
    hs = scan_headings(DOC)
    assert heading_path_at(hs, 0) == []                                # before any heading
    assert heading_path_at(hs, DOC.index("text")) == ["Title"]
    assert heading_path_at(hs, DOC.index("body")) == ["Title", "Config", "Env vars"]
    assert heading_path_at(hs, DOC.index("## Running")) == ["Title", "Running"]


def test_closing_hashes_and_whitespace_are_stripped():
    hs = scan_headings("##   Spaced Heading ##  \n")
    assert hs == [Heading(text="Spaced Heading", level=2, line=1, offset=0)]
