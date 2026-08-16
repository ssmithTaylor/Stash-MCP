"""Tests for frontmatter / leading-blockquote metadata extraction."""

from stash_mcp.frontmatter import (
    extract_metadata,
    normalize_key,
    parse_frontmatter,
    parse_leading_blockquote,
)


class TestParseFrontmatter:
    def test_no_block_returns_empty_and_untouched_body(self):
        content = "# Title\n\nBody."
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_scalars_are_stringified(self):
        content = (
            "---\n"
            "describes: openpilot@v0.11.1\n"
            "commit: 4df40d2c\n"
            "layer: frogpilot\n"
            "verified: 2026-08-16\n"
            "stale: false\n"
            "count: 3\n"
            "---\n"
            "# Title\n"
        )
        meta, body = parse_frontmatter(content)
        assert meta == {
            "describes": "openpilot@v0.11.1",
            "commit": "4df40d2c",
            "layer": "frogpilot",
            "verified": "2026-08-16",
            "stale": "false",
            "count": "3",
        }
        assert body == "# Title\n"

    def test_non_scalar_values_are_dropped(self):
        content = "---\ntags: [a, b]\nnested:\n  k: v\nlayer: x\n---\nbody"
        meta, body = parse_frontmatter(content)
        assert meta == {"layer": "x"}
        assert body == "body"

    def test_invalid_yaml_leaves_block_in_body(self):
        content = "---\nkey: [unclosed\n---\nbody"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_keys_are_normalized(self):
        meta, _ = parse_frontmatter("---\nDescribes Ref: x\nlast-verified: y\n---\n")
        assert meta == {"describes_ref": "x", "last_verified": "y"}

    def test_bom_and_dots_terminator(self):
        meta, body = parse_frontmatter("﻿---\na: 1\n...\nrest")
        assert meta == {"a": "1"}
        assert body == "rest"

    def test_block_without_terminator_is_not_frontmatter(self):
        content = "---\na: 1\n# heading\n"
        assert parse_frontmatter(content) == ({}, content)


class TestParseLeadingBlockquote:
    def test_existing_store_format(self):
        body = (
            "# pandad\n\n"
            "> Path in repo: `selfdrive/pandad`   Domain: core/hardware   "
            "Investigated: 2026-08-15 (commit v0.11.1 / 4df40d2c)\n\n"
            "## Role\n"
        )
        assert parse_leading_blockquote(body) == {
            "path_in_repo": "selfdrive/pandad",
            "domain": "core/hardware",
            "investigated": "2026-08-15 (commit v0.11.1 / 4df40d2c)",
        }

    def test_proposed_describes_format(self):
        body = (
            "# Doc\n"
            "> Describes: openpilot@v0.11.1 (4df40d2c) · layer: frogpilot · "
            "verified: 2026-08-16\n"
        )
        assert parse_leading_blockquote(body) == {
            "describes": "openpilot@v0.11.1 (4df40d2c)",
            "layer": "frogpilot",
            "verified": "2026-08-16",
        }

    def test_multi_line_blockquote_and_pipe_separator(self):
        body = "# T\n\n> Layer: a | Verified: 2026-01-01\n> Owner: me\n\nText"
        assert parse_leading_blockquote(body) == {
            "layer": "a", "verified": "2026-01-01", "owner": "me",
        }

    def test_blockquote_after_prose_is_ignored(self):
        body = "# T\n\nSome prose first.\n\n> Layer: a\n"
        assert parse_leading_blockquote(body) == {}

    def test_fields_without_colon_ignored_and_first_colon_splits(self):
        body = "> just words · Repo: https://github.com/x/y\n"
        assert parse_leading_blockquote(body) == {"repo": "https://github.com/x/y"}


class TestExtractMetadata:
    def test_frontmatter_wins_over_blockquote(self):
        content = "---\nlayer: fm\n---\n# T\n> layer: bq · verified: v\n"
        meta, body = extract_metadata(content)
        assert meta == {"layer": "fm", "verified": "v"}
        assert body == "# T\n> layer: bq · verified: v\n"

    def test_normalize_key(self):
        assert normalize_key("  Describes Ref ") == "describes_ref"
        assert normalize_key("last-verified") == "last_verified"
