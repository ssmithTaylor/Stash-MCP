"""Tests for frontmatter / leading-blockquote metadata extraction."""

from stash_mcp.frontmatter import (
    extract_metadata,
    merge_frontmatter,
    normalize_key,
    parse_frontmatter,
    parse_leading_blockquote,
    split_frontmatter_block,
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


class TestMergeFrontmatter:
    def test_creates_block_when_missing(self):
        new, meta = merge_frontmatter("# T\n\nbody\n", {"verified": "2026-08-16"})
        assert new.startswith("---\nverified: '2026-08-16'\n---\n# T\n\nbody\n") or \
               new.startswith("---\nverified: 2026-08-16\n---\n# T\n\nbody\n")
        assert meta == {"verified": "2026-08-16"}

    def test_merges_preserving_order_and_body(self):
        content = "---\nlayer: fm\ntags: [a, b]\nowner: me\n---\n# T\n> quote\nbody\n"
        new, meta = merge_frontmatter(
            content, {"owner": "you", "verified": "2026-01-01"}, ["layer"]
        )
        assert new.endswith("---\n# T\n> quote\nbody\n")
        block = split_frontmatter_block(new)[0]
        assert block.index("tags") < block.index("owner") < block.index("verified")
        assert "layer" not in block
        assert meta == {"owner": "you", "verified": "2026-01-01"}   # tags (list) not a scalar

    def test_removes_block_when_empty(self):
        new, meta = merge_frontmatter("---\nlayer: x\n---\nbody", {}, ["layer"])
        assert new == "body" and meta == {}

    def test_invalid_yaml_raises(self):
        import pytest

        with pytest.raises(ValueError):
            merge_frontmatter("---\nkey: [unclosed\n---\nbody", {"a": "b"})

    def test_bom_preserved(self):
        new, _ = merge_frontmatter("\ufeff# T\n", {"a": "1"})
        assert new.startswith("\ufeff---\n")


class TestMergeFrontmatterSurgicalEdits:
    """merge_frontmatter must touch only the lines of keys it sets/unsets."""

    def test_comments_survive_an_unrelated_set(self):
        content = (
            "---\n"
            "# TODO: re-verify this quarterly\n"
            "verified: 2026-01-01\n"
            "owner: taylor  # primary owner\n"
            "---\n"
            "body\n"
        )
        new, _ = merge_frontmatter(content, {"layer": "core"})
        block = split_frontmatter_block(new)[0]
        assert "# TODO: re-verify this quarterly" in block
        assert "owner: taylor  # primary owner" in block
        assert "verified: 2026-01-01" in block
        assert "layer: core" in block
        # Every original line is untouched; only a new line was appended.
        old_block = split_frontmatter_block(content)[0]
        for line in old_block.splitlines():
            assert line in block

    def test_set_matches_on_disk_key_regardless_of_case_no_duplicate(self):
        content = "---\nLayer: frogpilot\ndescribes: x\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"layer": "moretore"})
        block = split_frontmatter_block(new)[0]
        assert block.count("\n") == 1  # still exactly two lines -> one \n between them
        assert "Layer: moretore" in block  # on-disk spelling kept, value updated in place
        assert "frogpilot" not in block
        assert "layer:" not in block  # no second, normalized-form duplicate
        assert meta == {"layer": "moretore", "describes": "x"}

    def test_unset_matches_normalized_on_disk_key(self):
        content = "---\nLayer: frogpilot\ndescribes: x\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["layer"])
        block = split_frontmatter_block(new)[0]
        assert "layer" not in block.lower()
        assert "describes: x" in block
        assert meta == {"describes": "x"}

    def test_set_then_unset_leaves_no_trace_of_either_casing(self):
        content = "---\nLayer: frogpilot\ndescribes: x\n---\nbody\n"
        updated, _ = merge_frontmatter(content, {"layer": "moretore"})
        final, meta = merge_frontmatter(updated, {}, ["layer"])
        block = split_frontmatter_block(final)[0]
        assert "layer" not in block.lower()
        assert "moretore" not in block
        assert "frogpilot" not in block
        assert meta == {"describes": "x"}

    def test_flow_style_list_unchanged_by_unrelated_set(self):
        content = "---\ntags: [a, b]\nowner: me\n---\nbody\n"
        new, _ = merge_frontmatter(content, {"owner": "you"})
        block = split_frontmatter_block(new)[0]
        assert "tags: [a, b]" in block

    def test_multiline_block_list_survives_unrelated_set(self):
        content = "---\ntags:\n  - a\n  - b\nowner: me\n---\nbody\n"
        new, _ = merge_frontmatter(content, {"owner": "you"})
        block = split_frontmatter_block(new)[0]
        assert "tags:\n  - a\n  - b" in block
        assert "owner: you" in block

    def test_merge_key_document_falls_back_instead_of_lying(self):
        # `shared` only exists via the `<<: *b` anchor merge, not as its own
        # top-level entry -- surgical span-matching can't remove it, so this
        # must fall back rather than report success while leaving it in place.
        content = "---\nbase: &b\n  shared: 1\nlayer: fm\n<<: *b\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["shared"])
        reparsed, _ = extract_metadata(new)
        assert "shared" not in reparsed
        assert meta == reparsed


class TestMergeFrontmatterSwallowedTrailer:
    """A block-collection value's node span can swallow trailing blank lines
    and whole comment lines (never interior ones, never for block scalars) --
    the entry's span must be trimmed back to its own true last content line.
    """

    def test_unset_block_list_leaves_trailing_comment_block_intact(self):
        content = (
            "---\n"
            "tags:\n  - a\n  - b\n"
            "\n"
            "# ------------------\n"
            "# Provenance block, do not edit\n"
            "# ------------------\n"
            "verified: 2026-01-01\n"
            "---\nbody\n"
        )
        new, meta = merge_frontmatter(content, {}, ["tags"])
        block = split_frontmatter_block(new)[0]
        assert "# ------------------" in block
        assert "# Provenance block, do not edit" in block
        assert "verified: 2026-01-01" in block
        assert "tags" not in block
        assert meta == {"verified": "2026-01-01"}

    def test_unset_nested_map_leaves_trailing_comment_intact(self):
        content = "---\nnested:\n  x: 1\n  y: 2\n\n# note\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["nested"])
        block = split_frontmatter_block(new)[0]
        assert "# note" in block
        assert "owner: me" in block
        assert "nested" not in block
        assert meta == {"owner": "me"}

    def test_interior_list_comment_not_mistaken_for_a_trailing_boundary(self):
        # "# about b" sits between two list items -- it comes *before* the
        # final item, so a correct trailing-only walk-back never reaches it
        # while looking for where the value truly ends. It's genuinely part
        # of the (about-to-be-deleted) tags value, unlike the *trailing*
        # "# trailing note" after the blank line, which must survive.
        content = (
            "---\n"
            "tags:\n  - a\n  # about b\n  - b\n"
            "\n# trailing note\n"
            "owner: me\n"
            "---\nbody\n"
        )
        new, meta = merge_frontmatter(content, {}, ["tags"])
        block = split_frontmatter_block(new)[0]
        assert "# trailing note" in block
        assert "owner: me" in block
        assert "tags" not in block
        assert "about b" not in block
        assert meta == {"owner": "me"}

    def test_block_scalar_trailing_comment_was_already_correct(self):
        content = "---\nnotes: |\n  line1\n  line2\n\n# note\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["notes"])
        block = split_frontmatter_block(new)[0]
        assert "# note" in block
        assert "owner: me" in block
        assert "notes" not in block
        assert meta == {"owner": "me"}


class TestMergeFrontmatterUnsafeShapesFallBack:
    """Shapes where per-key span editing can't be done safely at all must
    route to the whole-block fallback rather than silently drop or
    misreport a sibling key.
    """

    def test_flow_mapping_top_level_falls_back_instead_of_losing_sibling(self):
        content = "---\n{a: 1, b: 2}\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["a"])
        reparsed, _ = extract_metadata(new)
        assert meta == reparsed
        assert "b" in reparsed
        assert "a" not in reparsed

    def test_duplicate_top_level_keys_fall_back_instead_of_lying(self):
        content = "---\nlayer: a\nlayer: b\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["layer"])
        reparsed, _ = extract_metadata(new)
        assert meta == reparsed
        assert "layer" not in reparsed
        assert reparsed["owner"] == "me"


class TestMergeFrontmatterBlockCollectionDirectlyTouched:
    """Regression coverage: earlier tests only ever set/unset a scalar key
    *unrelated* to a block-collection value, so a broken/deleted walk-back
    (the exact trim that fixes TestMergeFrontmatterSwallowedTrailer) would
    not have failed any of them. These touch the block-collection key itself.
    """

    def test_set_block_collection_valued_key_directly(self):
        content = "---\ntags:\n  - a\n  - b\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"tags": "flat"})
        block = split_frontmatter_block(new)[0]
        assert "tags: flat" in block
        assert "owner: me" in block
        assert meta == {"tags": "flat", "owner": "me"}

    def test_unset_block_collection_valued_key_directly(self):
        content = "---\ntags:\n  - a\n  - b\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["tags"])
        block = split_frontmatter_block(new)[0]
        assert "tags" not in block
        assert "owner: me" in block
        assert meta == {"owner": "me"}


class TestMergeFrontmatterTruthfulnessInvariant:
    """The structural safety net: whatever the surgical splice produces, the
    returned metadata must always match what re-parsing that exact content
    would show. When it wouldn't, merge_frontmatter must fall back rather
    than return a result that fails this check.
    """

    def test_returned_metadata_always_matches_reparsed_content(self):
        cases = [
            ("---\ntags: [a, b]\nowner: me\n---\nbody\n", {"owner": "you"}, []),
            ("---\n{a: 1, b: 2}\n---\nbody\n", {}, ["a"]),
            ("---\nlayer: a\nlayer: b\nowner: me\n---\nbody\n", {}, ["layer"]),
            (
                "---\nbase: &b\n  shared: 1\nlayer: fm\n<<: *b\nowner: me\n---\nbody\n",
                {},
                ["shared"],
            ),
            (
                "---\ntags:\n  - a\n  - b\n\n# note\nowner: me\n---\nbody\n",
                {},
                ["tags"],
            ),
            ("---\nnotes: |\n  a\n  b\nowner: me\n---\nbody\n", {"owner": "you"}, []),
        ]
        for content, values, unset in cases:
            new, meta = merge_frontmatter(content, values, unset)
            reparsed, _ = parse_frontmatter(new)
            assert meta == reparsed, f"mismatch for {content!r}: {meta} != {reparsed}"
