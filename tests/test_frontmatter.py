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

    The untouched neighbor carries an inline comment and the assertion
    checks for it verbatim: a whole-block _redump fallback always strips
    comments, so if the truthfulness guard were silently rescuing a broken
    splice (rather than the splice genuinely succeeding), that comment would
    be gone and these tests would catch it -- unlike checking only that the
    new/removed value is present/absent, which a correct-but-reformatted
    fallback result would also satisfy.
    """

    def test_set_block_collection_valued_key_directly(self):
        content = "---\ntags:\n  - a\n  - b\nowner: me  # keep this comment\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"tags": "flat"})
        block = split_frontmatter_block(new)[0]
        assert "owner: me  # keep this comment" in block
        assert "tags: flat" in block
        assert meta == {"tags": "flat", "owner": "me"}

    def test_unset_block_collection_valued_key_directly(self):
        content = "---\ntags:\n  - a\n  - b\nowner: me  # keep this comment\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["tags"])
        block = split_frontmatter_block(new)[0]
        assert "owner: me  # keep this comment" in block
        assert "tags" not in block
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


class TestMergeFrontmatterTruthfulnessGuardFaultInjection:
    """Prove _splice_block's truthfulness check is itself load-bearing (not
    just something every natural test happens to satisfy either way): force
    _splice_entries to return a candidate that doesn't match the intended
    metadata, and confirm merge_frontmatter discards it for the always-
    truthful whole-block redump instead of writing (or reporting) the wrong
    thing. Covers all four shapes _splice_block is called for: set only,
    unset only, both together, and creating a block from scratch (an empty
    raw_block, where _locate_entries("") returns {} rather than None).
    """

    def test_bogus_splice_discarded_on_set(self, monkeypatch):
        import stash_mcp.frontmatter as fm

        monkeypatch.setattr(fm, "_splice_entries", lambda *a, **k: "not: [valid")
        content = "---\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"verified": "2026-08-16"})
        assert meta == {"owner": "me", "verified": "2026-08-16"}
        reparsed, _ = parse_frontmatter(new)
        assert reparsed == meta

    def test_bogus_splice_discarded_on_unset(self, monkeypatch):
        import stash_mcp.frontmatter as fm

        monkeypatch.setattr(fm, "_splice_entries", lambda *a, **k: "wrong: nonsense")
        content = "---\nowner: me\nlayer: x\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["layer"])
        assert meta == {"owner": "me"}
        reparsed, _ = parse_frontmatter(new)
        assert reparsed == meta

    def test_bogus_splice_discarded_on_set_and_unset(self, monkeypatch):
        import stash_mcp.frontmatter as fm

        monkeypatch.setattr(fm, "_splice_entries", lambda *a, **k: "totally: [broken")
        content = "---\nowner: me\nlayer: x\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"verified": "2026-08-16"}, ["layer"])
        assert meta == {"owner": "me", "verified": "2026-08-16"}
        reparsed, _ = parse_frontmatter(new)
        assert reparsed == meta

    def test_bogus_splice_discarded_on_block_creation(self, monkeypatch):
        import stash_mcp.frontmatter as fm

        monkeypatch.setattr(fm, "_splice_entries", lambda *a, **k: "still: [wrong")
        content = "# T\n\nbody\n"
        new, meta = merge_frontmatter(content, {"verified": "2026-08-16"})
        assert meta == {"verified": "2026-08-16"}
        reparsed, _ = parse_frontmatter(new)
        assert reparsed == meta

    def test_bogus_splice_with_duplicate_keys_discarded(self, monkeypatch):
        # This candidate is *valid, parseable* YAML that even reparses to
        # the right scalar values overall -- yaml.safe_load collapses the
        # duplicate "verified" to its last occurrence, which happens to be
        # the correct one. A truthfulness check that only compares reparsed
        # scalars against `data` would wrongly call this candidate truthful
        # and write the duplicate to disk. It's only caught because
        # _is_truthful also requires the candidate itself to be safely
        # re-editable (_locate_entries(candidate) is not None), which
        # duplicate top-level keys fail.
        import stash_mcp.frontmatter as fm

        monkeypatch.setattr(
            fm,
            "_splice_entries",
            lambda *a, **k: "owner: me\nverified: wrong\nverified: 2026-08-16",
        )
        content = "---\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"verified": "2026-08-16"})
        block = split_frontmatter_block(new)[0]
        assert block.count("verified") == 1
        assert meta == {"owner": "me", "verified": "2026-08-16"}


class TestMergeFrontmatterSetPathNormalizedKeyMatching:
    """The set path must match existing on-disk keys by normalized name,
    exactly like unset already does -- otherwise a caller passing back the
    normalized spelling read_content showed it (e.g. "Verified" against an
    on-disk "verified") appends a *second*, duplicate key instead of
    updating the existing one in place. This is the finding-2 class
    (normalized-key matching) recurring on the set side specifically.
    """

    def test_set_with_different_casing_updates_in_place_no_duplicate(self):
        content = "---\nverified: 2026-01-01\nowner: me  # keep this comment\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"Verified": "2026-08-16"})
        block = split_frontmatter_block(new)[0]
        assert block.count("verified") == 1
        assert "2026-01-01" not in block
        assert "2026-08-16" in block
        # A whole-block redump fallback rescuing a duplicate-key mistake
        # would also end up with one "verified" line and no stale date --
        # but it would also strip this comment, so its presence proves the
        # true surgical in-place update ran, not a fallback masking a bug.
        assert "owner: me  # keep this comment" in block
        assert meta == {"verified": "2026-08-16", "owner": "me"}

    def test_set_with_hyphen_vs_underscore_updates_in_place_no_duplicate(self):
        content = "---\nlast_verified: 2026-01-01\nowner: me  # keep this comment\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"last-verified": "2026-08-16"})
        block = split_frontmatter_block(new)[0]
        assert block.count("verified") == 1
        assert "2026-01-01" not in block
        # Same reasoning as the test above: only the true in-place surgical
        # update keeps this comment -- a fallback rescuing a duplicate-key
        # mistake would also land on one correct "verified" line, but would
        # strip it, so its presence is what actually proves the direct fix
        # ran rather than the (separate) output-duplicate guard masking it.
        assert "owner: me  # keep this comment" in block
        assert meta == {"last_verified": "2026-08-16", "owner": "me"}


class TestMergeFrontmatterTrimGatedToBlockCollections:
    """The blank/comment-aware trim (_trim_swallowed_trailer) must only
    apply to genuine block-style collections (block sequences/mappings).
    A block scalar's own content can contain a line that merely *looks*
    like a trailing comment or blank line -- applying the collection-aware
    trim there mistakes real value content for swallowed noise and orphans
    a fragment of the value in the document after set/unset.
    """

    def test_block_scalar_hash_line_not_orphaned_on_unset(self):
        content = "---\nnotes: |\n  line1\n  # literal hash line\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {}, ["notes"])
        block = split_frontmatter_block(new)[0]
        assert "literal hash line" not in block
        assert "notes" not in block
        assert "owner: me" in block
        assert meta == {"owner": "me"}

    def test_block_scalar_hash_line_not_orphaned_on_set(self):
        content = "---\nnotes: |\n  line1\n  # literal hash line\nowner: me\n---\nbody\n"
        new, meta = merge_frontmatter(content, {"notes": "flat"})
        block = split_frontmatter_block(new)[0]
        assert "literal hash line" not in block
        assert "notes: flat" in block
        assert "owner: me" in block
        assert meta == {"notes": "flat", "owner": "me"}
