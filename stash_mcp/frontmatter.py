"""Document metadata extraction: YAML frontmatter and the leading blockquote.

Pure functions, no I/O. Used by the search indexer (chunk metadata), the MCP
resource description, and the UI (metadata card).
"""

import logging
import re
from datetime import date, datetime

import yaml

logger = logging.getLogger(__name__)

_FM_TERMINATOR_RE = re.compile(r"^(?:---|\.\.\.)\s*$")
_FIELD_SPLIT_RE = re.compile(r"\s+[·|]\s+|;\s*|\s{2,}")
_SCALAR_TYPES = (str, int, float, bool, date, datetime)


def normalize_key(key: str) -> str:
    """Lower-case a metadata key and turn spaces/hyphens into underscores."""
    return re.sub(r"[\s\-]+", "_", key.strip().lower())


def _stringify(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def split_frontmatter_block(text: str) -> tuple[str, str] | None:
    """Return (raw_yaml_block, body) when *text* starts with a well-formed block, else None."""
    if not text.startswith("---"):
        return None
    first_nl = text.find("\n")
    if first_nl == -1 or text[:first_nl].strip() != "---":
        return None
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if _FM_TERMINATOR_RE.match(lines[i]):
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    return None


def _scalars(data: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, _SCALAR_TYPES) and value is not None:
            out[normalize_key(str(key))] = _stringify(value)
    return out


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Return ``(metadata, body)`` for a document with optional YAML frontmatter.

    The block must start on the first line (a UTF-8 BOM is tolerated) with
    ``---`` and end at the next line that is exactly ``---`` or ``...``.
    Only top-level scalar values are kept, stringified. Invalid YAML or a
    missing terminator means "no frontmatter": ``({}, content)``.
    """
    text = content[1:] if content.startswith("﻿") else content
    split = split_frontmatter_block(text)
    if split is None:
        return {}, content
    raw_block, body = split
    try:
        data = yaml.safe_load(raw_block)
    except yaml.YAMLError as exc:
        logger.debug("Ignoring invalid frontmatter: %s", exc)
        return {}, content
    if not isinstance(data, dict):
        return {}, content
    return _scalars(data), body


def _trim_trailing_newlines(raw_block: str, start: int, end: int) -> int:
    """Strip trailing newline character(s) that a value's end_mark swallowed
    to disambiguate itself from a following sibling key. Correct for any
    value shape that never swallows more than that: scalars (plain, quoted,
    or block `|`/`>`) and flow collections (`[...]`, `{...}`) all stop right
    after their own last real character even when followed by another key,
    *except* for the separating newline(s) themselves, which this strips.
    """
    while end > start and raw_block[end - 1] == "\n":
        end -= 1
    return end


def _trim_swallowed_trailer(raw_block: str, start: int, end: int) -> int:
    """Shrink `end` back past trailing separator newlines, blank lines, and
    whole comment lines that a block-style *collection's* (block sequence or
    block mapping) node span swallows while PyYAML scans ahead for the next
    token -- to check whether the collection continues, it must look past
    intervening blanks/comments, and it does this even when nothing follows
    at all, e.g. a trailing comment with no further key.

    Only ever called for block collections (see _locate_entries): a block
    scalar's own content can contain a line that merely *looks* blank or
    comment-shaped (a literal line reading `# not a comment` inside a `|`
    block, say), which this function would wrongly treat as trailing noise
    and strip -- orphaning a fragment of the value in the document once it's
    set/unset. _trim_trailing_newlines is the correct, safe choice for every
    other value shape, none of which swallow more than a bare newline.

    Walking only ever moves toward `start` and only ever removes *trailing*
    blank/comment lines, so an interior comment inside a block collection
    (e.g. between two list items, which is real content of the value that
    comes *before* its true last line) is never at risk: the walk stops the
    moment it finds a non-blank, non-comment line, which for any value with
    trailing noise is always its own genuine last content line, encountered
    before the walk could ever reach further back to an interior comment.
    """
    while end > start:
        probe = end - 1 if raw_block[end - 1] == "\n" else end
        line_start = raw_block.rfind("\n", start, probe) + 1
        if line_start < start:
            line_start = start
        line = raw_block[line_start:probe].strip()
        if line == "" or line.startswith("#"):
            end = line_start
            continue
        end = probe
        break
    return end


def _locate_entries(raw_block: str) -> dict[str, tuple[int, int, str]] | None:
    """Map each top-level key's normalized name to its (start, end, raw_key) span.

    ``raw_block[start:end]`` is exactly the on-disk "key: value" text for that
    entry -- key through the last character of its value, however many lines
    the value spans -- with none of the surrounding blank lines / comments /
    other entries. ``end`` is trimmed so it never extends into text that
    doesn't genuinely belong to the value: by _trim_swallowed_trailer for a
    block-style sequence/mapping value, or by the simpler, always-safe
    _trim_trailing_newlines for every other value shape (see each function's
    docstring for why they aren't interchangeable).

    Returns None when the block's structure can't be safely edited key-by-key
    so the caller should fall back to a full reparse/redump instead:

    - Unparseable text, or a non-scalar top-level key (a sequence/mapping
      key is unhashable in Python, so this is already rejected earlier by
      the calling merge_frontmatter's `yaml.safe_load` validation -- kept
      here anyway as defense-in-depth).
    - A YAML merge key (``<<: *anchor``): it can inject top-level keys that
      have no literal span of their own to locate, so surgically touching
      *those* keys would silently leave the anchor's copy in place while
      claiming success.
    - The top-level mapping is flow-styled (``{a: 1, b: 2}``), or any two
      top-level key spans share a physical line: line-based edits assume
      one key per line, so unsetting one flow-mapping key would delete its
      siblings too.
    - The same normalized key names more than one top-level entry (e.g.
      ``layer: a`` and ``layer: b`` both present): only one span could be
      touched, silently leaving the other's stale value behind.

    Every one of these is "refuse to guess, let the caller fall back to the
    always-correct-but-reformatting whole-block dump" rather than risk
    silently writing something wrong.
    """
    if not raw_block.strip():
        return {}
    try:
        root = yaml.compose(raw_block, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return None
    if root is None:
        return {}
    if not isinstance(root, yaml.MappingNode):
        return None
    if root.flow_style:
        return None
    spans: list[tuple[str, int, int, str]] = []
    prev_end: int | None = None
    for key_node, value_node in root.value:
        if not isinstance(key_node, yaml.ScalarNode):
            return None
        if key_node.tag == "tag:yaml.org,2002:merge":
            return None
        start = key_node.start_mark.index
        if prev_end is not None and "\n" not in raw_block[prev_end:start]:
            return None
        raw_end = value_node.end_mark.index
        is_block_collection = (
            isinstance(value_node, (yaml.SequenceNode, yaml.MappingNode))
            and not value_node.flow_style
        )
        end = (
            _trim_swallowed_trailer(raw_block, start, raw_end)
            if is_block_collection
            else _trim_trailing_newlines(raw_block, start, raw_end)
        )
        # An alias-valued key (`b: *a`) resolves to the *anchor's* node, whose
        # end_mark sits back at the anchor's definition -- before this key even
        # begins. `raw_block[start:end]` would then be empty and a splice would
        # duplicate text rather than replace it, so refuse the block outright
        # and let the caller fall back to the whole-block redump.
        if end < start:
            return None
        spans.append((normalize_key(key_node.value), start, end, key_node.value))
        prev_end = end
    norms = [norm for norm, *_ in spans]
    if len(norms) != len(set(norms)):
        return None
    return {norm: (start, end, raw_key) for norm, start, end, raw_key in spans}


def _dump_pair(key: str, value: str) -> str:
    """Serialize a single key/value pair as one line, e.g. "key: 'value'"."""
    return yaml.safe_dump({key: value}, default_flow_style=False, allow_unicode=True).strip()


def _redump(data: dict) -> str:
    """Whole-block reparse/redump of *data*: always correct, but loses
    comments and re-renders flow collections in block style. The
    safety-net fallback used whenever the surgical path can't be trusted.
    """
    return yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()


def _splice_entries(
    raw_block: str,
    entries: dict[str, tuple[int, int, str]],
    set_values: dict[str, str],
    unset_keys: list[str] | tuple[str, ...],
) -> str:
    """Apply set/unset to *raw_block* using *entries*' already-located spans."""
    # Keyed by normalized name, same as `entries` and `unset_norms`: an
    # existing on-disk `Layer` must match a requested `layer`/`Layer`/
    # `LAYER` alike, exactly as the dict-based bookkeeping in
    # merge_frontmatter already does. Comparing normalized keys against a
    # dict keyed by the caller's raw spelling (the bug this replaces) never
    # matches an existing entry, so every set silently became a brand-new
    # appended line -- a permanent duplicate next to the untouched original.
    pending_set = {normalize_key(k): v for k, v in set_values.items()}
    unset_norms = {normalize_key(k) for k in unset_keys}
    edits: list[tuple[int, int, str | None]] = []
    for norm, (start, end, raw_key) in entries.items():
        if norm in unset_norms:
            line_start = raw_block.rfind("\n", 0, start) + 1
            nl = raw_block.find("\n", end)
            line_end = nl + 1 if nl != -1 else len(raw_block)
            edits.append((line_start, line_end, None))
        elif norm in pending_set:
            new_pair = _dump_pair(raw_key, pending_set.pop(norm))
            edits.append((start, end, new_pair))

    result = raw_block
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        result = result[:start] + (replacement or "") + result[end:]
    result = result.rstrip("\n")

    if pending_set:
        appended = "\n".join(_dump_pair(k, v) for k, v in pending_set.items())
        result = f"{result}\n{appended}" if result else appended
    return result


def _is_truthful(candidate: str, data: dict) -> bool:
    """Does re-parsing *candidate* give back exactly the metadata *data* implies?

    The structural safety net beneath every specific _locate_entries guard:
    whatever YAML shape defeats (or wasn't anticipated by) those checks,
    this still catches it -- a surgical result that doesn't reparse back to
    the metadata it would be returned alongside is never used.

    Three checks, not one. `_scalars(reparsed) == _scalars(data)` alone isn't
    enough on either axis:

    - It only compares *scalar* values -- `_scalars()` drops list/dict values
      from both sides -- so a non-scalar key that a mis-computed span dropped
      or duplicated would slip past unnoticed. Comparing the key *sets* too
      closes that. They are compared normalized because a brand-new key is
      appended in its normalized spelling while `data` still carries the
      caller's raw one (`Layer` -> `layer`), and that difference is by
      design, not corruption.
    - yaml.safe_load collapses duplicate top-level keys to the last one, so a
      candidate that (by some bug) wrote a key twice can still reparse to
      exactly the right values while permanently duplicating a line on disk.
      Requiring `_locate_entries(candidate) is not None` closes that hole by
      demanding the candidate itself be safely re-editable -- which duplicate
      (or flow-mapping-sibling, or merge-key-shadowed) top-level keys are not
      -- the same standard already applied to the *input* raw_block, now also
      applied to what we're about to write.
    """
    try:
        reparsed = yaml.safe_load(candidate) if candidate.strip() else {}
    except yaml.YAMLError:
        return False
    if reparsed is None:
        reparsed = {}
    if not isinstance(reparsed, dict):
        return False
    if _scalars(reparsed) != _scalars(data):
        return False
    if {normalize_key(str(k)) for k in reparsed} != {normalize_key(str(k)) for k in data}:
        return False
    return _locate_entries(candidate) is not None


def _splice_block(
    raw_block: str,
    set_values: dict[str, str],
    unset_keys: list[str] | tuple[str, ...],
    data: dict,
) -> str:
    """Return *raw_block* with only the touched keys' own lines changed.

    Untouched keys, comments, blank lines, and value formatting (flow lists,
    block scalars, ...) pass through byte-for-byte -- only the exact spans of
    keys named in *set_values* / *unset_keys* (matched by normalized name,
    case- and separator-insensitively) are replaced or removed. New keys are
    appended, written in their normalized form.

    Falls back to _redump(data) (loses comments, re-renders flow collections
    in block style, but is always correct) whenever the surgical path can't
    be trusted: _locate_entries refuses the block's shape outright, or the
    surgical result fails the _is_truthful check.
    """
    entries = _locate_entries(raw_block)
    if entries is not None:
        candidate = _splice_entries(raw_block, entries, set_values, unset_keys)
        if _is_truthful(candidate, data):
            return candidate
    return _redump(data)


def merge_frontmatter(
    content: str, set_values: dict[str, str], unset_keys: list[str] | tuple[str, ...] = (),
) -> tuple[str, dict[str, str]]:
    """Merge *set_values* / remove *unset_keys* in the frontmatter.

    Returns (new_content, metadata). set/unset match keys by normalized name
    (case- and separator-insensitively), same as the metadata this module
    hands back to callers, so a caller acting on what read_content showed it
    always lands on the right key. Edits are surgical -- see _splice_block.
    """
    bom = "\ufeff" if content.startswith("\ufeff") else ""
    text = content[len(bom):]
    data: dict = {}
    body = text
    raw_block = ""
    split = split_frontmatter_block(text)
    if split is not None:
        raw_block, body = split
        try:
            loaded = yaml.safe_load(raw_block) if raw_block.strip() else {}
        except yaml.YAMLError as exc:
            raise ValueError(f"frontmatter is not valid YAML: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("frontmatter is not a mapping")
        data = dict(loaded)

    # Normalized-key-aware pop/set: an on-disk `Layer` matches a requested
    # `layer`, is updated/removed in place, and keeps its on-disk spelling.
    normalized_existing = {normalize_key(str(k)): k for k in data}
    for key in unset_keys:
        raw_key = normalized_existing.pop(normalize_key(key), None)
        if raw_key is not None:
            data.pop(raw_key, None)
    for key, value in set_values.items():
        norm = normalize_key(key)
        raw_key = normalized_existing.get(norm)
        if raw_key is not None:
            data[raw_key] = value
        else:
            data[key] = value
            normalized_existing[norm] = key

    if not data:
        return bom + body, {}

    new_raw_block = _splice_block(raw_block, set_values, unset_keys, data)
    return f"{bom}---\n{new_raw_block}\n---\n{body}", _scalars(data)


def parse_leading_blockquote(body: str) -> dict[str, str]:
    """Extract ``Key: value`` fields from the blockquote directly under the H1.

    Only a blockquote that appears before any prose (H1 and blank lines are
    allowed before it) is considered. Fields are separated by `` · ``,
    `` | ``, ``;`` or two-plus spaces; each ``Key: value`` pair (split on
    the first colon) becomes ``normalize_key(Key) -> value`` with
    surrounding backticks removed. Fields without a colon are ignored.
    """
    quote_lines: list[str] = []
    for raw in body.split("\n"):
        line = raw.strip()
        if not quote_lines:
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                quote_lines.append(line.lstrip(">").strip())
                continue
            return {}
        if line.startswith(">"):
            quote_lines.append(line.lstrip(">").strip())
        else:
            break
    metadata: dict[str, str] = {}
    for qline in quote_lines:
        for field in _FIELD_SPLIT_RE.split(qline):
            field = field.strip()
            if ":" not in field:
                continue
            key, _, value = field.partition(":")
            key = normalize_key(key)
            value = value.strip().strip("`").strip()
            if key and value:
                metadata[key] = value
    return metadata


def extract_metadata(content: str) -> tuple[dict[str, str], str]:
    """Frontmatter ∪ leading blockquote (frontmatter wins); body keeps the blockquote."""
    fm, body = parse_frontmatter(content)
    merged = parse_leading_blockquote(body)
    merged.update(fm)
    return merged, body
