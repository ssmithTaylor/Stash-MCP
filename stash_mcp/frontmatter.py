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


def _locate_entries(raw_block: str) -> dict[str, tuple[int, int, str]] | None:
    """Map each top-level key's normalized name to its (start, end, raw_key) span.

    ``raw_block[start:end]`` is exactly the on-disk "key: value" text for that
    entry -- key through the last character of its value, however many lines
    the value spans -- with none of the surrounding blank lines / comments /
    other entries. ``end`` is normalized to exclude any trailing newline(s)
    that a block-style value's node span swallows to disambiguate itself from
    a following sibling key (PyYAML only does this for block collections and
    block scalars, never for plain scalars or flow collections; without the
    normalization, `end` would sometimes land mid-way into the *next* entry).

    Returns None when the block's structure can't be safely edited key-by-key
    so the caller should fall back to a full reparse/redump instead:
    a non-scalar top-level key, unparseable text, or a YAML merge key
    (``<<: *anchor``) -- a merge key can inject top-level keys (e.g. `shared`
    above) that have no literal span of their own to locate, so surgically
    touching *those* keys would silently leave the anchor's copy in place
    while claiming success; the whole-block fallback at least keeps the
    returned metadata truthful about what's actually on disk.
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
    entries: dict[str, tuple[int, int, str]] = {}
    for key_node, value_node in root.value:
        if not isinstance(key_node, yaml.ScalarNode):
            return None
        if key_node.tag == "tag:yaml.org,2002:merge":
            return None
        start = key_node.start_mark.index
        end = value_node.end_mark.index
        while end > start and raw_block[end - 1] == "\n":
            end -= 1
        entries[normalize_key(key_node.value)] = (start, end, key_node.value)
    return entries


def _dump_pair(key: str, value: str) -> str:
    """Serialize a single key/value pair as one line, e.g. "key: 'value'"."""
    return yaml.safe_dump({key: value}, default_flow_style=False, allow_unicode=True).strip()


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
    appended, written in their normalized form. Falls back to a full
    reparse-and-redump of *data* (loses comments, re-renders flow collections
    in block style) when the block's key shapes defeat span-based location.
    """
    entries = _locate_entries(raw_block)
    if entries is None:
        return yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=False
        ).strip()

    pending_set = dict(set_values)
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
        appended = "\n".join(_dump_pair(normalize_key(k), v) for k, v in pending_set.items())
        result = f"{result}\n{appended}" if result else appended
    return result


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
