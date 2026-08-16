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


def merge_frontmatter(
    content: str, set_values: dict[str, str], unset_keys: list[str] | tuple[str, ...] = (),
) -> tuple[str, dict[str, str]]:
    """Merge *set_values* / remove *unset_keys* in the frontmatter.

    Returns (new_content, metadata).
    """
    bom = "\ufeff" if content.startswith("\ufeff") else ""
    text = content[len(bom):]
    data: dict = {}
    body = text
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
    for key in unset_keys:
        data.pop(key, None)
    for key, value in set_values.items():
        data[key] = value
    if not data:
        return bom + body, {}
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"{bom}---\n{dumped}---\n{body}", _scalars(data)


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
