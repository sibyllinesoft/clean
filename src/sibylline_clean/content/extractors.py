"""String extraction from structured documents.

Each extractor pulls all human-readable strings (keys, values, cells, text
nodes, attributes) from a structured format and records byte offsets so
detection results can be mapped back to the original document.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from typing import Protocol, runtime_checkable

from .spans import ExtractedString

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[StringExtractor]] = {}


def register_extractor(cls: type[StringExtractor]) -> type[StringExtractor]:
    """Register an extractor class for its declared content types."""
    for ct in cls.content_types():
        _REGISTRY[ct] = cls
    return cls


def get_extractor(content_type: str) -> type[StringExtractor] | None:
    """Look up an extractor by content type (case-insensitive, ignores params)."""
    # Strip parameters like charset
    base = content_type.split(";")[0].strip().lower()
    return _REGISTRY.get(base)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StringExtractor(Protocol):
    """Extracts strings from a structured document."""

    def extract(self, raw: bytes) -> list[ExtractedString]: ...

    @staticmethod
    def content_types() -> list[str]: ...


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


@register_extractor
class JsonExtractor:
    """Extract all string keys and values from JSON with byte offsets."""

    @staticmethod
    def content_types() -> list[str]:
        return ["application/json", "text/json"]

    def extract(self, raw: bytes) -> list[ExtractedString]:
        text = raw.decode("utf-8")
        parsed = json.loads(text)
        entries: list[ExtractedString] = []
        self._walk(parsed, "", text, raw, 0, entries)
        return entries

    def _walk(
        self,
        node: object,
        path: str,
        text: str,
        raw: bytes,
        search_from: int,
        out: list[ExtractedString],
    ) -> int:
        """Recursively walk a parsed JSON value, recording string positions.

        Returns an updated ``search_from`` cursor so that duplicate string
        values in different positions are located correctly.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = f"{path}/{key}"
                # Locate the key string in the raw bytes
                search_from = self._add_string(
                    key, child_path + "~key", text, raw, search_from, out
                )
                search_from = self._walk(value, child_path, text, raw, search_from, out)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                search_from = self._walk(value, f"{path}/{i}", text, raw, search_from, out)
        elif isinstance(node, str):
            search_from = self._add_string(node, path, text, raw, search_from, out)
        return search_from

    @staticmethod
    def _add_string(
        value: str,
        path: str,
        text: str,
        raw: bytes,
        search_from: int,
        out: list[ExtractedString],
    ) -> int:
        """Find *value* as a JSON-encoded string literal in *raw* and record it."""
        # json.dumps produces the exact in-file representation including escapes
        needle = json.dumps(value, ensure_ascii=False)
        idx = text.find(needle, search_from)
        if idx == -1:
            # Retry with ensure_ascii=True for files that escape non-ASCII
            needle = json.dumps(value, ensure_ascii=True)
            idx = text.find(needle, search_from)
        if idx == -1:
            return search_from

        # Byte offset of the string *content* (skip opening quote)
        content_start = len(text[: idx + 1].encode("utf-8"))
        content_byte_len = len(text[idx + 1 : idx + len(needle) - 1].encode("utf-8"))

        out.append(
            ExtractedString(
                text=value,
                byte_offset=content_start,
                byte_length=content_byte_len,
                path=path,
            )
        )
        return idx + len(needle)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


@register_extractor
class CsvExtractor:
    """Extract all cell values from CSV with byte offsets."""

    @staticmethod
    def content_types() -> list[str]:
        return ["text/csv"]

    def extract(self, raw: bytes) -> list[ExtractedString]:
        text = raw.decode("utf-8")
        entries: list[ExtractedString] = []

        # Track byte offset at start of each line
        line_byte_offsets: list[int] = []
        offset = 0
        for line in text.split("\n"):
            line_byte_offsets.append(offset)
            offset += len(line.encode("utf-8")) + 1  # +1 for \n

        reader = csv.reader(io.StringIO(text))
        for row_idx, row in enumerate(reader):
            if row_idx >= len(line_byte_offsets):
                break
            line_start = line_byte_offsets[row_idx]
            # Get the raw line text to locate cells within it
            raw_line = text.split("\n")[row_idx] if row_idx < len(text.split("\n")) else ""
            cell_search_from = 0

            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                path = f"{row_idx}:{col_idx}"

                # Find cell content in the raw line
                # Handle both quoted and unquoted cells
                cell_pos = raw_line.find(cell, cell_search_from)
                if cell_pos == -1:
                    continue

                byte_offset_in_line = len(raw_line[:cell_pos].encode("utf-8"))
                cell_byte_len = len(cell.encode("utf-8"))

                entries.append(
                    ExtractedString(
                        text=cell,
                        byte_offset=line_start + byte_offset_in_line,
                        byte_length=cell_byte_len,
                        path=path,
                    )
                )
                cell_search_from = cell_pos + len(cell)

        return entries


# ---------------------------------------------------------------------------
# XML / HTML
# ---------------------------------------------------------------------------


@register_extractor
class XmlHtmlExtractor:
    """Extract text nodes and attribute values from XML/HTML."""

    @staticmethod
    def content_types() -> list[str]:
        return ["text/xml", "application/xml", "text/html"]

    def extract(self, raw: bytes) -> list[ExtractedString]:
        text = raw.decode("utf-8")
        entries: list[ExtractedString] = []
        # Use a cursor to track search position
        search_from = 0

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return entries

        self._walk(root, "", text, raw, entries, search_from)
        return entries

    def _walk(
        self,
        elem: ET.Element,
        parent_path: str,
        text: str,
        raw: bytes,
        out: list[ExtractedString],
        search_from: int,
    ) -> int:
        # Strip namespace for path
        tag = re.sub(r"\{[^}]+\}", "", elem.tag)
        path = f"{parent_path}/{tag}"

        # Extract attributes
        for attr_name, attr_value in elem.attrib.items():
            attr_name_clean = re.sub(r"\{[^}]+\}", "", attr_name)
            if attr_value.strip():
                search_from = self._add_attr(
                    attr_value, f"{path}/@{attr_name_clean}", text, raw, search_from, out
                )

        # Extract text content
        if elem.text and elem.text.strip():
            search_from = self._add_text(
                elem.text.strip(), f"{path}/text()", text, raw, search_from, out
            )

        # Recurse children
        for child in elem:
            search_from = self._walk(child, path, text, raw, out, search_from)

        # Extract tail text
        if elem.tail and elem.tail.strip():
            search_from = self._add_text(
                elem.tail.strip(), f"{parent_path}/text()", text, raw, search_from, out
            )

        return search_from

    @staticmethod
    def _add_text(
        value: str,
        path: str,
        text: str,
        raw: bytes,
        search_from: int,
        out: list[ExtractedString],
    ) -> int:
        idx = text.find(value, search_from)
        if idx == -1:
            return search_from
        byte_offset = len(text[:idx].encode("utf-8"))
        byte_length = len(value.encode("utf-8"))
        out.append(
            ExtractedString(text=value, byte_offset=byte_offset, byte_length=byte_length, path=path)
        )
        return idx + len(value)

    @staticmethod
    def _add_attr(
        value: str,
        path: str,
        text: str,
        raw: bytes,
        search_from: int,
        out: list[ExtractedString],
    ) -> int:
        # Attributes appear as attr="value" or attr='value'
        for quote in ('"', "'"):
            needle = f"{quote}{value}{quote}"
            idx = text.find(needle, search_from)
            if idx != -1:
                # Offset to content (skip quote)
                content_idx = idx + 1
                byte_offset = len(text[:content_idx].encode("utf-8"))
                byte_length = len(value.encode("utf-8"))
                out.append(
                    ExtractedString(
                        text=value,
                        byte_offset=byte_offset,
                        byte_length=byte_length,
                        path=path,
                    )
                )
                return idx + len(needle)
        return search_from


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


@register_extractor
class YamlExtractor:
    """Extract all string values from YAML using PyYAML's node API.

    Requires ``pyyaml`` (optional dependency).
    """

    @staticmethod
    def content_types() -> list[str]:
        return ["text/yaml", "application/yaml", "application/x-yaml", "text/x-yaml"]

    def extract(self, raw: bytes) -> list[ExtractedString]:
        try:
            import yaml
        except ImportError:
            return []

        text = raw.decode("utf-8")
        entries: list[ExtractedString] = []

        try:
            node = yaml.compose(text)
        except yaml.YAMLError:
            return entries

        if node is not None:
            self._walk(node, "", text, raw, entries)
        return entries

    def _walk(
        self,
        node: object,
        path: str,
        text: str,
        raw: bytes,
        out: list[ExtractedString],
    ) -> None:
        import yaml

        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                key_str = key_node.value if isinstance(key_node, yaml.ScalarNode) else str(key_node)
                child_path = f"{path}.{key_str}" if path else key_str

                if isinstance(key_node, yaml.ScalarNode) and key_node.value.strip():
                    self._add_scalar(key_node, child_path + "~key", text, raw, out)

                self._walk(value_node, child_path, text, raw, out)

        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                self._walk(item, f"{path}[{i}]", text, raw, out)

        elif isinstance(node, yaml.ScalarNode):
            if node.value.strip():
                self._add_scalar(node, path, text, raw, out)

    @staticmethod
    def _add_scalar(
        node: object,
        path: str,
        text: str,
        raw: bytes,
        out: list[ExtractedString],
    ) -> None:
        import yaml

        if not isinstance(node, yaml.ScalarNode):
            return
        # PyYAML start_mark gives line + column
        mark = node.start_mark
        if mark is None:
            return

        # Convert line:column to byte offset
        lines = text.split("\n")
        byte_offset = 0
        for i in range(min(mark.line, len(lines))):
            byte_offset += len(lines[i].encode("utf-8")) + 1  # +1 for \n

        # Add column offset — account for YAML quoting
        line = lines[mark.line] if mark.line < len(lines) else ""
        col_text = line[: mark.column]
        byte_offset += len(col_text.encode("utf-8"))

        value = node.value
        value_byte_len = len(value.encode("utf-8"))

        # Adjust for quote characters if the scalar is quoted
        if node.style in ("'", '"'):
            byte_offset += 1  # skip opening quote

        out.append(
            ExtractedString(
                text=value,
                byte_offset=byte_offset,
                byte_length=value_byte_len,
                path=path,
            )
        )
