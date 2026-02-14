"""Virtual text construction and offset mapping.

Builds a "virtual text" by joining extracted strings with newline separators,
then maps detection spans back to original byte positions in the source document.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExtractedString:
    """A string extracted from a structured document."""

    text: str
    """The string content."""

    byte_offset: int
    """Start position in original document (bytes)."""

    byte_length: int
    """Length in original document (bytes)."""

    path: str
    """Location reference (JSON pointer / CSV row:col / XPath / dot notation)."""


@dataclass(slots=True)
class OriginalSpan:
    """A span mapped back to original document byte positions."""

    byte_start: int
    """Start in original document."""

    byte_end: int
    """End in original document."""

    entry_index: int
    """Which ExtractedString this maps to."""

    path: str
    """Location reference from the ExtractedString."""


class SpanMap:
    """Maps between virtual text character offsets and original document byte offsets.

    Virtual text is constructed by joining all extracted strings with ``\\n``
    separators. Detection runs on this virtual text, then ``map_span`` translates
    character ranges back to byte positions in the original document.
    """

    __slots__ = ("entries", "virtual_offsets", "virtual_text")

    def __init__(
        self,
        entries: list[ExtractedString],
        virtual_offsets: list[int],
        virtual_text: str,
    ) -> None:
        self.entries = entries
        self.virtual_offsets = virtual_offsets
        self.virtual_text = virtual_text

    @staticmethod
    def build(entries: list[ExtractedString]) -> SpanMap:
        """Construct a SpanMap from extracted strings.

        Each entry's text is placed in the virtual text separated by ``\\n``.
        ``virtual_offsets[i]`` records the character position where
        ``entries[i].text`` begins in the virtual text.
        """
        if not entries:
            return SpanMap(entries=[], virtual_offsets=[], virtual_text="")

        parts: list[str] = []
        offsets: list[int] = []
        pos = 0
        for entry in entries:
            offsets.append(pos)
            parts.append(entry.text)
            pos += len(entry.text) + 1  # +1 for \n separator

        virtual_text = "\n".join(parts)
        return SpanMap(entries=entries, virtual_offsets=offsets, virtual_text=virtual_text)

    def map_span(self, virt_start: int, virt_end: int) -> list[OriginalSpan]:
        """Map a virtual-text character span to original byte positions.

        A single virtual span may cross entry boundaries (the ``\\n`` separators
        between extracted strings). In that case multiple :class:`OriginalSpan`
        objects are returned — one per entry touched. Characters that fall on a
        separator are skipped.

        Character offsets in virtual text are converted to byte offsets in the
        original document via UTF-8 encoding of the relevant slice.
        """
        if not self.entries:
            return []

        results: list[OriginalSpan] = []
        n = len(self.entries)

        for i in range(n):
            entry = self.entries[i]
            entry_vstart = self.virtual_offsets[i]
            entry_vend = entry_vstart + len(entry.text)

            # Check overlap between [virt_start, virt_end) and [entry_vstart, entry_vend)
            overlap_start = max(virt_start, entry_vstart)
            overlap_end = min(virt_end, entry_vend)

            if overlap_start >= overlap_end:
                continue

            # Convert character offsets within entry to byte offsets
            local_char_start = overlap_start - entry_vstart
            local_char_end = overlap_end - entry_vstart

            byte_start_in_entry = len(entry.text[:local_char_start].encode("utf-8"))
            byte_end_in_entry = len(entry.text[:local_char_end].encode("utf-8"))

            results.append(
                OriginalSpan(
                    byte_start=entry.byte_offset + byte_start_in_entry,
                    byte_end=entry.byte_offset + byte_end_in_entry,
                    entry_index=i,
                    path=entry.path,
                )
            )

        return results
