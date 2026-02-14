"""Redaction and format-specific annotation.

Three annotation modes:

* **structured** — Machine-readable: adds ``_data_integrity`` and
  ``_detections`` metadata to the document (JSON top-level keys, CSV
  metadata row, XML processing instruction).
* **cli** — Human-readable: wraps detected regions in
  ``<injection category="..." score="...">`` tags.
* **plain** — Minimal: redacts hot bytes and returns sidecar metadata dict.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scanner import Detection


# ---------------------------------------------------------------------------
# Byte-level redaction (shared by all modes)
# ---------------------------------------------------------------------------


def redact_bytes(raw: bytes, detections: list[Detection]) -> bytes:
    """Replace detected sub-spans with ``*`` of the same byte length.

    Works on the raw document bytes. Each detection's ``original_spans``
    contain byte offsets that point inside string content (never structural
    characters), so redaction preserves document validity.
    """
    result = bytearray(raw)
    for det in detections:
        for span in det.original_spans:
            length = span.byte_end - span.byte_start
            result[span.byte_start : span.byte_end] = b"*" * length
    return bytes(result)


# ---------------------------------------------------------------------------
# Structured annotation
# ---------------------------------------------------------------------------


def annotate_structured(
    raw: bytes,
    detections: list[Detection],
    content_type: str,
    score: float,
) -> bytes:
    """Redact hot spans and add structured metadata.

    For JSON: parse redacted output, insert ``_data_integrity`` and
    ``_detections`` at top level, re-serialize.
    For CSV: redact in-place, append a metadata comment row.
    For XML/HTML: redact text nodes, add a processing instruction.
    For YAML: redact in-place, add a metadata comment.
    """
    redacted = redact_bytes(raw, detections)
    base = content_type.split(";")[0].strip().lower()

    if base in ("application/json", "text/json"):
        return _annotate_json(redacted, detections, score)
    if base == "text/csv":
        return _annotate_csv(redacted, detections, score)
    if base in ("text/xml", "application/xml", "text/html"):
        return _annotate_xml(redacted, detections, score)
    if base in ("text/yaml", "application/yaml", "application/x-yaml", "text/x-yaml"):
        return _annotate_yaml(redacted, detections, score)

    # Fallback: just redact
    return redacted


def _annotate_json(
    redacted: bytes,
    detections: list[Detection],
    score: float,
) -> bytes:
    try:
        parsed = json.loads(redacted)
    except json.JSONDecodeError:
        return redacted

    meta = {
        "_data_integrity": "untrusted",
        "_detections": [_detection_dict(d) for d in detections],
        "_score": round(score, 4),
    }

    if isinstance(parsed, dict):
        parsed.update(meta)
    elif isinstance(parsed, list):
        parsed = {"_data_integrity": "untrusted", "data": parsed, **meta}
    else:
        parsed = {"_data_integrity": "untrusted", "data": parsed, **meta}

    return json.dumps(parsed, indent=2, ensure_ascii=False).encode("utf-8")


def _annotate_csv(
    redacted: bytes,
    detections: list[Detection],
    score: float,
) -> bytes:
    det_summary = "; ".join(
        f"{d.category}({d.heat:.1f})@{d.original_spans[0].path}"
        for d in detections
        if d.original_spans
    )
    meta_row = f'\n# _data_integrity=untrusted score={score:.4f} detections="{det_summary}"'
    return redacted + meta_row.encode("utf-8")


def _annotate_xml(
    redacted: bytes,
    detections: list[Detection],
    score: float,
) -> bytes:
    det_summary = "; ".join(
        f"{d.category}({d.heat:.1f})@{d.original_spans[0].path}"
        for d in detections
        if d.original_spans
    )
    pi = f'<?data-integrity untrusted score="{score:.4f}" detections="{det_summary}"?>\n'
    return pi.encode("utf-8") + redacted


def _annotate_yaml(
    redacted: bytes,
    detections: list[Detection],
    score: float,
) -> bytes:
    det_summary = "; ".join(
        f"{d.category}({d.heat:.1f})@{d.original_spans[0].path}"
        for d in detections
        if d.original_spans
    )
    comment = f'# _data_integrity: untrusted  score: {score:.4f}  detections: "{det_summary}"\n'
    return comment.encode("utf-8") + redacted


# ---------------------------------------------------------------------------
# CLI annotation
# ---------------------------------------------------------------------------


def annotate_cli(raw: bytes, detections: list[Detection]) -> bytes:
    """Redact hot spans and wrap detected regions in ``<injection>`` tags.

    Tags are inserted end-to-start to preserve byte positions.
    """
    result = bytearray(redact_bytes(raw, detections))

    # Collect all (byte_start, byte_end, category, heat) across detections,
    # then sort by start descending so insertions don't shift earlier offsets.
    tag_spans: list[tuple[int, int, str, float]] = []
    for det in detections:
        for span in det.original_spans:
            tag_spans.append((span.byte_start, span.byte_end, det.category, det.heat))

    tag_spans.sort(key=lambda t: t[0], reverse=True)

    for byte_start, byte_end, category, heat in tag_spans:
        close_tag = b"</injection>"
        open_tag = f'<injection category="{category}" score="{heat:.1f}">'.encode()
        result[byte_end:byte_end] = close_tag
        result[byte_start:byte_start] = open_tag

    return bytes(result)


# ---------------------------------------------------------------------------
# Plain annotation
# ---------------------------------------------------------------------------


def annotate_plain(
    raw: bytes,
    detections: list[Detection],
    score: float,
    content_type: str,
) -> tuple[bytes, dict]:
    """Redact hot spans and return sidecar metadata.

    Returns:
        Tuple of (redacted bytes, metadata dict).
    """
    redacted = redact_bytes(raw, detections)
    metadata = {
        "data_integrity": "untrusted",
        "score": round(score, 4),
        "content_type": content_type,
        "detections": [_detection_dict(d) for d in detections],
    }
    return redacted, metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detection_dict(det: Detection) -> dict:
    return {
        "category": det.category,
        "heat": round(det.heat, 2),
        "paths": list({s.path for s in det.original_spans}),
        "snippet": det.text_snippet[:120],
    }
