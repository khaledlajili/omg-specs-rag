#!/usr/bin/env python3
"""Inspect Docling output and build a first clause tree from headings."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "data" / "parsed"

# Typical OMG numbered heading: 7, 7.2, 7.2.1, A.1, B.2.3 …
HEADING_RE = re.compile(
    r"^(?P<num>(?:\d+|[A-Z])(?:\.\d+){0,5})\s+(?P<title>.+)$"
)


def page_of(item: dict) -> int | None:
    """Best-effort page number from Docling provenance."""
    prov = item.get("prov") or []
    if not prov:
        return None
    # prov is usually a list of {page_no, bbox, ...}
    first = prov[0] if isinstance(prov, list) else prov
    if isinstance(first, dict):
        return first.get("page_no") or first.get("page")
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_parsed.py <stem>")
        print("Example: python scripts/inspect_parsed.py XMI_2_5_1_formal")
        return 1

    stem = sys.argv[1]
    path = PARSED / f"{stem}.json"
    if not path.exists():
        print(f"Not found: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    export = data["export"]
    texts = export.get("texts") or []
    tables = export.get("tables") or []

    print(f"Source: {data['source']}")
    print(f"Texts: {len(texts)}  Tables: {len(tables)}  Elapsed: {data.get('elapsed_seconds')}s")

    # Collect text items with page
    items = []
    for t in texts:
        text = (t.get("text") or t.get("orig") or "").strip()
        if not text:
            continue
        page = page_of(t)
        label = t.get("label")  # e.g. section_header, paragraph, ...
        items.append({"text": text, "page": page, "label": label})

    pages_with_text = sorted({i["page"] for i in items if i["page"] is not None})
    print(f"Pages that have text items: {len(pages_with_text)}  "
          f"(min={pages_with_text[0] if pages_with_text else None}, "
          f"max={pages_with_text[-1] if pages_with_text else None})")

    # Detect candidate headings
    headings = []
    for i in items:
        m = HEADING_RE.match(i["text"])
        if m and len(m.group("title")) > 2:
            # Prefer items Docling already marked as headers
            score = 2 if i["label"] and "header" in str(i["label"]).lower() else 1
            headings.append({
                "number": m.group("num"),
                "title": m.group("title").strip()[:80],
                "page": i["page"],
                "label": i["label"],
                "score": score,
            })

    # Deduplicate by number (keep highest score / first page)
    by_num = {}
    for h in headings:
        n = h["number"]
        if n not in by_num or h["score"] > by_num[n]["score"]:
            by_num[n] = h

    ordered = sorted(by_num.values(), key=lambda h: (
        [int(x) if x.isdigit() else ord(x) for x in re.split(r"\.", h["number"])],
        h["page"] or 0,
    ))

    print(f"\nCandidate clauses / headings found: {len(ordered)}")
    print("-" * 70)
    for h in ordered[:40]:
        print(f"  {h['number']:<12} p.{h['page'] or '?':<4}  {h['title']}")
    if len(ordered) > 40:
        print(f"  ... and {len(ordered) - 40} more")

    # Simple quality signals
    missing_page = sum(1 for i in items if i["page"] is None)
    print(f"\nQuality signals:")
    print(f"  text items with no page: {missing_page} / {len(items)}")
    print(f"  headings with no page:   {sum(1 for h in ordered if h['page'] is None)}")

    # Save clause list for later
    out = PARSED / f"{stem}_clauses.json"
    out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"\nWrote clause list → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
