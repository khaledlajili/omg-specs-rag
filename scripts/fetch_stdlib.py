#!/usr/bin/env python3
"""Stdlib-only PDF fetcher for OMG corpus (R1)."""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "corpus.yaml"
PDF_DIR = ROOT / "data" / "pdfs"
META_PATH = PDF_DIR / "download_meta.json"


def parse_simple_yaml(text: str) -> list[dict]:
    docs = []
    current = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.strip() == "documents:":
            continue
        m = re.match(r"^  - acronym:\s*(.+)$", line)
        if m:
            if current:
                docs.append(current)
            current = {"acronym": m.group(1).strip().strip('"')}
            continue
        if current is None:
            continue
        m = re.match(r"^    (\w+):\s*(.+)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val in ("null", "Null", "~"):
                val = None
            elif val in ("true", "True"):
                val = True
            elif val in ("false", "False"):
                val = False
            else:
                val = val.strip('"').strip("'")
            current[key] = val
    if current:
        docs.append(current)
    return docs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def page_count_est(path: Path) -> int | None:
    try:
        raw = path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page(?![s/])", raw))
    except Exception:
        return None


def filename_for(doc: dict) -> str:
    version = str(doc["version"]).replace(".", "_").replace(" ", "_")
    return f"{doc['acronym']}_{version}_{doc.get('status', 'formal')}.pdf"


def main() -> int:
    dry = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    only = None
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        only = set(a.upper() for a in sys.argv[i + 1 :] if not a.startswith("-"))

    text = MANIFEST.read_text(encoding="utf-8")
    docs = parse_simple_yaml(text)
    if only:
        docs = [d for d in docs if d["acronym"].upper() in only]

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    ctx = ssl.create_default_context()

    print(f"{'Acronym':<8} {'Version':<10} {'Status':<8} {'Action':<22} {'Pages':>6}  SHA")
    print("-" * 80)

    total_pages = 0
    total_bytes = 0

    for doc in docs:
        key = f"{doc['acronym']}:{doc['version']}:{doc.get('status', 'formal')}"
        name = filename_for(doc)
        path = PDF_DIR / name
        url = doc["pdf_url"]

        if path.exists() and not force:
            digest = sha256_file(path)
            if meta.get(key, {}).get("sha256") == digest:
                pages = meta[key].get("page_count_est") or page_count_est(path)
                size = path.stat().st_size
                total_pages += pages or 0
                total_bytes += size
                print(f"{doc['acronym']:<8} {doc['version']:<10} {doc.get('status','?'):<8} {'skip':<22} {pages or '?':>6}  {digest[:12]}")
                continue

        if dry:
            print(f"{doc['acronym']:<8} {doc['version']:<10} {doc.get('status','?'):<8} {'would download':<22} {'-':>6}  -")
            continue

        print(f"→ GET {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "omg-specs-rag/0.1 (research)"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                data = resp.read()
            path.write_bytes(data)
            digest = sha256_file(path)
            pages = page_count_est(path)
            size = path.stat().st_size
            meta[key] = {
                "file": name,
                "url": url,
                "sha256": digest,
                "size_bytes": size,
                "page_count_est": pages,
            }
            total_pages += pages or 0
            total_bytes += size
            print(f"{doc['acronym']:<8} {doc['version']:<10} {doc.get('status','?'):<8} {'downloaded':<22} {pages or '?':>6}  {digest[:12]}")
        except Exception as e:
            print(f"{doc['acronym']:<8} {doc['version']:<10} {doc.get('status','?'):<8} FAILED: {e}")

    if not dry:
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nMeta → {META_PATH}")
    print(f"TOTAL est. pages: {total_pages}   size: {total_bytes/1e6:.1f} MB")
    print("Note: page counts are approximate. Accurate counts come in R2 with Docling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
