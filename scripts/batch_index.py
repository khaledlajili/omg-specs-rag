#!/usr/bin/env python3
"""Batch: one document at a time, explicit cleanup between steps."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "corpus.yaml"
PDF_DIR = ROOT / "data" / "pdfs"
PARSED = ROOT / "data" / "parsed"
META = PDF_DIR / "download_meta.json"


def stem_for(doc: dict) -> str:
    version = str(doc["version"]).replace(".", "_").replace(" ", "_")
    status = doc.get("status", "formal")
    return f"{doc['acronym']}_{version}_{status}"


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    docs = yaml.safe_load(MANIFEST.read_text())["documents"]
    meta = json.loads(META.read_text()) if META.exists() else {}

    def pages_est(doc):
        key = f"{doc['acronym']}:{doc['version']}:{doc.get('status','formal')}"
        return meta.get(key, {}).get("page_count_est") or 9999

    docs = sorted(docs, key=pages_est)

    for doc in docs:
        stem = stem_for(doc)
        key = f"{doc['acronym']}:{doc['version']}:{doc.get('status','formal')}"
        pdf_name = meta.get(key, {}).get("file") or f"{stem}.pdf"
        pdf_path = PDF_DIR / pdf_name
        json_path = PARSED / f"{stem}.json"
        clauses_path = PARSED / f"{stem}_clauses.json"

        print(f"\n========== {doc['acronym']} {doc['version']} ({doc.get('status')}) ==========")

        if not pdf_path.exists():
            print(f"SKIP – PDF missing: {pdf_path}")
            continue

        # 1. Parse
        if not json_path.exists():
            rc = run([sys.executable, "scripts/parse_spike.py", pdf_name])
            if rc != 0:
                print(f"Parse failed for {stem}")
                continue
        else:
            print("Already parsed")

        gc.collect()

        # 2. Inspect
        if not clauses_path.exists():
            rc = run([sys.executable, "scripts/inspect_parsed.py", stem])
            if rc != 0:
                print(f"Inspect failed for {stem}")
                continue

        gc.collect()

        # 3. Load clauses
        run([sys.executable, "scripts/load_clauses.py", stem])
        gc.collect()

        # 4. Chunk + embed
        run([sys.executable, "scripts/chunk_and_embed.py", stem])
        gc.collect()

        print(f"Finished {stem}")

    print("\n========== BATCH DONE ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
