#!/usr/bin/env python3
"""Load manifest entries into the documents table (R2 helper)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import psycopg
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "corpus.yaml"
META = ROOT / "data" / "pdfs" / "download_meta.json"
DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"


def main() -> int:
    docs = yaml.safe_load(MANIFEST.read_text())["documents"]
    meta = {}
    if META.exists():
        meta = json.loads(META.read_text())

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for d in docs:
                key = f"{d['acronym']}:{d['version']}:{d.get('status', 'formal')}"
                m = meta.get(key, {})
                cur.execute(
                    """
                    INSERT INTO documents (
                        acronym, title, version, status, is_latest_formal,
                        superseded_by, pdf_url, sha256, page_count, file_name
                    ) VALUES (
                        %(acronym)s, %(title)s, %(version)s, %(status)s, %(is_latest_formal)s,
                        %(superseded_by)s, %(pdf_url)s, %(sha256)s, %(page_count)s, %(file_name)s
                    )
                    ON CONFLICT (acronym, version, status) DO UPDATE SET
                        sha256 = EXCLUDED.sha256,
                        page_count = EXCLUDED.page_count,
                        file_name = EXCLUDED.file_name
                    """,
                    {
                        "acronym": d["acronym"],
                        "title": d.get("title") or d["acronym"],
                        "version": str(d["version"]),
                        "status": d.get("status", "formal"),
                        "is_latest_formal": bool(d.get("is_latest_formal", False)),
                        "superseded_by": d.get("superseded_by"),
                        "pdf_url": d["pdf_url"],
                        "sha256": m.get("sha256"),
                        "page_count": m.get("page_count_est"),
                        "file_name": m.get("file"),
                    },
                )
        conn.commit()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT acronym, version, status, page_count, file_name FROM documents ORDER BY acronym, version")
            rows = cur.fetchall()
            print(f"{'Acronym':<8} {'Version':<8} {'Status':<8} {'Pages':>6}  File")
            print("-" * 60)
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<8} {r[2]:<8} {str(r[3] or '-'):>6}  {r[4] or '-'}")
            print(f"\nTotal rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
