#!/usr/bin/env python3
"""Load clause lists (from inspect_parsed) into the clauses table."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "data" / "parsed"
DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"

# Map parsed stem → (acronym, version, status)
STEM_MAP = {
    "XMI_2_5_1_formal": ("XMI", "2.5.1", "formal"),
    "XMI_2_4_2_formal": ("XMI", "2.4.2", "formal"),
    "XMI_2_5_beta": ("XMI", "2.5", "beta"),
    "BPMN_2_0_2_formal": ("BPMN", "2.0.2", "formal"),
    "CMMN_1_1_formal": ("CMMN", "1.1", "formal"),
    "MOF_2_5_1_formal": ("MOF", "2.5.1", "formal"),
    "OCL_2_4_formal": ("OCL", "2.4", "formal"),
    "SysML_1_7_formal": ("SysML", "1.7", "formal"),
    "DMN_1_6_formal": ("DMN", "1.6", "formal"),
    "DMN_1_5_beta": ("DMN", "1.5", "beta"),
    "FUML_1_5_formal": ("FUML", "1.5", "formal"),
}


def is_noise(number: str, title: str) -> bool:
    """Drop obvious false positives."""
    if number == "0":
        return True
    if len(title) < 3:
        return True
    if re.match(r"^\.+$", title):
        return True
    return False


def main() -> int:
    stems = sys.argv[1:] or list(STEM_MAP.keys())

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for stem in stems:
                if stem not in STEM_MAP:
                    print(f"Skip unknown stem: {stem}")
                    continue
                acronym, version, status = STEM_MAP[stem]
                clauses_path = PARSED / f"{stem}_clauses.json"
                if not clauses_path.exists():
                    print(f"Missing {clauses_path}")
                    continue

                cur.execute(
                    "SELECT id FROM documents WHERE acronym=%s AND version=%s AND status=%s",
                    (acronym, version, status),
                )
                row = cur.fetchone()
                if not row:
                    print(f"Document not in DB: {acronym} {version} {status}")
                    continue
                doc_id = row[0]

                # Replace existing clauses for this document
                cur.execute("DELETE FROM clauses WHERE document_id=%s", (doc_id,))

                clauses = json.loads(clauses_path.read_text(encoding="utf-8"))
                kept = 0
                for c in clauses:
                    if is_noise(c["number"], c["title"]):
                        continue
                    cur.execute(
                        """
                        INSERT INTO clauses (document_id, number, title, heading_path, page_start, page_end, normative)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_id,
                            c["number"],
                            c["title"],
                            c["number"],  # simple path for now
                            c.get("page"),
                            c.get("page"),
                            None,
                        ),
                    )
                    kept += 1
                print(f"{acronym} {version}: loaded {kept} clauses (from {len(clauses)} candidates)")

        conn.commit()

        # Summary
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.acronym, d.version, COUNT(c.id)
                FROM documents d
                LEFT JOIN clauses c ON c.document_id = d.id
                GROUP BY d.acronym, d.version
                ORDER BY d.acronym, d.version
                """
            )
            print("\nClauses per document:")
            for r in cur.fetchall():
                print(f"  {r[0]:<8} {r[1]:<8}  {r[2]} clauses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
