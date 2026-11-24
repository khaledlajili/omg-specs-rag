#!/usr/bin/env python3
"""R5: validate gold set schema and that expected clauses exist in the DB."""

import json
from pathlib import Path
import psycopg

GOLD = Path("data/gold/gold.jsonl")
DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"
REQUIRED = {"id", "question", "answerable"}

def main():
    lines = GOLD.read_text().strip().splitlines()
    errors = []
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        for i, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON – {e}")
                continue
            missing = REQUIRED - obj.keys()
            if missing:
                errors.append(f"{obj.get('id','?')}: missing fields {missing}")
            # Check expected clauses exist if specified
            for clause in obj.get("expected_clauses") or []:
                if not obj.get("acronym"):
                    continue
                cur.execute("""
                    SELECT 1 FROM clauses cl
                    JOIN documents d ON d.id = cl.document_id
                    WHERE d.acronym = %s AND cl.number = %s
                    LIMIT 1
                """, (obj["acronym"], clause))
                if not cur.fetchone():
                    errors.append(f"{obj['id']}: expected clause {clause} not found for {obj['acronym']}")
    print(f"Validated {len(lines)} questions")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print("  -", e)
    else:
        print("All gold records valid and expected clauses resolve.")

if __name__ == "__main__":
    main()
