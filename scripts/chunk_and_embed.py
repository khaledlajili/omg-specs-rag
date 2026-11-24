#!/usr/bin/env python3
"""R3: chunk parsed Docling JSON and embed with BGE-small (XMI pilot)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "data" / "parsed"
DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"

# Target chunk size in characters (approx tokens * 4)
MIN_CHARS = 400
MAX_CHARS = 1600  # ~400 tokens
HEADING_RE = re.compile(r"^(?P<num>(?:\d+|[A-Z])(?:\.\d+){0,5})\s+(?P<title>.+)$")


def page_of(item: dict) -> int | None:
    prov = item.get("prov") or []
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    if isinstance(first, dict):
        return first.get("page_no") or first.get("page")
    return None


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def split_prose(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split long prose on paragraph boundaries, then sentences if needed."""
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                # hard split on sentences
                sents = re.split(r"(?<=[.!?])\s+", p)
                buf = ""
                for s in sents:
                    if len(buf) + len(s) + 1 <= max_chars:
                        buf = f"{buf} {s}".strip() if buf else s
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks_from_export(export: dict, acronym: str = "", version: str = "") -> list[dict]:
    texts = export.get("texts") or []
    tables = export.get("tables") or []

    # Group consecutive prose under the current heading
    current_heading = "Preamble"
    current_number = ""
    buffer = []
    buffer_pages = []
    results = []

    def flush():
        nonlocal buffer, buffer_pages
        if not buffer:
            return
        full = "\n\n".join(buffer).strip()
        if not full:
            buffer, buffer_pages = [], []
            return
        pages = [p for p in buffer_pages if p is not None]
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else page_start
        for piece in split_prose(full):
            results.append({
                "chunk_type": "prose",
                "text": piece,
                "context_header": f"{acronym} {version} > {current_number} {current_heading}".strip(),
                "page_pdf_start": page_start,
                "page_pdf_end": page_end,
                "has_shall": "shall" in piece.lower(),
            })
        buffer, buffer_pages = [], []

    for t in texts:
        text = (t.get("text") or t.get("orig") or "").strip()
        if not text:
            continue
        page = page_of(t)
        m = HEADING_RE.match(text)
        if m and len(m.group("title")) > 2 and m.group("num") != "0":
            flush()
            current_number = m.group("num")
            current_heading = m.group("title").strip()[:100]
            continue
        buffer.append(text)
        buffer_pages.append(page)
    flush()

    # Tables as their own chunks
    for tbl in tables:
        # Docling table data is complex; take a simple string representation
        data = tbl.get("data") or {}
        # Prefer markdown-like export if present
        raw = tbl.get("text") or str(data)[:3000]
        if not raw or len(raw) < 10:
            continue
        page = page_of(tbl) or 1
        results.append({
            "chunk_type": "table",
            "text": raw[:4000],
            "context_header": "Table",
            "page_pdf_start": page,
            "page_pdf_end": page,
            "has_shall": "shall" in raw.lower(),
        })

    return results


def main() -> int:
    stem = sys.argv[1] if len(sys.argv) > 1 else "XMI_2_5_1_formal"
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
    if stem not in STEM_MAP:
        print(f"Unknown stem: {stem}")
        return 1
    acronym, version, status = STEM_MAP[stem]

    path = PARSED / f"{stem}.json"
    if not path.exists():
        print(f"Missing {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = build_chunks_from_export(data["export"], acronym, version)
    print(f"Built {len(chunks)} chunks from {stem}")

    # Embed
    print("Loading BGE-small …")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    # BGE retrieval: optional query instruction; for documents just encode
    print("Embedding …")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Ensure vector column exists (384 dims)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='chunks' AND column_name='embedding'
                    ) THEN
                        ALTER TABLE chunks ADD COLUMN embedding vector(384);
                    END IF;
                END $$;
                """
            )
            cur.execute(
                "SELECT id FROM documents WHERE acronym=%s AND version=%s AND status=%s",
                (acronym, version, status),
            )
            row = cur.fetchone()
            if not row:
                print("Document not in DB")
                return 1
            doc_id = row[0]

            cur.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))

            for c, vec in zip(chunks, vectors):
                cur.execute(
                    """
                    INSERT INTO chunks (
                        document_id, chunk_type, text, context_header,
                        page_pdf_start, page_pdf_end, token_count, has_shall,
                        text_hash, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        doc_id,
                        c["chunk_type"],
                        c["text"],
                        c["context_header"],
                        c["page_pdf_start"],
                        c["page_pdf_end"],
                        max(1, len(c["text"]) // 4),
                        c["has_shall"],
                        text_hash(c["text"]),
                        list(map(float, vec)),
                    ),
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id=%s", (doc_id,))
            n = cur.fetchone()[0]
            cur.execute(
                "SELECT chunk_type, COUNT(*) FROM chunks WHERE document_id=%s GROUP BY chunk_type",
                (doc_id,),
            )
            print(f"\nStored {n} chunks for {acronym} {version}")
            for r in cur.fetchall():
                print(f"  {r[0]}: {r[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
