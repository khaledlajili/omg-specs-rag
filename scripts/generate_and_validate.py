#!/usr/bin/env python3
"""R7 pilot: hybrid retrieval + LLM generation + citation validator."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import psycopg
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"
RRF_K = 60
TOP_DENSE = 15
TOP_LEX = 15
TOP_FINAL = 6

# Configure via environment (or hard-code for a quick local test)
# Examples:
#   export LLM_BASE_URL=http://localhost:11434/v1
#   export LLM_API_KEY=ollama
#   export LLM_MODEL=llama3.1:8b
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")


def rrf_fuse(rank_lists, k=RRF_K):
    scores = defaultdict(float)
    for ranks in rank_lists:
        for rank, cid in enumerate(ranks, 1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query: str, model) -> list[dict]:
    qvec = list(map(float, model.encode(query, normalize_embeddings=True)))
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        # Dense
        cur.execute("""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.is_latest_formal = true
            ORDER BY c.embedding <=> %s::vector LIMIT %s
        """, (qvec, TOP_DENSE))
        dense = [r[0] for r in cur.fetchall()]

        # Lexical
        cur.execute("""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.is_latest_formal = true
              AND c.tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC
            LIMIT %s
        """, (query, query, TOP_LEX))
        lex = [r[0] for r in cur.fetchall()]

        fused = rrf_fuse([dense, lex])[:TOP_FINAL]
        if not fused:
            return []

        ids = [c for c, _ in fused]
        cur.execute("""
            SELECT c.id, c.context_header, c.page_pdf_start, c.text, d.acronym, d.version
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
        """, (ids,))
        rows = {r[0]: r for r in cur.fetchall()}

        sources = []
        for i, cid in enumerate(ids, 1):
            r = rows[cid]
            sources.append({
                "id": i,
                "chunk_id": cid,
                "header": r[1],
                "page": r[2],
                "text": r[3],
                "acronym": r[4],
                "version": r[5],
            })
        return sources


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate(answer: dict, sources: list[dict]) -> tuple[bool, list[str]]:
    errors = []
    citations = answer.get("citations") or []
    by_id = {s["id"]: s for s in sources}

    for cit in citations:
        sid = cit.get("source")
        quote = cit.get("quote", "")
        if sid not in by_id:
            errors.append(f"Citation references unknown source {sid}")
            continue
        src_text = normalize(by_id[sid]["text"])
        if normalize(quote) not in src_text:
            errors.append(f"Quote not found in source {sid}: {quote[:80]}...")
    return len(errors) == 0, errors


def main() -> int:
    query = " ".join(sys.argv[1:]) or "What is required for XMI Schema Compliance?"
    print(f"Query: {query}\n")

    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    sources = retrieve(query, embed_model)
    if not sources:
        print("No sources retrieved. Refusing.")
        return 1

    # Confidence gate (R7)
    # Re-score top source with a quick embedding similarity proxy if needed
    # For now, refuse if the first source header is empty or we want a stricter check later
    print(f"Top source: {sources[0]['header']} p.{sources[0]['page']}")

    print("Retrieved sources:")
    for s in sources:
        print(f"  [S{s['id']}] {s['acronym']} {s['version']} > {s['header']}  p.{s['page']}")

    # Build prompt
    context = ""
    for s in sources:
        context += f"\n[S{s['id']}] {s['acronym']} {s['version']} > {s['header']} (page {s['page']})\n{s['text']}\n"

    prompt = f"""You are a precise assistant that answers questions about OMG specifications.
Answer ONLY from the sources below. If the sources do not contain the answer, set answerable to false.

Sources:
{context}

Question: {query}

Return ONLY valid JSON with this schema:
{{
  "answerable": true or false,
  "answer": "short answer that cites sources like [S1]",
  "citations": [
    {{"source": 1, "quote": "exact short quote from that source"}}
  ]
}}

Rules:
- Every factual claim must have a citation.
- Quotes must be exact substrings of the source text.
- If not answerable, set answerable=false and citations=[].
"""

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print(f"\nCalling model: {MODEL} @ {BASE_URL}")
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    # Extract JSON if the model wrapped it
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        print("Model did not return JSON:")
        print(raw)
        return 1
    try:
        answer = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print("JSON parse failed:", e)
        print(raw)
        return 1

    ok, errors = validate(answer, sources)
    print("\n=== Answer ===")
    print(json.dumps(answer, indent=2))
    print(f"\nValidation: {'PASS' if ok else 'FAIL'}")
    for e in errors:
        print("  -", e)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
