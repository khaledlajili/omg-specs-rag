#!/usr/bin/env python3
"""R4: hybrid retrieval (dense + full-text + RRF + FlashRank)."""

from __future__ import annotations

import sys
from collections import defaultdict

import psycopg
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"
RRF_K = 60
TOP_DENSE = 20
TOP_LEX = 20
TOP_RRF = 15
TOP_FINAL = 8


def rrf_fuse(rank_lists, k=RRF_K):
    scores = defaultdict(float)
    for ranks in rank_lists:
        for rank, cid in enumerate(ranks, 1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def main() -> int:
    query = " ".join(sys.argv[1:]) or "What is required for XMI Schema Compliance?"
    print(f"Query: {query}\n")

    embed = SentenceTransformer("BAAI/bge-small-en-v1.5")
    qvec = list(map(float, embed.encode(query, normalize_embeddings=True)))

    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.cursor()

        # Dense (latest formal by default)
        cur.execute("""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.is_latest_formal = true
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
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

        fused = rrf_fuse([dense, lex])[:TOP_RRF]
        if not fused:
            print("No results")
            return 0

        ids = [c for c, _ in fused]
        cur.execute("""
            SELECT c.id, c.context_header, c.page_pdf_start, c.text, d.acronym, d.version
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
        """, (ids,))
        rows = {r[0]: r for r in cur.fetchall()}

        # FlashRank
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        passages = [{"id": str(cid), "text": rows[cid][3]} for cid in ids]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked = ranker.rerank(rerank_req)

        print("Top results (FlashRank):")
        for i, item in enumerate(ranked[:TOP_FINAL], 1):
            cid = int(item["id"])
            r = rows[cid]
            print(f"--- [{i}] score={item['score']:.3f}  {r[4]} {r[5]} > {r[1]}  p.{r[2]}")
            print(r[3][:180].replace("\n", " "))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
