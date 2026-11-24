#!/usr/bin/env python3
"""R4: hybrid retrieval = dense + full-text + Reciprocal Rank Fusion."""

from __future__ import annotations

import sys
from collections import defaultdict

import psycopg
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"
RRF_K = 60
TOP_DENSE = 20
TOP_LEX = 20
TOP_FINAL = 8


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for rank, chunk_id in enumerate(ranks, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def main() -> int:
    query = " ".join(sys.argv[1:]) or "What is required for XMI Schema Compliance?"
    print(f"Query: {query}\n")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    qvec = list(map(float, model.encode(query, normalize_embeddings=True)))

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Dense
            cur.execute(
                """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.acronym = 'XMI' AND d.version = '2.5.1'
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (qvec, TOP_DENSE),
            )
            dense_ids = [r[0] for r in cur.fetchall()]

            # Lexical
            cur.execute(
                """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.acronym = 'XMI' AND d.version = '2.5.1'
                  AND c.tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC
                LIMIT %s
                """,
                (query, query, TOP_LEX),
            )
            lex_ids = [r[0] for r in cur.fetchall()]

            fused = rrf_fuse([dense_ids, lex_ids])[:TOP_FINAL]
            if not fused:
                print("No results")
                return 0

            ids = [cid for cid, _ in fused]
            score_map = dict(fused)

            cur.execute(
                """
                SELECT c.id, c.context_header, c.page_pdf_start, LEFT(c.text, 220)
                FROM chunks c
                WHERE c.id = ANY(%s)
                """,
                (ids,),
            )
            rows = {r[0]: r for r in cur.fetchall()}

            for i, cid in enumerate(ids, 1):
                r = rows[cid]
                print(f"--- [{i}] rrf={score_map[cid]:.4f}  {r[1]}  p.{r[2]}")
                print(r[3].replace("\n", " ")[:200])
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
