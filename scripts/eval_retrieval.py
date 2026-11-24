#!/usr/bin/env python3
"""R6: evaluate hybrid retrieval against the gold set."""

from __future__ import annotations

import json
import re
from pathlib import Path

import psycopg
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"
GOLD = Path("data/gold/gold.jsonl")
RRF_K = 60
TOP_DENSE = 20
TOP_LEX = 20
TOP_RRF = 15
TOP_FINAL = 8


def rrf_fuse(rank_lists, k=RRF_K):
    from collections import defaultdict
    scores = defaultdict(float)
    for ranks in rank_lists:
        for rank, cid in enumerate(ranks, 1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query: str, embed, ranker, acronym=None, version=None):
    qvec = list(map(float, embed.encode(query, normalize_embeddings=True)))
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        where = ["d.is_latest_formal = true"]
        params = []
        if acronym:
            where.append("d.acronym = %s")
            params.append(acronym)
        if version:
            where.append("d.version = %s")
            params.append(version)
        where_sql = " AND ".join(where)

        cur.execute(f"""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, params + [qvec, TOP_DENSE])
        dense = [r[0] for r in cur.fetchall()]

        cur.execute(f"""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql} AND c.tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC
            LIMIT %s
        """, params + [query, query, TOP_LEX])
        lex = [r[0] for r in cur.fetchall()]

        fused = rrf_fuse([dense, lex])[:TOP_RRF]
        if not fused:
            return []

        ids = [c for c, _ in fused]
        cur.execute("""
            SELECT c.id, c.context_header, c.page_pdf_start, c.text, d.acronym, d.version
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
        """, (ids,))
        rows = {r[0]: r for r in cur.fetchall()}

        passages = [{"id": str(cid), "text": rows[cid][3]} for cid in ids]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            {
                "header": rows[int(item["id"])][1],
                "page": rows[int(item["id"])][2],
                "acronym": rows[int(item["id"])][4],
                "version": rows[int(item["id"])][5],
                "score": item["score"],
            }
            for item in ranked[:TOP_FINAL]
        ]


def main():
    gold = [json.loads(l) for l in GOLD.read_text().strip().splitlines()]
    embed = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

    hits = 0
    total_ans = 0
    refused = 0
    total_unans = 0

    for g in gold:
        results = retrieve(g["question"], embed, ranker, g.get("acronym"), g.get("version"))
        headers = " ".join(r["header"] for r in results)

        if g["answerable"]:
            total_ans += 1
            expected = g.get("expected_clauses") or []
            if not expected:
                # no specific clause expected – count as hit if any result
                if results:
                    hits += 1
            else:
                if any(any(ec in h for h in headers.split(" > ")) for ec in expected):
                    hits += 1
        else:
            total_unans += 1
            # simple refusal proxy: no high-scoring result or empty
            if not results or results[0]["score"] < 0.5:
                refused += 1

    print(f"Gold questions: {len(gold)}")
    print(f"Answerable recall@8 (clause match or any result): {hits}/{total_ans} = {hits/max(total_ans,1):.2f}")
    print(f"Unanswerable refusal proxy: {refused}/{total_unans} = {refused/max(total_unans,1):.2f}")


if __name__ == "__main__":
    main()
