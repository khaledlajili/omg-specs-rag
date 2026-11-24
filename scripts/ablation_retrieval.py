#!/usr/bin/env python3
"""R6: simple retrieval ablations on the gold set."""

import json
from collections import defaultdict
from pathlib import Path

import psycopg
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://omg:omg@localhost:5433/omg_rag"
GOLD = Path("data/gold/gold.jsonl")
RRF_K = 60


def rrf(rank_lists, k=RRF_K):
    scores = defaultdict(float)
    for ranks in rank_lists:
        for rank, cid in enumerate(ranks, 1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def get_headers(ids, conn):
    if not ids:
        return []
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, c.context_header
        FROM chunks c WHERE c.id = ANY(%s)
    """, (ids,))
    rows = {r[0]: r[1] for r in cur.fetchall()}
    return [rows.get(i, "") for i in ids]


def evaluate(mode: str, embed, ranker, gold):
    hits = 0
    total = 0
    with psycopg.connect(DATABASE_URL) as conn:
        for g in gold:
            if not g["answerable"]:
                continue
            total += 1
            q = g["question"]
            qvec = list(map(float, embed.encode(q, normalize_embeddings=True)))
            cur = conn.cursor()

            # Dense
            cur.execute("""
                SELECT c.id FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.is_latest_formal = true
                ORDER BY c.embedding <=> %s::vector LIMIT 20
            """, (qvec,))
            dense = [r[0] for r in cur.fetchall()]

            if mode == "dense":
                top = dense[:8]
            else:
                # Lexical
                cur.execute("""
                    SELECT c.id FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.is_latest_formal = true
                      AND c.tsv @@ websearch_to_tsquery('english', %s)
                    ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC
                    LIMIT 20
                """, (q, q))
                lex = [r[0] for r in cur.fetchall()]
                fused = [c for c, _ in rrf([dense, lex])][:15]
                if mode == "hybrid":
                    top = fused[:8]
                else:  # hybrid+rerank
                    if not fused:
                        top = []
                    else:
                        cur.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (fused,))
                        texts = {r[0]: r[1] for r in cur.fetchall()}
                        passages = [{"id": str(cid), "text": texts[cid]} for cid in fused if cid in texts]
                        ranked = ranker.rerank(RerankRequest(query=q, passages=passages))
                        top = [int(item["id"]) for item in ranked[:8]]

            headers = " ".join(get_headers(top, conn))
            expected = g.get("expected_clauses") or []
            if not expected:
                if top:
                    hits += 1
            else:
                if any(ec in headers for ec in expected):
                    hits += 1
    return hits, total


def main():
    gold = [json.loads(l) for l in GOLD.read_text().strip().splitlines()]
    embed = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

    for mode in ["dense", "hybrid", "hybrid+rerank"]:
        hits, total = evaluate(mode, embed, ranker, gold)
        print(f"{mode:16}  {hits}/{total} = {hits/max(total,1):.2f}")


if __name__ == "__main__":
    main()
