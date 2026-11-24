"""R9: minimal FastAPI endpoint for the OMG Specs assistant."""

from __future__ import annotations

import os
import re
import json
from collections import defaultdict

import psycopg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from flashrank import Ranker, RerankRequest
from openai import OpenAI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI(title="OMG Specs RAG")
origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
CONFIDENCE_THRESHOLD = 0.3

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://omg:omg@localhost:5433/omg_rag")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

embed_model = None
ranker = None


def get_embed():
    global embed_model
    if embed_model is None:
        embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embed_model


def get_ranker():
    global ranker
    if ranker is None:
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
    return ranker


def normalize_query(q: str) -> str:
    return re.sub(r"[?!.]+$", "", q).strip()

class AskRequest(BaseModel):
    question: str


def rrf_fuse(rank_lists, k=60):
    scores = defaultdict(float)
    for ranks in rank_lists:
        for rank, cid in enumerate(ranks, 1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query: str):
    model = get_embed()
    qvec = [float(x) for x in next(model.embed([query]))]
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.is_latest_formal = true
            ORDER BY c.embedding <=> %s::vector LIMIT 15
        """, (qvec,))
        dense = [r[0] for r in cur.fetchall()]

        cur.execute("""
            SELECT c.id FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.is_latest_formal = true
              AND c.tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC
            LIMIT 15
        """, (query, query))
        lex = [r[0] for r in cur.fetchall()]

        fused = rrf_fuse([dense, lex])[:10]
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
        ranked = get_ranker().rerank(RerankRequest(query=query, passages=passages))
        sources = []
        for i, item in enumerate(ranked[:6], 1):
            r = rows[int(item["id"])]
            sources.append({
                "id": i,
                "header": r[1],
                "page": r[2],
                "text": r[3],
                "acronym": r[4],
                "version": r[5],
            })
        return sources


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


@app.post("/ask")
def ask(req: AskRequest):
    question = normalize_query(req.question)
    sources = retrieve(question)
    outcome = "answered"
    if not sources:
        outcome = "refused_gate"
        _log(req.question, outcome)
        return {"answerable": False, "answer": "Not found in the indexed specifications.", "citations": [], "sources": []}

    # Hard confidence gate: refuse if best source looks irrelevant
    best = sources[0]
    header = best.get("header", "").lower()
    text_snippet = (best.get("text") or "")[:200].lower()
    # Simple keyword overlap check
    q_words = set(question.lower().split())
    header_words = set(header.split())
    if len(q_words & header_words) == 0 and "shall" not in text_snippet:
        outcome = "refused_gate"
        _log(req.question, outcome)
        return {"answerable": False, "answer": "Not found in the indexed specifications.", "citations": [], "sources": []}

    context = ""
    for s in sources:
        context += f"\n[S{s['id']}] {s['acronym']} {s['version']} > {s['header']} (page {s['page']})\n{s['text']}\n"

    prompt = f"""Answer ONLY from the sources. If they do not contain the answer, set answerable to false.
Return ONLY valid JSON: {{"answerable": bool, "answer": str, "citations": [{{"source": int, "quote": str}}]}}
Sources:{context}
Question: {req.question}"""

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"answerable": False, "answer": "", "citations": [], "error": "no json"}
    try:
        answer = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"answerable": False, "answer": "", "citations": [], "error": "bad json"}

    # Validate quotes
    by_id = {s["id"]: s for s in sources}
    valid_cits = []
    for cit in answer.get("citations") or []:
        sid = cit.get("source")
        quote = cit.get("quote", "")
        if sid in by_id and normalize(quote) in normalize(by_id[sid]["text"]):
            valid_cits.append(cit)
    answer["citations"] = valid_cits
    answer["sources"] = [{"id": s["id"], "header": s["header"], "page": s["page"], "acronym": s["acronym"]} for s in sources]
    return answer


def _log(question: str, outcome: str):
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO queries (question, outcome) VALUES (%s, %s)", (question, outcome))
            conn.commit()
    except Exception:
        pass
