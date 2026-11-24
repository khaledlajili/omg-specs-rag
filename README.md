# OMG Specs Assistant — Citation-Backed Hybrid RAG

A domain-specific Retrieval-Augmented Generation (RAG) assistant for querying formal **Object Management Group (OMG)** standards—including **BPMN, CMMN, DMN, SysML, MOF, OCL, XMI, and fUML**—with verbatim, verified citations and strict refusal guardrails.

Unlike generic LLM wrappers that hallucinate normative requirements across dense 500-page PDF specifications, this system enforces **grounded-only answering**: every claim is backed by retrieved specification passages, quoted verbatim against the source chunk, and refused when evidence is missing.

---

## Why This Exists

Enterprise architects, systems engineers, and tool vendors regularly work across intersecting OMG standards (e.g., BPMN/CMMN/DMN process modeling, SysML systems engineering, MOF/XMI metamodeling, OCL constraints, and fUML execution semantics). Searching across multi-hundred-page PDFs is slow, while general-purpose LLMs frequently blend versions or invent compliance rules.

**OMG Specs Assistant** solves this with:
- **Hybrid Retrieval (Dense + Lexical):** Combines `BAAI/bge-small-en-v1.5` semantic vector search (`pgvector`) with PostgreSQL full-text search (`websearch_to_tsquery` + `ts_rank_cd`), merged via **Reciprocal Rank Fusion (RRF)** to capture both conceptual queries and exact specification terminology.
- **Version-Aware Filtering:** Queries strictly against `is_latest_formal = true` specifications so deprecated drafts never pollute answers.
- **Confidence Gating & Strict Refusal:** Rejects low-relevance queries before LLM generation and instructs the model to abstain (`answerable: false`) rather than guess.
- **Programmatic Quote Verification:** Every citation returned by the LLM is verified against the actual retrieved chunk text—unverified or fabricated quotes are stripped before reaching the UI.
- **Lean Memory Footprint:** Runs `BAAI/bge-small-en-v1.5` via ONNX Runtime (`fastembed`) instead of heavy PyTorch dependencies, keeping cloud deployment lightweight and fast.

---

## Architecture

```text
User Question (React + TypeScript UI)
        │
        ▼
FastAPI Backend (/ask)
        ├──► Dense Search: FastEmbed (BAAI/bge-small-en-v1.5) ──► PostgreSQL + pgvector
        ├──► Lexical Search: PostgreSQL tsvector / ts_rank_cd
        │
        ▼
Reciprocal Rank Fusion (RRF) ──► Top-6 Specification Chunks (Section Header + PDF Page)
        │
        ▼
Relevance Confidence Gate ──► Refuses if out-of-scope
        │
        ▼
LLM Grounded Synthesis (OpenAI-compatible API, temperature=0, strict JSON)
        │
        ▼
Substring Quote Verification ──► Verified Citations + Page-Level Sources
```

---

## Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, Psycopg 3, Pydantic
- **Retrieval & Embeddings:** PostgreSQL + `pgvector`, Full-Text Search (`tsvector`), `fastembed` (`BAAI/bge-small-en-v1.5`), Reciprocal Rank Fusion (RRF)
- **LLM Integration:** OpenAI-compatible SDK (configured for OpenRouter / custom endpoints) with structured JSON output
- **Frontend:** React 18, TypeScript, Vite
- **Deployment:** Render (API), Neon Serverless PostgreSQL (`pgvector`), GitHub Pages (Frontend)

---

## API Overview

### `POST /ask`
Queries the indexed OMG specifications and returns a structured, citation-verified response.

**Request:**
```json
{
  "question": "What is required for XMI Schema Compliance?"
}
```

**Response:**
```json
{
  "answerable": true,
  "answer": "To achieve XMI Schema Compliance...",
  "citations": [
    {
      "source": 1,
      "quote": "Exact verbatim substring verified against source chunk [S1]..."
    }
  ],
  "sources": [
    {
      "id": 1,
      "header": "2.1 XMI Schema Compliance",
      "page": 14,
      "acronym": "XMI"
    }
  ]
}
```

### `GET /` & `GET /health` (supports `HEAD`)
Lightweight health endpoints for uptime monitoring.

---

## Local Setup

### 1. Backend
```bash
# Install dependencies with uv
uv sync

# Configure environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/omg_rag?sslmode=require"
export LLM_BASE_URL="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)"
export LLM_API_KEY="your_api_key"
export LLM_MODEL="nvidia/nemotron-3-super-120b-a12b:free"
export CORS_ORIGINS="http://localhost:5173"

# Start the FastAPI server
uv run uvicorn omg_specs_rag.api:app --reload --port 8000
```

### 2. Frontend
```bash
cd ui
npm install
npm run dev
```