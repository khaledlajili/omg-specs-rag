-- OMG Specs RAG – initial schema (R1)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    acronym         TEXT NOT NULL,
    title           TEXT NOT NULL,
    version         TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('formal', 'beta')),
    is_latest_formal BOOLEAN NOT NULL DEFAULT false,
    publication_date DATE,
    superseded_by   TEXT,
    pdf_url         TEXT NOT NULL,
    sha256          TEXT,
    page_count      INT,
    file_name       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (acronym, version, status)
);

CREATE TABLE IF NOT EXISTS clauses (
    id              SERIAL PRIMARY KEY,
    document_id     INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    number          TEXT,
    title           TEXT,
    heading_path    TEXT,
    page_start      INT,
    page_end        INT,
    normative       BOOLEAN
);

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_id       INT REFERENCES clauses(id) ON DELETE SET NULL,
    chunk_type      TEXT NOT NULL CHECK (chunk_type IN ('prose','table','list','code','caption')),
    text            TEXT NOT NULL,
    context_header  TEXT NOT NULL,
    page_pdf_start  INT NOT NULL,
    page_pdf_end    INT NOT NULL,
    page_label_start TEXT,
    page_label_end  TEXT,
    token_count     INT NOT NULL,
    has_shall       BOOLEAN NOT NULL DEFAULT false,
    text_hash       TEXT NOT NULL,
    -- embedding vector(...) will be added in R3 after model choice
    tsv             tsvector GENERATED ALWAYS AS (
                      to_tsvector('english', context_header || ' ' || text)
                    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS idx_documents_acronym ON documents(acronym);

CREATE TABLE IF NOT EXISTS queries (
    id              BIGSERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    filters         JSONB,
    outcome         TEXT,
    latency_ms      JSONB,
    tokens_in       INT,
    tokens_out      INT,
    provider        TEXT,
    model           TEXT,
    config_hash     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_chunks (
    query_id        BIGINT REFERENCES queries(id) ON DELETE CASCADE,
    chunk_id        BIGINT REFERENCES chunks(id) ON DELETE CASCADE,
    stage           TEXT,
    rank            INT,
    score           FLOAT
);

CREATE TABLE IF NOT EXISTS answers (
    id              BIGSERIAL PRIMARY KEY,
    query_id        BIGINT REFERENCES queries(id) ON DELETE CASCADE,
    answer          TEXT,
    citations       JSONB,
    validation_ok   BOOLEAN,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_budget (
    day             DATE NOT NULL,
    ip_hash         TEXT NOT NULL,
    count           INT NOT NULL DEFAULT 0,
    PRIMARY KEY (day, ip_hash)
);
