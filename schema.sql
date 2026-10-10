-- Policy Assistant schema (PostgreSQL + pgvector)
-- all-MiniLM-L6-v2 produces 384-dimensional embeddings.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id    TEXT PRIMARY KEY,
    policy_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    section     TEXT NOT NULL,
    page        INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS policy_chunks_embedding_idx
    ON policy_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS policy_chunks_policy_id_idx
    ON policy_chunks (policy_id);
