"""
Ingest textbook PDFs into PostgreSQL + pgvector.

Usage:
    python ingest.py
    python ingest.py --policies-dir "C:\\path\\to\\pdfs"
    python ingest.py --replace
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from pathlib import Path

import pdfplumber
from pgvector.psycopg2 import register_vector

from db import embed_texts, get_db_connection

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 max sequence length is 256 tokens; keep chunks under that.
MAX_CHARS = 900
OVERLAP_CHARS = 180
HEADING_RE = re.compile(r"^(\d+(\.\d+)*|[A-Z][A-Z0-9 .:&/-]{6,})$")


def policy_id_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    match = re.search(r"(POL[-_][A-Z0-9]+[-_]?\d+)", stem, re.IGNORECASE)
    if match:
        return match.group(1).upper().replace("_", "-")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").upper()
    return f"POL-{slug[:40]}" if slug else f"POL-{hashlib.md5(stem.encode()).hexdigest()[:8]}"


def title_from_filename(path: Path) -> str:
    return re.sub(r"[-_]+", " ", path.stem).strip()


def guess_section(page_text: str, chunk_text: str) -> str:
    for line in page_text.splitlines():
        cleaned = line.strip()
        if cleaned and HEADING_RE.match(cleaned) and len(cleaned) < 80:
            if cleaned.lower() in chunk_text.lower() or cleaned[:20].lower() in chunk_text.lower():
                return cleaned[:120]
    first = chunk_text.strip().split("\n", 1)[0].strip()
    if 8 <= len(first) <= 80:
        return first
    return "General"


def chunk_page_text(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    if len(text) <= MAX_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start + MAX_CHARS // 2, end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    policy_id = policy_id_from_filename(pdf_path)
    title = title_from_filename(pdf_path)
    records: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            logger.warning("No pages in %s", pdf_path.name)
            return []

        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            tables = page.extract_tables() or []
            table_bits = []
            for table in tables:
                rows = [" | ".join((cell or "").strip() for cell in row) for row in table if row]
                table_bits.append("\n".join(rows))
            combined = page_text
            if table_bits:
                combined = (page_text + "\n\n" + "\n\n".join(table_bits)).strip()

            for i, chunk in enumerate(chunk_page_text(combined)):
                digest = hashlib.sha256(f"{policy_id}:{page_number}:{i}:{chunk}".encode()).hexdigest()[:16]
                records.append(
                    {
                        "chunk_id": f"{policy_id}-p{page_number}-{i}-{digest}",
                        "policy_id": policy_id,
                        "title": title,
                        "section": guess_section(page_text, chunk),
                        "page": page_number,
                        "text": chunk,
                    }
                )

    return records


def ingest_directory(policies_dir: Path, replace: bool) -> int:
    pdfs = sorted(policies_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {policies_dir.resolve()}")

    all_records: list[dict] = []
    for pdf in pdfs:
        logger.info("Parsing %s", pdf.name)
        chunks = extract_pdf_chunks(pdf)
        logger.info("  -> %d chunks", len(chunks))
        all_records.extend(chunks)

    if not all_records:
        raise RuntimeError("PDFs were found but no text could be extracted. Scanned-image PDFs need OCR.")

    logger.info("Embedding %d chunks...", len(all_records))
    vectors = embed_texts([r["text"] for r in all_records])
    for record, vector in zip(all_records, vectors):
        record["embedding"] = vector

    conn = get_db_connection()
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            if replace:
                cur.execute("TRUNCATE TABLE policy_chunks;")

            insert_sql = """
                INSERT INTO policy_chunks (
                    chunk_id, policy_id, title, section, page, text, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    section = EXCLUDED.section,
                    page = EXCLUDED.page,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    ingested_at = NOW();
            """
            rows = [
                (
                    r["chunk_id"],
                    r["policy_id"],
                    r["title"],
                    r["section"],
                    r["page"],
                    r["text"],
                    r["embedding"],
                )
                for r in all_records
            ]
            cur.executemany(insert_sql, rows)
        conn.commit()
    finally:
        conn.close()

    return len(all_records)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest textbook PDFs into pgvector")
    parser.add_argument(
        "--policies-dir",
        default=os.environ.get("POLICIES_DIR", "policies"),
        help="Folder containing textbook PDFs (default: ./policies)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear existing chunks before inserting (full re-ingest)",
    )
    args = parser.parse_args()

    policies_dir = Path(args.policies_dir)
    if not policies_dir.exists():
        policies_dir.mkdir(parents=True)
        raise SystemExit(
            f"Created empty folder {policies_dir.resolve()}. "
            "Copy your textbook PDFs into it, then run this command again."
        )

    count = ingest_directory(policies_dir, replace=args.replace)
    print(f"Ingested {count} chunks from {policies_dir.resolve()}")


if __name__ == "__main__":
    main()
