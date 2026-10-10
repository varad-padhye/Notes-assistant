"""Retrieval-augmented generation pipeline for the Textbook Assistant."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder

from db import embed_query, get_db_connection

GEMINI_KEY_FILE = Path(__file__).resolve().parent / "gemini_key.txt"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

logger = logging.getLogger(__name__)

_reranker_model: CrossEncoder | None = None


@dataclass
class PolicyChunk:
    chunk_id: str
    policy_id: str
    title: str
    section: str
    page: int
    text: str
    score: float = 0.0


@dataclass
class Citation:
    policy_id: str
    title: str
    section: str
    page: int


@dataclass
class AssistantResponse:
    answer_found: bool
    conflict_detected: bool
    answer: str
    citations: list[Citation]
    confidence: str
    escalation_note: Optional[str]
    latency_ms: int
    chunks_retrieved: int


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def vector_search(embedding: list[float], top_k: int = 8) -> list[PolicyChunk]:
    sql = """
    SELECT
        chunk_id,
        policy_id,
        title,
        section,
        page,
        text,
        1 - (embedding <=> %s::vector) AS score
    FROM policy_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding, embedding, top_k))
            rows = cur.fetchall()
        return [
            PolicyChunk(
                chunk_id=row[0],
                policy_id=row[1],
                title=row[2],
                section=row[3],
                page=row[4],
                text=row[5],
                score=float(row[6]),
            )
            for row in rows
        ]
    finally:
        conn.close()


def rerank(query: str, chunks: list[PolicyChunk], top_k: int = 5) -> list[PolicyChunk]:
    if not chunks:
        return []

    reranker = get_reranker_model()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]


def build_context_block(chunks: list[PolicyChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{chunk.policy_id} | {chunk.title} | "
            f"{chunk.section} | p.{chunk.page}]"
        )
        parts.append(f"Excerpt {i}:\n{header}\n{chunk.text}")
    return "\n\n".join(parts)


SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful textbook learning assistant.

Answer questions using ONLY the retrieved textbook excerpts below. Explain
ideas clearly and help the learner understand definitions, methods, and
examples when the excerpts support them.

RETRIEVED CONTEXT
─────────────────
{context_block}

Rules:

1. Use only the retrieved textbook excerpts. Do not use outside knowledge.

2. Cite every factual statement using the textbook title and page number
   from the retrieved context, for example: "[Textbook title, p. 5]".

3. If the excerpts do not contain enough information to answer, say that the
   topic is not covered in the retrieved textbook excerpts and set
   answer_found = false.

4. If the excerpts contain conflicting explanations, describe the difference
   with citations, set conflict_detected = true, and recommend checking the
   cited textbook edition or asking an instructor.

5. Return the following JSON structure:

{{
  "answer_found": true,
  "conflict_detected": false,
  "answer": "Statistical learning uses data to estimate a function that can explain or predict outcomes [Textbook title, p. 5].",
  "citations": [
    {{
      "policy_id": "POL-ISLP-WEBSITE",
      "title": "An Introduction to Statistical Learning",
      "section": "What Is Statistical Learning?",
      "page": 5
    }}
  ],
  "confidence": "high",
  "escalation_note": null
}}

6. Return ONLY valid JSON.
"""


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _citation_from_dict(raw: dict) -> Citation:
    return Citation(
        policy_id=str(raw.get("policy_id", "")),
        title=str(raw.get("title", "")),
        section=str(raw.get("section", "")),
        page=int(raw.get("page") or 0),
    )


def load_gemini_api_key() -> str:
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    if GEMINI_KEY_FILE.exists():
        saved = GEMINI_KEY_FILE.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    typed = input("Enter Gemini API key: ").strip()
    if not typed:
        raise SystemExit("A Gemini API key is required. Get one at https://aistudio.google.com/apikey")
    GEMINI_KEY_FILE.write_text(typed, encoding="utf-8")
    print(f"Saved API key to {GEMINI_KEY_FILE.name} for next runs.")
    return typed


def make_gemini_client() -> genai.Client:
    return genai.Client(api_key=load_gemini_api_key())


def call_gemini(
    query: str,
    context_block: str,
    client: genai.Client,
    model: str | None = None,
    max_tokens: int = 4096,
) -> dict:
    system = SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block)
    model_name = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    message = client.models.generate_content(
        model=model_name,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    raw_text = _strip_json_fences(message.text or "")
    if not raw_text:
        raise RuntimeError("Gemini returned an empty response")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        candidates = message.candidates or []
        finish_reason = (
            candidates[0].finish_reason if candidates else "unknown"
        )
        logger.error(
            "Gemini returned non-JSON (finish_reason=%s): %s",
            finish_reason,
            raw_text[:300],
        )
        raise ValueError(
            f"Malformed JSON from model (finish_reason={finish_reason}): {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON was not an object")
    return parsed


def answer_policy_question(
    query: str,
    client: genai.Client,
    retrieval_top_k: int = 8,
    rerank_top_k: int = 5,
) -> AssistantResponse:
    t0 = time.monotonic()

    try:
        embedding = embed_query(query)
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return _refusal_response(t0, chunks_retrieved=0)

    try:
        raw_chunks = vector_search(embedding, top_k=retrieval_top_k)
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        return _refusal_response(t0, chunks_retrieved=0)

    if not raw_chunks:
        logger.info("No chunks retrieved for query: %r", query[:80])
        return _refusal_response(t0, chunks_retrieved=0)

    try:
        ranked_chunks = rerank(query, raw_chunks, top_k=rerank_top_k)
    except Exception as exc:
        logger.warning("Re-rank failed, falling back to bi-encoder order: %s", exc)
        ranked_chunks = raw_chunks[:rerank_top_k]

    context_block = build_context_block(ranked_chunks)

    raw_response: dict = {}
    for attempt in (1, 2):
        try:
            raw_response = call_gemini(query, context_block, client)
            break
        except ValueError as exc:
            logger.error("JSON parse failure on attempt %d: %s", attempt, exc)
            if attempt == 2:
                return _refusal_response(
                    t0,
                    len(ranked_chunks),
                    reason=f"Gemini returned an invalid response: {exc}",
                )
            time.sleep(2)
        except Exception as exc:
            logger.warning("Gemini API error attempt %d: %s", attempt, exc)
            if attempt == 2:
                return _refusal_response(
                    t0,
                    len(ranked_chunks),
                    reason=f"Gemini generation failed: {exc}",
                )
            time.sleep(2)

    try:
        citations_raw = raw_response.get("citations") or []
        response = AssistantResponse(
            answer_found=bool(raw_response.get("answer_found", False)),
            conflict_detected=bool(raw_response.get("conflict_detected", False)),
            answer=str(raw_response.get("answer", "")),
            citations=[_citation_from_dict(c) for c in citations_raw if isinstance(c, dict)],
            confidence=str(raw_response.get("confidence", "low")),
            escalation_note=raw_response.get("escalation_note"),
            latency_ms=int((time.monotonic() - t0) * 1000),
            chunks_retrieved=len(ranked_chunks),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Response schema mismatch: %s — raw: %s", exc, raw_response)
        return _refusal_response(t0, len(ranked_chunks))

    return response


def _refusal_response(
    t0: float,
    chunks_retrieved: int,
    reason: str | None = None,
) -> AssistantResponse:
    return AssistantResponse(
        answer_found=False,
        conflict_detected=False,
        answer=(
            "I couldn't generate an answer from the retrieved textbook "
            "excerpts. Please check the configured Gemini model and API key."
            if reason and reason.startswith("Gemini")
            else "I wasn't able to retrieve a reliable answer from the "
            "textbook materials right now. Please check that the textbook "
            "is indexed and the database is available."
        ),
        citations=[],
        confidence="low",
        escalation_note=reason or "System could not complete retrieval or generation.",
        latency_ms=int((time.monotonic() - t0) * 1000),
        chunks_retrieved=chunks_retrieved,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = make_gemini_client()

    print("Textbook Assistant Ready")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a textbook question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        try:
            response = answer_policy_question(query=query, client=client)

            print("\n=== RESPONSE ===")
            print(f"Answer Found      : {response.answer_found}")
            print(f"Confidence        : {response.confidence}")
            print(f"Conflict Detected : {response.conflict_detected}")
            print(f"Latency           : {response.latency_ms} ms")
            print(f"Chunks Retrieved  : {response.chunks_retrieved}")
            print("\nAnswer:")
            print(response.answer)

            if response.citations:
                print("\nCitations:")
                for c in response.citations:
                    print(f"- {c.title} | {c.section} | p.{c.page}")

            if response.escalation_note:
                print("\nEscalation Note:")
                print(response.escalation_note)

            print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as exc:
            print(f"\nUnexpected error: {exc}\n")
