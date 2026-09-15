"""
LLM-based chunk reranker (extracted from notebook Part 8).

Pipeline role: reorder an already-retrieved chunk pool by query relevance.
Does not search the vector store.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from groq import Groq

from .config import settings

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b(10|[0-9])\b")


def extract_llm_text(message: Any) -> str:
    """
    Prefer message.content; fall back to reasoning for GPT-OSS-style replies.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    if isinstance(message, dict):
        for key in ("content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def parse_relevance_score(raw_text: str, default: float = 0.0) -> float:
    """
    Parse a 0–10 relevance score from model text.

    Prefers explicit markers (SCORE: N) then "N out of 10", then the first 0–10 digit.
    """
    if not raw_text or not raw_text.strip():
        return default

    text = raw_text.strip()
    # Fast path: entire reply is an integer
    if re.fullmatch(r"10|[0-9]", text):
        return float(text)

    marked = re.search(
        r"(?:final\s+)?(?:score|relevance)\s*[:=]\s*(10|[0-9])\b",
        text,
        flags=re.I,
    )
    if marked:
        return float(marked.group(1))

    # Prefer "N out of 10" / "N/10" style before falling back to the first 0–10 digit.
    out_of = re.search(r"\b(10|[0-9])\s*(?:/|out of)\s*10\b", text, flags=re.I)
    if out_of:
        return float(out_of.group(1))

    matches = _SCORE_RE.findall(text)
    if not matches:
        return default
    # Prefer the last digit for long chain-of-thought answers that end with the score.
    return float(matches[-1])


class LLMReranker:
    """Score retrieved chunks with Groq and return them in descending score order."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        client: Groq | None = None,
    ) -> None:
        self.model_name = model_name or settings.groq_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        key = api_key or settings.groq_api_key
        if client is not None:
            self.client = client
        elif key:
            self.client = Groq(api_key=key)
        else:
            self.client = None

    def require_client(self) -> Groq:
        if self.client is None:
            raise ValueError(
                "Groq client is required for LLM reranking. "
                "Set GROQ_API_KEY in .env or pass api_key/client."
            )
        return self.client

    def score_chunk(self, query: str, chunk_text: str) -> tuple[float, str | None]:
        """
        Return (score, warning). On failure, score defaults to 0.0 and warning is set.
        """
        client = self.require_client()
        prompt = (
            "You score how relevant a resume chunk is to a recruiter query.\n"
            f"Query: {query}\n"
            f"Chunk: {chunk_text}\n\n"
            "Think briefly if needed, then end with exactly one line:\n"
            "SCORE: <integer from 0 to 10>\n"
            "Do not write anything after that line."
        )
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = response.choices[0].message if response.choices else None
            raw = extract_llm_text(message) if message is not None else ""
            if not raw:
                warning = "empty LLM score response; defaulting to 0"
                logger.warning(warning)
                return 0.0, warning

            score = parse_relevance_score(raw, default=-1.0)
            if score < 0:
                warning = f"unparseable LLM score text={raw!r}; defaulting to 0"
                logger.warning(warning)
                return 0.0, warning

            # Clamp to expected range
            score = max(0.0, min(10.0, score))
            return score, None
        except Exception as exc:
            warning = f"LLM scoring failed: {exc}"
            logger.warning(warning)
            return 0.0, warning

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Reorder retrieved hit dicts by LLM relevance score (desc).

        Each input item should look like VectorStoreManager.similarity_search output:
        {"document", "metadata", "distance", "id", ...}.

        Adds ``reranker_score`` and optional ``reranker_warning``; never drops items
        from the pool before applying top_k.
        """
        if not documents:
            return []

        scored: list[dict[str, Any]] = []
        for index, hit in enumerate(documents, start=1):
            item = dict(hit)
            chunk_text = str(item.get("document") or "")
            score, warning = self.score_chunk(query, chunk_text)
            item["reranker_score"] = score
            item["original_rank"] = index
            if warning:
                item["reranker_warning"] = warning
            scored.append(item)

        scored.sort(
            key=lambda x: (float(x.get("reranker_score", 0.0)), -int(x.get("original_rank", 0))),
            reverse=True,
        )
        return scored[: max(0, top_k)]
