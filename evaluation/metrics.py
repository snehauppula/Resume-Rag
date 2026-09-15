"""Candidate-level retrieval metrics for Resume-RAG evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def dedupe_candidates(
    hits: Sequence[dict],
    id_key: str = "candidate_id",
) -> list[str]:
    """
    Preserve first-seen order of candidate IDs from chunk hits.

    Accepts either hit dicts with ``metadata.candidate_id`` or a top-level id_key.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        metadata = hit.get("metadata") if isinstance(hit, dict) else None
        if isinstance(metadata, dict) and metadata.get(id_key) is not None:
            cid = str(metadata.get(id_key))
        elif isinstance(hit, dict) and hit.get(id_key) is not None:
            cid = str(hit.get(id_key))
        else:
            continue
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    rel = {str(x) for x in relevant}
    if not rel:
        return 0.0
    top = [str(x) for x in retrieved[:k]]
    hit = sum(1 for c in top if c in rel)
    return hit / len(rel)


def precision_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = {str(x) for x in relevant}
    top = [str(x) for x in retrieved[:k]]
    hit = sum(1 for c in top if c in rel)
    return hit / k


def mrr(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    rel = {str(x) for x in relevant}
    if not rel:
        return 0.0
    for rank, cid in enumerate((str(x) for x in retrieved), start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(list(relevances)[:k], start=1):
        total += float(rel) / math.log2(i + 1)
    return total


def ndcg_at_k(
    relevant: Sequence[str],
    retrieved: Sequence[str],
    k: int,
    relevance_grades: dict[str, float] | None = None,
) -> float:
    """
    nDCG@k with binary grades by default (relevant=1).

    Optional ``relevance_grades`` maps candidate_id -> grade (e.g. 2/1/0).
    """
    if k <= 0:
        return 0.0

    grades = relevance_grades or {str(c): 1.0 for c in relevant}
    retrieved_list = [str(x) for x in retrieved[:k]]
    gains = [float(grades.get(cid, 0.0)) for cid in retrieved_list]
    dcg = dcg_at_k(gains, k)

    ideal = sorted((float(v) for v in grades.values() if float(v) > 0), reverse=True)
    if not ideal:
        # No relevant items labeled for this query
        return 0.0
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_metric(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
