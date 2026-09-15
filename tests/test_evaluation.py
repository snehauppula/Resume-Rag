"""Unit tests for evaluation metrics and reranker parsing (no Groq / no Chroma)."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    dedupe_candidates,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from project.reranker import extract_llm_text, parse_relevance_score


def test_recall_at_4_full_coverage():
    relevant = ["A", "B"]
    retrieved = ["C", "A", "D", "B"]
    assert recall_at_k(relevant, retrieved, 4) == 1.0


def test_recall_at_1_partial():
    relevant = ["A", "B"]
    retrieved = ["C", "A", "D", "B"]
    assert recall_at_k(relevant, retrieved, 1) == 0.0
    assert recall_at_k(relevant, retrieved, 2) == pytest.approx(0.5)


def test_precision_at_4():
    relevant = ["A", "B"]
    retrieved = ["C", "A", "D", "B"]
    assert precision_at_k(relevant, retrieved, 4) == pytest.approx(0.5)


def test_mrr_first_relevant_at_rank_2():
    relevant = ["A", "B"]
    retrieved = ["C", "A", "D", "B"]
    assert mrr(relevant, retrieved) == pytest.approx(0.5)


def test_mrr_none_relevant():
    assert mrr(["A"], ["X", "Y", "Z"]) == 0.0


def test_ndcg_at_4_binary():
    relevant = ["A", "B"]
    retrieved = ["A", "C", "B", "D"]
    # gains: 1, 0, 1, 0
    dcg = 1 / math.log2(2) + 0 + 1 / math.log2(4) + 0
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(relevant, retrieved, 4) == pytest.approx(dcg / idcg)


def test_ndcg_zero_relevant_labels():
    assert ndcg_at_k([], ["A", "B"], 4) == 0.0


def test_dedupe_candidates_preserves_first_rank():
    hits = [
        {"metadata": {"candidate_id": "A"}, "document": "a1"},
        {"metadata": {"candidate_id": "B"}, "document": "b1"},
        {"metadata": {"candidate_id": "A"}, "document": "a2"},
        {"metadata": {"candidate_id": "C"}, "document": "c1"},
        {"candidate_id": "B"},
    ]
    assert dedupe_candidates(hits) == ["A", "B", "C"]


def test_dedupe_empty():
    assert dedupe_candidates([]) == []


def test_parse_relevance_score_plain_and_noisy():
    assert parse_relevance_score("8") == 8.0
    assert parse_relevance_score("Score: 10") == 10.0
    assert parse_relevance_score("SCORE: 4") == 4.0
    assert parse_relevance_score("I think it is a 7 out of 10") == 7.0
    assert parse_relevance_score("long reasoning ... SCORE: 9") == 9.0
    assert parse_relevance_score("no score here", default=0.0) == 0.0
    assert parse_relevance_score("", default=0.0) == 0.0


def test_extract_llm_text_prefers_content_then_reasoning():
    class Msg:
        def __init__(self, content=None, reasoning=None):
            self.content = content
            self.reasoning = reasoning

    assert extract_llm_text(Msg(content="9", reasoning="thinking")) == "9"
    assert extract_llm_text(Msg(content="", reasoning="Final score 6")) == "Final score 6"
    assert extract_llm_text(Msg(content=None, reasoning=None)) == ""
