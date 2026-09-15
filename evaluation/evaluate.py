"""
Resume-RAG evaluation CLI.

Usage (from repo root):
  python -m evaluation.evaluate
  python -m evaluation.evaluate --dense-only
  python -m evaluation.evaluate --reranker
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow `python -m evaluation.evaluate` from repo root without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.metrics import (  # noqa: E402
    dedupe_candidates,
    mean_metric,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from project.config import configure_logging, settings  # noqa: E402
from project.embedding import EmbeddingManager  # noqa: E402
from project.reranker import LLMReranker  # noqa: E402
from project.vector_store import VectorStoreManager  # noqa: E402

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = EVAL_DIR / "queries.json"
DEFAULT_RESULTS_JSON = EVAL_DIR / "results.json"
DEFAULT_RESULTS_CSV = EVAL_DIR / "results.csv"

# Retrieve enough chunks so candidate-level@10 is usually achievable after dedupe.
DEFAULT_DENSE_CHUNK_POOL = 30
DEFAULT_RERANK_POOL = 15
DEFAULT_RERANK_TOP_K = 4


def load_queries(path: Path = DEFAULT_QUERIES) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries", data if isinstance(data, list) else [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def _candidate_ranking_from_hits(hits: list[dict[str, Any]]) -> list[str]:
    return dedupe_candidates(hits)


def _per_query_metrics(
    relevant: list[str],
    ranked_candidates: list[str],
) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(relevant, ranked_candidates, 1),
        "recall@4": recall_at_k(relevant, ranked_candidates, 4),
        "recall@10": recall_at_k(relevant, ranked_candidates, 10),
        "precision@4": precision_at_k(relevant, ranked_candidates, 4),
        "mrr": mrr(relevant, ranked_candidates),
        "ndcg@4": ndcg_at_k(relevant, ranked_candidates, 4),
    }


def evaluate_dense_retrieval(
    queries: list[dict[str, Any]],
    embedding_manager: EmbeddingManager,
    vector_store: VectorStoreManager,
    chunk_pool: int = DEFAULT_DENSE_CHUNK_POOL,
) -> list[dict[str, Any]]:
    """
    Dense retrieval only (no Groq).

    Retrieves ``chunk_pool`` chunks, dedupes by candidate_id (first-seen order),
    then computes candidate-level metrics.
    """
    results: list[dict[str, Any]] = []
    for item in queries:
        qid = item["id"]
        query = item["query"]
        relevant = list(item.get("relevant_candidates") or [])
        logger.info("Dense eval %s: %s", qid, query)

        warnings: list[str] = []
        try:
            query_embedding = embedding_manager.generate_embeddings([query])
            hits = vector_store.similarity_search(query_embedding, top_k=chunk_pool)
            ranked = _candidate_ranking_from_hits(hits)
            metrics = _per_query_metrics(relevant, ranked)
        except Exception as exc:
            warning = f"dense retrieval failed: {exc}"
            logger.error("%s (%s)", warning, qid)
            warnings.append(warning)
            ranked = []
            hits = []
            metrics = _per_query_metrics(relevant, ranked)

        results.append(
            {
                "id": qid,
                "query": query,
                "difficulty": item.get("difficulty"),
                "relevant_candidates": relevant,
                "dense_retrieved_candidates": ranked[:10],
                "dense_chunk_count": len(hits),
                "dense_metrics": metrics,
                "warnings": warnings,
            }
        )
    return results


def evaluate_reranked_retrieval(
    queries: list[dict[str, Any]],
    embedding_manager: EmbeddingManager,
    vector_store: VectorStoreManager,
    reranker: LLMReranker,
    retrieve_pool: int = DEFAULT_RERANK_POOL,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[dict[str, Any]]:
    """
    Dense overfetch -> LLM rerank chunks -> candidate dedupe -> metrics.

    Reranker only reorders the retrieved pool; it does not search Chroma.
    """
    results: list[dict[str, Any]] = []
    for item in queries:
        qid = item["id"]
        query = item["query"]
        relevant = list(item.get("relevant_candidates") or [])
        logger.info("Rerank eval %s: %s", qid, query)

        warnings: list[str] = []
        try:
            query_embedding = embedding_manager.generate_embeddings([query])
            pool_hits = vector_store.similarity_search(
                query_embedding, top_k=retrieve_pool
            )
            # Score all pool hits; request full pool then take candidate top_k after dedupe
            reranked_hits = reranker.rerank(
                query, pool_hits, top_k=len(pool_hits) or rerank_top_k
            )
            for hit in reranked_hits:
                if hit.get("reranker_warning"):
                    warnings.append(str(hit["reranker_warning"]))
            ranked = _candidate_ranking_from_hits(reranked_hits)
            metrics = _per_query_metrics(relevant, ranked)
        except Exception as exc:
            warning = f"reranked retrieval failed: {exc}"
            logger.error("%s (%s)", warning, qid)
            warnings.append(warning)
            ranked = []
            metrics = _per_query_metrics(relevant, ranked)

        results.append(
            {
                "id": qid,
                "query": query,
                "difficulty": item.get("difficulty"),
                "relevant_candidates": relevant,
                "reranked_candidates": ranked[:rerank_top_k],
                "reranked_metrics": {
                    "recall@4": metrics["recall@4"],
                    "precision@4": metrics["precision@4"],
                    "mrr": metrics["mrr"],
                    "ndcg@4": metrics["ndcg@4"],
                },
                "warnings": warnings,
            }
        )
    return results


def _aggregate(metric_maps: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return {
        key: mean_metric([m.get(key, 0.0) for m in metric_maps]) for key in keys
    }


def merge_results(
    dense_rows: list[dict[str, Any]],
    rerank_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_id = {row["id"]: dict(row) for row in dense_rows}
    if rerank_rows:
        for row in rerank_rows:
            target = by_id.setdefault(row["id"], {"id": row["id"], "query": row["query"]})
            target["reranked_candidates"] = row.get("reranked_candidates", [])
            target["reranked_metrics"] = row.get("reranked_metrics", {})
            warnings = list(target.get("warnings") or [])
            warnings.extend(row.get("warnings") or [])
            target["warnings"] = warnings
            if "relevant_candidates" not in target:
                target["relevant_candidates"] = row.get("relevant_candidates", [])
            if "difficulty" not in target:
                target["difficulty"] = row.get("difficulty")
    return [by_id[qid] for qid in by_id]


def format_summary(
    dense_agg: dict[str, float],
    rerank_agg: dict[str, float] | None,
    n_queries: int,
) -> str:
    lines = [
        "==============================",
        "Resume RAG Evaluation",
        "==============================",
        "",
        f"Queries evaluated: {n_queries}",
        "",
        "Dense Retrieval",
        "---------------",
        f"Recall@1 : {dense_agg.get('recall@1', 0):.4f}",
        f"Recall@4 : {dense_agg.get('recall@4', 0):.4f}",
        f"Recall@10: {dense_agg.get('recall@10', 0):.4f}",
        f"Precision@4: {dense_agg.get('precision@4', 0):.4f}",
        f"MRR      : {dense_agg.get('mrr', 0):.4f}",
        f"nDCG@4   : {dense_agg.get('ndcg@4', 0):.4f}",
    ]
    if rerank_agg is not None:
        lines.extend(
            [
                "",
                "LLM Reranked",
                "------------",
                f"Recall@4 : {rerank_agg.get('recall@4', 0):.4f}",
                f"Precision@4: {rerank_agg.get('precision@4', 0):.4f}",
                f"MRR      : {rerank_agg.get('mrr', 0):.4f}",
                f"nDCG@4   : {rerank_agg.get('ndcg@4', 0):.4f}",
                "",
                "Improvement",
                "-----------",
                f"MRR       : {rerank_agg.get('mrr', 0) - dense_agg.get('mrr', 0):+.4f}",
                f"nDCG@4    : {rerank_agg.get('ndcg@4', 0) - dense_agg.get('ndcg@4', 0):+.4f}",
                f"Precision : {rerank_agg.get('precision@4', 0) - dense_agg.get('precision@4', 0):+.4f}",
            ]
        )
    return "\n".join(lines)


def save_results(
    rows: list[dict[str, Any]],
    dense_agg: dict[str, float],
    rerank_agg: dict[str, float] | None,
    json_path: Path = DEFAULT_RESULTS_JSON,
    csv_path: Path = DEFAULT_RESULTS_CSV,
) -> None:
    payload = {
        "summary": {
            "queries_evaluated": len(rows),
            "dense": dense_agg,
            "reranked": rerank_agg,
        },
        "queries": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "id",
        "difficulty",
        "query",
        "relevant_candidates",
        "dense_retrieved_candidates",
        "reranked_candidates",
        "dense_recall@1",
        "dense_recall@4",
        "dense_recall@10",
        "dense_precision@4",
        "dense_mrr",
        "dense_ndcg@4",
        "rerank_recall@4",
        "rerank_precision@4",
        "rerank_mrr",
        "rerank_ndcg@4",
        "warnings",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            dm = row.get("dense_metrics") or {}
            rm = row.get("reranked_metrics") or {}
            writer.writerow(
                {
                    "id": row.get("id"),
                    "difficulty": row.get("difficulty"),
                    "query": row.get("query"),
                    "relevant_candidates": "|".join(row.get("relevant_candidates") or []),
                    "dense_retrieved_candidates": "|".join(
                        row.get("dense_retrieved_candidates") or []
                    ),
                    "reranked_candidates": "|".join(row.get("reranked_candidates") or []),
                    "dense_recall@1": dm.get("recall@1"),
                    "dense_recall@4": dm.get("recall@4"),
                    "dense_recall@10": dm.get("recall@10"),
                    "dense_precision@4": dm.get("precision@4"),
                    "dense_mrr": dm.get("mrr"),
                    "dense_ndcg@4": dm.get("ndcg@4"),
                    "rerank_recall@4": rm.get("recall@4"),
                    "rerank_precision@4": rm.get("precision@4"),
                    "rerank_mrr": rm.get("mrr"),
                    "rerank_ndcg@4": rm.get("ndcg@4"),
                    "warnings": " || ".join(row.get("warnings") or []),
                }
            )
    logger.info("Wrote %s and %s", json_path, csv_path)


def run_evaluation(
    *,
    dense_only: bool = False,
    reranker_only: bool = False,
    queries_path: Path = DEFAULT_QUERIES,
) -> dict[str, Any]:
    configure_logging("evaluation")
    queries = load_queries(queries_path)
    logger.info("Loaded %s evaluation queries from %s", len(queries), queries_path)

    embedding_manager = EmbeddingManager()
    vector_store = VectorStoreManager()
    doc_count = vector_store.get_document_count()
    if doc_count <= 0:
        raise RuntimeError(
            "Vector store is empty. Build it first via workspace/baseline_rag.ipynb "
            "Parts 1–4 (or an equivalent ingest script)."
        )
    logger.info("Vector store collection has %s chunks", doc_count)

    run_dense = not reranker_only
    run_rerank = not dense_only

    dense_rows: list[dict[str, Any]] = []
    rerank_rows: list[dict[str, Any]] | None = None

    if run_dense:
        dense_rows = evaluate_dense_retrieval(queries, embedding_manager, vector_store)
    else:
        # Still include query metadata when only reranking
        dense_rows = [
            {
                "id": q["id"],
                "query": q["query"],
                "difficulty": q.get("difficulty"),
                "relevant_candidates": list(q.get("relevant_candidates") or []),
                "dense_retrieved_candidates": [],
                "dense_metrics": {},
                "warnings": [],
            }
            for q in queries
        ]

    if run_rerank:
        if not settings.groq_api_key:
            message = (
                "GROQ_API_KEY is not set. Dense metrics can still run with --dense-only. "
                "To evaluate the LLM reranker, set GROQ_API_KEY in .env."
            )
            logger.error(message)
            if dense_only is False and reranker_only:
                raise SystemExit(message)
            print(message)
            run_rerank = False
        else:
            reranker = LLMReranker()
            rerank_rows = evaluate_reranked_retrieval(
                queries, embedding_manager, vector_store, reranker
            )

    merged = merge_results(dense_rows, rerank_rows if run_rerank else None)

    dense_agg = _aggregate(
        [r.get("dense_metrics") or {} for r in merged if r.get("dense_metrics")],
        ["recall@1", "recall@4", "recall@10", "precision@4", "mrr", "ndcg@4"],
    )
    rerank_agg = None
    if run_rerank and rerank_rows is not None:
        rerank_agg = _aggregate(
            [r.get("reranked_metrics") or {} for r in merged if r.get("reranked_metrics")],
            ["recall@4", "precision@4", "mrr", "ndcg@4"],
        )

    summary_text = format_summary(dense_agg, rerank_agg, len(queries))
    print(summary_text)
    save_results(merged, dense_agg, rerank_agg)
    return {"dense": dense_agg, "reranked": rerank_agg, "rows": merged}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Resume-RAG retrieval.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dense-only",
        action="store_true",
        help="Run dense retrieval metrics only (no Groq).",
    )
    mode.add_argument(
        "--reranker",
        action="store_true",
        help="Run LLM reranker evaluation (requires GROQ_API_KEY). Dense still runs unless you only need rerank rows.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help="Path to queries.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # Default: both. --reranker means include reranker (and dense). --dense-only skips reranker.
    run_evaluation(
        dense_only=bool(args.dense_only),
        reranker_only=False,
        queries_path=args.queries,
    )
    # If user passed --reranker explicitly with missing key, run_evaluation already messages.
    if args.reranker and not settings.groq_api_key:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
