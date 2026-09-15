# Resume-RAG

Recruiting-oriented **Retrieval-Augmented Generation** over a small resume corpus.

Ask a recruiter question → retrieve relevant resume chunks from ChromaDB → (optionally) LLM-rerank → generate an evidence-grounded candidate recommendation with Groq.

> **Privacy:** The PDFs under `data/resumes/` may contain personal contact information (PII). Do not upload them publicly or commit them to a shared remote without authorization.

---

## Architecture

```text
PDF resumes
  → LangChain load + chunk (~800 chars, overlap 100)
  → SentenceTransformer embeddings (all-MiniLM-L6-v2)
  → ChromaDB (cosine) collection `resume_baseline`
  → dense similarity search
  → optional LLM rerank (Groq scores chunks 0–10)
  → Groq generation (openai/gpt-oss-20b) grounded in retrieved evidence
```

| Component | Location |
|-----------|----------|
| Config / logging | `project/config.py` |
| Load + chunk | `project/data_loader.py` |
| Embeddings | `project/embedding.py` |
| Vector store | `project/vector_store.py` |
| LLM reranker | `project/reranker.py` |
| Baseline demo | `workspace/baseline_rag.ipynb` (Parts 0–8) |
| Evaluation | `evaluation/` |

---

## Current capabilities

- Index ~25 resume PDFs into a persistent Chroma store
- Dense retrieval baseline (Parts 5–7 in the notebook)
- Experimental **LLM reranking** (Part 8 concept; reusable in `project/reranker.py`)
- Offline **evaluation layer** comparing dense vs reranked retrieval (candidate-level metrics)

Not included yet: web UI, API, auth, hybrid BM25+dense, or cross-encoder reranking.

---

## Setup

```bash
cd d:\Resume-Rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GROQ_API_KEY=...
```

---

## Build / use the vector store

1. Open `workspace/baseline_rag.ipynb`
2. Select the project `venv` kernel
3. **Restart kernel**
4. Run Parts **0 → 4** (load → chunk → embed → Chroma ingest)

The store persists under `data/vector_store/`. Re-run Part 4 after changing chunking settings.

---

## Run the notebook baseline

Continue Parts **5–7** for retrieval + Groq recommendation.

Part **8** documents the reranking experiment.

Part **9** is a **hands-on evaluation walkthrough** (toy metrics → one real query → full dense eval → optional rerank). Prefer that for learning; use the CLI below for scripts/CI.

---

## Evaluation

Dataset: `evaluation/queries.json` (grounded in the actual resumes).

```bash
# Dense retrieval only (no Groq required)
python -m evaluation.evaluate --dense-only

# Dense + LLM reranker (requires GROQ_API_KEY)
python -m evaluation.evaluate

# Same as default, explicit reranker flag
python -m evaluation.evaluate --reranker
```

Outputs:

- Console metric summary
- `evaluation/results.json`
- `evaluation/results.csv`

### Metrics (candidate-level)

Chunks are deduplicated by `candidate_id` (first occurrence keeps the best rank) before scoring.

| Metric | Meaning |
|--------|---------|
| Recall@K | Fraction of labeled relevant candidates appearing in the top-K unique candidates |
| Precision@4 | Relevant candidates in top-4 / 4 |
| MRR | Reciprocal rank of the first relevant candidate |
| nDCG@4 | Ranking quality of the top-4 (binary relevance in the current dataset) |

**Dense vs rerank:** Dense search builds the candidate pool; the LLM reranker only **reorders** an overfetched pool (default top-15 chunks → new top-4). It cannot surface a resume that was never retrieved.

---

## Tests

```bash
pip install pytest
pytest -q
```

Metric / parsing tests do **not** require Groq or Chroma.

---

## Limitations

- Gold labels are human judgments from resume text; they are not exhaustive
- LLM-as-judge reranking is slower and non-deterministic vs a dedicated cross-encoder
- GPT-OSS responses may put text in `reasoning` instead of `content` (handled in code)
- No automated groundedness check on generated recommendations yet

---

## Roadmap (suggested)

1. Expand / adjudicate the eval set
2. Optional cross-encoder reranker compared on the same metrics
3. Hybrid retrieval (BM25 + dense)
4. Thin API or Streamlit UI with citations
