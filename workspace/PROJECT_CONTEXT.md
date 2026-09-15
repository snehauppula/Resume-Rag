# Resume-Rag — Project Context (for discussion / next steps)

**Date of this snapshot:** 2026-09-14  
**Repo path:** `d:\Resume-Rag`  
**Purpose:** Paste this into ChatGPT (or any advisor) to discuss what to build next. Do not paste `.env` or API keys.

---

## 1. What this project is

A **resume RAG (Retrieval-Augmented Generation)** demo for recruiting:

1. Index 25 resume PDFs into a vector database.
2. Ask a recruiter-style natural-language question.
3. Retrieve the most similar resume chunks.
4. Ask an LLM to recommend candidates **only from retrieved evidence** (no invention).

It is currently a **notebook-driven baseline + one improvement experiment (LLM re-ranking)**. There is no web app, API server, auth, or evaluation harness yet.

---

## 2. High-level architecture

```
data/resumes/*.pdf (25 PDFs)
        |
        v
[load pages] -> [chunk ~800 / overlap 100] -> [embed all-MiniLM-L6-v2]
        |
        v
ChromaDB (cosine)  collection: resume_baseline
  persist: data/vector_store/
        |
        v
similarity_search(top_k)  -->  retrieved chunks + metadata
        |
        +--> (baseline) Groq LLM recommendation  [Part 7]
        |
        +--> (experiment) overfetch top-15 -> Groq score 0-10 -> new top-4  [Part 8]
```

**LLM provider:** Groq  
**Current model ID:** `openai/gpt-oss-20b`  
(Older model `llama-3.1-8b-instant` was decommissioned ~2026-08-16 and must not be used.)

**Secrets:** `GROQ_API_KEY` in root `.env` (gitignored). Template: `.env.example`.

---

## 3. Repo layout

| Path | Role |
|------|------|
| `project/config.py` | Settings, paths, Groq model, logging helpers |
| `project/data_loader.py` | PDF load + chunking (LangChain) |
| `project/embedding.py` | `EmbeddingManager` (SentenceTransformer) |
| `project/vector_store.py` | `VectorStoreManager` (Chroma persist + search) |
| `workspace/baseline_rag.ipynb` | End-to-end demo notebook (Parts 0–8) |
| `data/resumes/` | 25 resume PDFs |
| `data/vector_store/` | Persisted Chroma DB (~706 chunks when built) |
| `logs/baseline_rag.log` | File logs when Part 0 runs |
| `requirements.txt` | Pinned deps from working venv |
| `.gitignore` | Ignores `.env`, `venv/`, `logs/`, vector store, etc. |
| `README.md` | **Missing** |

No tests, no CLI entrypoint, no FastAPI/Streamlit app.

---

## 4. Notebook map (`workspace/baseline_rag.ipynb`)

**19 cells total.** Part 8 is at the **bottom** (cells 16–18). If you don’t see it in the UI, scroll past Part 7 or reopen the file from disk.

| Part | What it does | Status |
|------|----------------|--------|
| **0 Setup** | Add repo to `sys.path`, load `.env`, reload `project.config`, configure logging | Code ready; may need kernel restart if imports look stale |
| **1 Load** | `load_documents()` → ~142 pages from 25 PDFs | Implemented; previously ran OK |
| **2 Chunk** | `split_documents()` → ~706 chunks | Implemented |
| **3 Embed** | `EmbeddingManager` + encode all chunk texts | Implemented |
| **4 Chroma** | Reset collection, `add_documents`, persist | Implemented; store exists on disk |
| **5 Retrieve** | Sample Java/AWS query → top-4 cosine hits | Implemented |
| **6 Inspect** | Print evidence (candidate_id, page, distance, text) | Implemented |
| **7 Generate** | Groq recruiter recommendation from evidence only | Implemented; uses `openai/gpt-oss-20b` |
| **8 Re-rank** | Concept + code + written comparison results | **Present in notebook**; LLM-as-reranker experiment |

### Part 8 details (important)

**Idea:** Vector top-4 is not always task-relevant. So:

1. Retrieve **top-15** (overfetch).
2. Ask Groq to score each chunk **0–10** for the query.
3. Sort by score; keep new **top-4**.

**Test query used:**  
“Which candidate has hands-on experience with automated testing frameworks like Selenium?”

**Documented outcome (from earlier run notes in notebook):**  
Naive top-4 put a less relevant BSA chunk first; after re-rank, `gautami_qa_mobile_testing` (Selenium/QA) rose, including a chunk that was originally rank 5 (outside naive top-4).

**Design rule recorded in notebook:**  
Re-ranking can only reorder what was retrieved; it cannot invent chunks never fetched.

**Caveats / incomplete engineering:**
- Re-ranking logic lives **only in the notebook**, not in `project/`.
- Depends on `client` / `settings` / `embedding_manager` / `vector_store` from earlier cells.
- Groq scoring can return empty `content` for GPT-OSS (reasoning models); Part 7 has a fallback, Part 8 score parsing is fragile.
- No dedicated cross-encoder; this is **LLM-as-judge reranking** (slower, costlier than a small cross-encoder).
- Part 8 cells currently have **no saved execution outputs** in the notebook JSON (notes markdown remains).

---

## 5. Key config defaults (`project/config.py`)

- `chunk_size=800`, `chunk_overlap=100`
- `embedding_model_name=all-MiniLM-L6-v2`
- `collection_name=resume_baseline`
- `default_top_k=4`
- `groq_model_name=openai/gpt-oss-20b`
- `groq_temperature=0.1`, `groq_max_tokens=1024`

---

## 6. What already works vs what’s missing

### Done / mostly done
- End-to-end baseline RAG in notebook + shared Python modules
- Persisted Chroma index for resumes
- Evidence-grounded generation prompt
- First re-ranking experiment + qualitative comparison notes
- Basic reproducibility scaffolding (`requirements.txt`, `.gitignore`, `.env.example`)
- Logging to console + `logs/baseline_rag.log`

### Not done / weak
- No README / onboarding docs
- No automated evaluation (precision@k, nDCG, groundedness checks)
- No hybrid search (BM25 + dense)
- No real cross-encoder reranker module
- No packaging of Part 8 into `project/`
- No product UI / API
- Jupyter sometimes showed stale imports (kernel cache); setup uses `importlib.reload`
- Resume PDFs contain PII — treat carefully if sharing/publishing

---

## 7. Likely next-step options (for discussion)

Pick a direction; don’t do all at once:

1. **Stabilize baseline** — clean notebook runbook, README, “restart kernel & run all” checklist.
2. **Productize retrieval** — move Part 8 into `project/reranker.py`; optional cross-encoder instead of Groq scoring.
3. **Evaluation** — fixed query set + gold candidates; compare baseline top-4 vs reranked top-4.
4. **Quality upgrades** — better chunking, metadata filters (role/skills), hybrid retrieval.
5. **App layer** — Streamlit/FastAPI: upload JD → recommend candidates with citations.
6. **Ops** — rotate Groq key if ever exposed; decide whether resumes/vector DB stay local-only.

---

## 8. How to run again (quick)

1. Activate `venv`
2. Open `workspace/baseline_rag.ipynb`
3. Select kernel = project `venv`
4. **Restart Kernel**
5. Run Part 0 → Part 7 (baseline)
6. Scroll to bottom for Part 8 (needs Parts 0–5 objects in memory; Part 7’s Groq `client` helps Part 8)

---

## 9. One-line status

**Baseline resume RAG is built and indexed; generation uses Groq GPT-OSS; Part 8 LLM re-ranking exists at the end of the notebook as an experiment, not yet a clean library module — next step should be chosen deliberately (stabilize, evaluate, or productize).**


