# 09 — Checklist: build a new RAG system with rag_kit

1. **Copy** `rag_kit/` into the new project (or `pip install -e` this repo). Copy `docker-compose.yml` and `.env.example`.
2. **Start infra:** `docker compose up -d` (or `WEAVIATE_MODE=embedded`).
3. **Write a `schema.py`** for your domain (use `fashion_hub/schema.py` as the template):
   - collection names
   - `text_fn(record)` → what gets embedded
   - Weaviate `Property` list (FIELD tokenization for filterable categoricals)
   - context fields for the prompt (only what answers need)
   - filter keys + relaxation order (if using metadata filters)
   - router instruction + 2–4 hard examples
4. **Ingest:** `WeaviateStore.create(props)` then `.ingest(records, id_key=...)`.
5. **Decide routes.** What kinds of questions exist? One handler per route. Each handler: retrieve → `format_records` → `grounded_answer_prompt` → `PreparedCall(prompt, params_for(task))`.
6. **Wire** `RAGApp(llm, router, handlers, fallback, answer_model=...)`.
7. **Trace:** `setup_phoenix(project)`; open http://localhost:6006.
8. **Build a test set:** 20–50 real queries with expected route and a check on the retrieved docs.
9. **Optimize in this order:** retrieval quality → routing accuracy → prompt tokens → model tiering.
10. **Write offline tests** with `FakeLLM` and a fake store (see `tests/`).

## Decision guide
- Corpus < 2k tokens? → maybe skip retrieval, stuff it.
- Queries contain hard constraints (price, size, category)? → metadata filters + relaxation.
- Queries are vague/semantic? → direct RAG is enough.
- Exact names/IDs/brands matter? → hybrid search (BM25 + vectors).
- Many near-duplicates in results? → rerank or MMR.
