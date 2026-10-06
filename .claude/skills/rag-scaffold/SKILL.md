---
name: rag-scaffold
description: Scaffold a new RAG system (routing + retrieval + grounded answers + tracing) for a new domain using the rag_kit module from this repo. Use when the user wants to build a RAG bot, search assistant, FAQ bot, or "chat with my data" project.
---

# Scaffold a RAG system with rag_kit

Follow `notes/09-build-any-rag-checklist.md`. Concretely:

1. **Gather the domain** (ask only if not inferable): data records + their fields, kinds of questions (routes), which fields are hard constraints (filters), LLM provider.
2. **Copy the kit:** `rag_kit/`, `docker-compose.yml`, `.env.example`, `pyproject.toml` deps into the target project. Do not modify `rag_kit` unless something is genuinely missing — domain code goes in a separate package.
3. **Create `<app>/schema.py`** modeled on `fashion_hub/schema.py`:
   - collection names, `text_fn` per collection
   - Weaviate `Property` list; `Tokenization.FIELD` for filterable categorical fields
   - `*_CONTEXT_FIELDS` with only the fields answers need
   - `FilterSchema` (categorical allowed values from data, numeric fields, relax order least→most important) if queries have hard constraints
   - router instruction (define each label in one sentence) + 2–4 hard examples
4. **Create `<app>/bot.py`** modeled on `fashion_hub/bot.py`: one handler per route returning `PreparedCall`; wire `RAGApp` with a fallback handler; cheap `router_model`, stronger `answer_model`.
5. **Ingest script** using `WeaviateStore.create` + `.ingest(records, id_key=...)`.
6. **Tests** with `FakeLLM` + a fake store (copy `tests/conftest.py`), covering: routing, which store each route hits, ledger steps, params per task.
7. **Run** `pytest`, then `docker compose up -d`, ingest, and try the CLI. Open Phoenix at http://localhost:6006.

Defaults: routing `PRESETS["classification"]`, extraction `PRESETS["extraction"]`, FAQ-like top-k=5, catalog top-k=20, most-relevant-last ordering.
