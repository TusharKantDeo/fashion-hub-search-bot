---
name: phoenix-tracing
description: Add Arize Phoenix / OpenTelemetry tracing to an LLM or RAG app with OpenInference span kinds, token counts and retrieved documents, so cost and latency can be inspected per step. Use when the user wants observability, tracing, cost monitoring or debugging of LLM calls.
---

# Phoenix tracing

Reference: `notes/07-phoenix-tracing.md`, code in `rag_kit/tracing.py`.

1. **Run Phoenix:** `docker compose up -d phoenix` (UI http://localhost:6006) or `pip install arize-phoenix && phoenix serve`.
2. **Install client:** `pip install arize-phoenix-otel opentelemetry-api`.
3. **Register once at startup:** `rag_kit.tracing.setup_phoenix(project, "http://localhost:6006/v1/traces")` (wraps `phoenix.otel.register(auto_instrument=False, batch=True)`). Use `auto_instrument=True` only if you want every OpenAI/LangChain call traced automatically.
4. **Instrument steps** with `with span(name, kind=..., input=...) as s:`
   - `agent` for the whole request, `chain` for routing/extraction, `llm` for model calls, `retriever` for searches.
   - On LLM spans call `record_llm_usage(s, resp, provider)` (sets `llm.token_count.*`, `llm.model_name`).
   - On retriever spans call `record_retrieval(s, docs)`.
5. **Cost:** in Phoenix → Settings → Models add each model id with $/1M input & output tokens.
6. Verify by running one request and opening the trace tree; check token counts appear on `llm` spans.

Never put secrets/PII in span inputs. In short-lived scripts, flush the tracer provider before exit.
