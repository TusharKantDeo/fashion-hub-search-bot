# 07 — Phoenix tracing

## Idea
A RAG answer is 3–5 calls (router → task nature → metadata → search → answer). When cost or latency spikes, you need to see **which step** did it. Phoenix collects OpenTelemetry spans and shows a tree per request, with token counts, cost, retrieved documents and errors.

## Run Phoenix
```bash
docker compose up -d phoenix            # UI on http://localhost:6006
# or without Docker:
pip install arize-phoenix && phoenix serve
```

## Connect the app
```python
from rag_kit.tracing import setup_phoenix
setup_phoenix(project_name="fashion-hub", endpoint="http://localhost:6006/v1/traces")
```
That calls `phoenix.otel.register(..., auto_instrument=False, batch=True)`.
`auto_instrument=False` on purpose: we only trace what we choose (the course did the same to keep test/example calls out of the traces).

## Spans with OpenInference kinds
`rag_kit.tracing.span` uses plain OpenTelemetry and sets the attributes Phoenix understands, so the code works with **or without** Phoenix:
```python
with span("route_faq_or_product", kind="chain", input={"query": q}) as s:
    ...
```
| kind | used for |
|------|----------|
| `agent` | the whole answer (`rag_answer`) |
| `chain` | routing, metadata extraction, relaxation loop |
| `llm` | each model call — with `llm.token_count.*`, `llm.model_name` |
| `retriever` | vector search — with `retrieval.documents.N.document.*` |

`record_llm_usage(span, resp)` sets:
```
llm.token_count.prompt / completion / total, llm.model_name, llm.provider, output.value
```
These are what Phoenix uses for **cost**. Add model prices in Phoenix → Settings → Models (name pattern = model id, $ per 1M input/output tokens).

## What to look at in the UI
- Sort traces by total tokens → find expensive query types.
- Open a trace → compare `metadata_llm` tokens vs `answer_llm` tokens.
- Retriever spans → check if the right documents came back (most RAG "LLM errors" are retrieval errors).
- Filter by `status_code == 'ERROR'`.

## Pitfalls
- Spans exported in batch: in short scripts, the process may exit before export. `register(batch=True)` flushes on shutdown; in notebooks call `trace.get_tracer_provider().force_flush()` if needed.
- Don't put secrets or PII in `input.value`.
