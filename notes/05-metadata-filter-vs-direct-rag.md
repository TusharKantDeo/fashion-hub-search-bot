# 05 — Metadata filtering vs direct RAG for product search

Two ways to find products for "blue T-shirts for men under $50":

| | Direct RAG | Metadata filter (self-query) |
|---|---|---|
| Steps | embed query → vector search | LLM → JSON filters → *filtered* vector search |
| Extra LLM tokens | **0** | ~1500 naive prompt; ~300–500 compact |
| Latency | lower | +1 LLM call |
| Hard constraints (gender, colour, price) | *soft* — embedding may ignore them | **enforced exactly** |
| Failure mode | returns women's / red / expensive items that are "semantically close" | over-filtering → zero results (needs relaxation) |
| Observed quality in the course | good enough for vague queries | **better**, especially with explicit constraints |

## Why the filter version is better
Embeddings capture *topic*, not *logic*. "under $50" or "for men" barely moves a sentence vector, and numbers are handled poorly. A filter is a database predicate: it's either true or not.

## Making the filter path cheaper (implemented in `rag_kit/metadata_filter.py`)
- Compact prompt: allowed values once as minified JSON + 1 line of rules.
- Let the model **omit** unknown keys instead of generating `["Any"]` for each.
- `temperature=0`, `max_tokens≈300`, cheap router model.
- **Validate** output against allowed values (`sanitize`) — never trust raw LLM JSON.
- Robust parsing (`parse_json_loose`): handles ```json fences, prose around JSON, bare `inf`.

## Relaxation (don't return nothing)
If filtered results < `min_results`, drop filters from least to most important and search again:
```python
RELAX_ORDER = ["baseColour", "season", "usage", "articleType", "masterCategory", "gender"]
docs, active = search_with_relaxation(store, query, filters, RELAX_ORDER, limit=20, min_results=5)
```
Filters are kept in a `{property: Filter}` dict, so dropping by name is trivial.
(The course version had `masterCategory` twice in its importance list; keyed dicts avoid that.)

## Best of both (hybrid ideas)
- **Rule-first extraction:** regex for prices/colours/gender, LLM only if rules find nothing.
- **Use filters only when the query has constraints** (a tiny router: "does this query mention attributes?").
- **Cache** extracted filters for repeated queries.
- Weaviate **hybrid search** (BM25 + vector) helps exact words like brand names.
- Rerank the top-20 with a cross-encoder.

## Measure it yourself
`python scripts/compare_strategies.py` prints, per query, retrieval tokens and top-5 precision against a hard-constraint check for both strategies.
