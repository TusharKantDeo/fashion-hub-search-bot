# 08 — Cost & token accounting

## Idea
Cost of one user message = **sum of every LLM call it triggered**, not just the final answer.

```
route (cheap)        ~90 tokens
task_nature (cheap)  ~70 tokens
metadata (cheap)     ~400 tokens (≈1500 with the naive prompt)
answer (strong)      ~1200 tokens
-----------------------------------
total                ~1760 tokens
```

## How (rag_kit)
```python
ans = bot.chat("Black party dress under $100")
ans.ledger.by_step()     # {'route': 92, 'task_nature': 71, 'metadata': 388, 'answer': 1210}
ans.ledger.total_tokens
ans.ledger.cost_usd({"model-id": (in_per_million, out_per_million)})
```
Every `LLMClient.complete` returns an `LLMResponse` with `prompt_tokens`, `completion_tokens`, `total_tokens`, `model`. Handlers add their calls to the ledger.

## Levers, by impact
1. **Don't send what you don't need** — retrieval top-k instead of whole corpus ([04](04-retrieve-dont-stuff.md)), only needed fields.
2. **Model tiering** — cheap model for routing/extraction, strong model for answers.
3. **Short prompts** for frequent calls ([03](03-prompt-token-optimization.md)).
4. **Cap outputs** with `max_tokens`.
5. **Skip calls** — direct RAG skips metadata extraction; rules can skip routing.
6. **History window** — last N turns, raw questions only.
7. **Caching** — provider prompt caching for a fixed system prompt; app-level cache for repeated queries.

## Course numbers to remember
- FAQ full-context ≈ 1220 tokens → top-5 retrieval < 500.
- Router prompt target < 180 tokens; task-nature < 150.
- Naive metadata extraction ≈ 1500 tokens per product query; direct RAG 0.
