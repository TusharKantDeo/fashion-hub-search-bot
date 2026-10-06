# 02 — LLM parameters by task nature

## Idea
Sampling parameters are not "set once". Choose them **per call** based on what the call does.

| Task | temperature | top_p | max_tokens | Why |
|------|-------------|-------|------------|-----|
| classification / routing | 0 | 1 | 1–10 | One deterministic label. Long output = waste + parse risk. |
| extraction (JSON filters) | 0 | 1 | ~300 | Deterministic, just big enough for the JSON. |
| technical answer (lists, prices, facts) | 0.3 | 0.7 | ~500 | Stay close to the context, little invention. |
| creative answer (styling, ideas) | 1.0 | 0.9 | ~700 | Variety and combinations are the point. |

These are the `PRESETS` in `rag_kit/params.py`; `params_for(label)` picks one with a safe default.

## What the knobs do
- **temperature** scales the logits before sampling. 0 → always the most likely token (near-deterministic). Higher → flatter distribution, more variety, more risk.
- **top_p (nucleus)** samples only from the smallest set of tokens whose probabilities add up to p. Lower → cut off the long tail of unlikely tokens.
- **max_tokens** caps the *output*. It's a cost guard and a format guard (a label can't turn into an essay).
- Tune temperature *or* top_p mostly — changing both strongly at once is hard to reason about.

## How (rag_kit)
```python
nature = task_router.classify(query)            # "creative" | "technical"
params = params_for(nature.label)               # GenParams(temperature, top_p, max_tokens)
llm.complete(messages, params, model=answer_model)
```

## Pitfalls
- Course code had `if technical: ... if creative: ... else: fallback` — the second `if/else` overwrote the technical params with the fallback. A dict lookup (`PRESETS.get`) avoids this class of bug. There's a regression test in `tests/test_params.py`.
- temperature 0 is *not* a guarantee of identical outputs across providers/batches, just much closer.
- If `max_tokens` is too small for an answer, the answer is silently truncated — check `finish_reason == "length"` in production.
