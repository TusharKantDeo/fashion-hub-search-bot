---
name: rag-token-optimizer
description: Reduce token usage and cost of an LLM/RAG pipeline without losing accuracy — shorten classifier prompts, pick generation params per task, replace full-context stuffing with retrieval, trim context fields and chat history. Use when the user asks to cut tokens, cost or latency of prompts or a RAG app.
---

# Optimize tokens in a RAG pipeline

Reference: `notes/03-prompt-token-optimization.md`, `notes/02-llm-params-by-task.md`, `notes/08-cost-and-token-accounting.md`.

## Process
1. **Measure first.** Make every LLM call return `usage` (see `rag_kit/llm.py` `LLMResponse`) and record per step with `TokenLedger`. Find the most expensive *frequent* step.
2. **Build/locate a labelled test set** (10–30 queries) for any classifier you change. Accuracy must not drop.
3. Apply levers in order of impact:
   - Whole corpus in prompt → top-k retrieval (`notes/04`).
   - Strong model on helper calls → cheap model for routing/extraction.
   - Classifier prompts: one-line label definitions, `query -> label` format, keep only examples the model gets wrong without them, query last, `temperature=0`, `max_tokens` 1–10.
   - JSON extraction: allowed values once as minified JSON, allow omitting keys, validate output.
   - Context: only fields the answer uses; character budget.
   - History: raw user questions, last N turns.
   - Skip calls: rules before LLM, cache repeated queries.
4. **Re-measure** with real `usage.total_tokens` (not word counts) and re-run the test set. Report before/after tokens and accuracy per step.

## Watch out
- Reasoning models + tiny `max_tokens` = empty output. Disable reasoning for routers.
- Check `finish_reason == "length"` for truncated answers.
