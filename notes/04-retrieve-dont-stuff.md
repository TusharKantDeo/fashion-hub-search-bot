# 04 — Retrieve, don't stuff (the "R" in RAG)

## Idea
The first FAQ version put **the entire FAQ** in every prompt (~1220 tokens per question). The improved version runs a semantic search and puts only the **top-5** FAQ entries in the prompt (<500 tokens) — same answer quality.

## Why it works
- Most of a corpus is irrelevant to any one question.
- Fewer, more relevant documents also *improve* answers: less distraction, less "lost in the middle".
- Cost and latency scale with prompt length; retrieval scales with log(N) in the vector index.

## How (rag_kit)
```python
docs = faq_store.search(query, limit=5)                        # semantic search
context = format_records(order_for_prompt(docs), {"Q": "question", "A": "answer"})
prompt = grounded_answer_prompt(query, context,
                                role="You answer customer questions for a clothing store.",
                                rules=["FAQ items are ordered least to most relevant; the last is the best match."],
                                context_tag="FAQ")
```

## Details that matter
- **Order:** search returns best-first. Reverse it so the best match sits **right before the question** (`order_for_prompt`) and tell the model about the ordering.
- **k:** 3–5 for FAQ-like short docs; ~20 for product catalogs where the answer model selects a few.
- **Tag the context** (`<FAQ>...</FAQ>`) so the model knows where evidence ends.
- **Grounding instruction:** "Answer using only the context". Add "If the answer is not there, say so".
- Add a **character budget** (`format_records(max_chars=...)`) as a safety net for long documents.

## When stuffing is fine
Corpus is tiny (< ~2k tokens), changes constantly, or questions need *global* knowledge (e.g. "summarize all policies"). Prompt caching can make this cheaper too.
