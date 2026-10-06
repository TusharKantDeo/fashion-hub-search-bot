# 03 — Prompt token optimization

## Idea
Every token in a prompt is paid for **on every request**. The routing prompts in the course started at ~200+ tokens; the goal was **<180** (FAQ/Product) and **<150** (creative/technical) *without losing accuracy*.

## Recipe
1. **Define the labels instead of showing them.** One sentence like "FAQ = policies, returns, shipping, sizes, contact, promotions" replaces 4–5 examples.
2. **Keep only the hard examples.** Run the prompt with zero examples on your test set and add an example only for the queries it gets wrong (or a harder version of them).
3. **Compact format.** `query -> label` lines instead of `Query: ... Label: ...` prose.
4. **Cap the output.** `max_tokens` 1–10 for a label, temperature 0.
5. **Put the query last** so the model's next token is the answer.
6. **Measure.** Use `response.usage.total_tokens`, not word counts — tokenization is not 1 word = 1 token.

### Before (~220 tokens)
```
Label the following instruction as an FAQ related answer or a product related answer for a clothing store.
Product related answers are answers specific about product information or that needs to use the products to give an answer.
Examples:
  Is there a refund for incorrectly bought clothes? Label: FAQ
  Where are your stores located?: Label: FAQ
  Tell me about the cheapest T-shirts that you have. Label: Product
  ... (8 examples)
Return only one of the two labels: FAQ or Product, nothing more.
Query to classify: {query}
```

### After (~90 tokens)
```
Classify a clothing-store query as FAQ (policies, returns, shipping, sizes, contact, promotions) or Product (needs catalog items: find, list, suggest, style). Reply with one word.
What sizes do your t-shirts come in? -> FAQ
Do you have blue T-shirts under $100? -> Product
Give me ideas for a sunny look. -> Product
{query} ->
```

## Same idea applied to other prompts
- **Context fields:** only include the fields the answer uses (`PRODUCT_CONTEXT_FIELDS`). Dropping `year`, `subCategory` and repeated labels saves ~15 tokens × 20 products per query.
- **Metadata extraction:** list allowed values once as compact JSON, no prose example, let the model omit keys it can't infer instead of filling `["Any"]` everywhere (`rag_kit/metadata_filter.py`).
- **Chat history:** store the user's *raw question* in history, not the context-stuffed prompt (`RAGApp.answer`). Otherwise every turn re-sends all previous retrieved documents.
- **Model tiering:** cheap model for routing/extraction, strong model only for the final answer.

## Pitfalls
- Shorter isn't always cheaper overall: if accuracy drops, you pay in wrong-pipeline costs. Always re-run the labelled test set.
- Don't remove the instruction that constrains the output format ("Reply with one word").
