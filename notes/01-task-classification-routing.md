# 01 — Task classification & routing

## Idea
Before doing anything expensive, ask a **small, cheap model** one question: *what kind of request is this?*
The label picks the pipeline (FAQ vs Product) and later the generation settings (creative vs technical).

## Why
- Different questions need different data sources. Sending an FAQ question through product search wastes tokens and gives bad answers.
- A routing call costs ~50–150 tokens; a wrong pipeline can cost thousands.
- The label is also a free signal for analytics ("70% of traffic is returns questions").

## How (rag_kit)
```python
from rag_kit import LLMRouter

router = LLMRouter(
    llm,
    labels=["FAQ", "Product"],
    instruction="Classify a clothing-store query as FAQ (policies, returns, shipping, sizes, contact, "
                "promotions) or Product (needs catalog items: find, list, suggest, style). Reply with one word.",
    examples=[("What sizes do your t-shirts come in?", "FAQ"),
              ("Do you have blue T-shirts under $100?", "Product"),
              ("Give me ideas for a sunny look.", "Product")],
    model="small-cheap-model",
)
router.classify("How do I return a dress?")   # RouteResult(label="FAQ", tokens=~90, ...)
```
Two levels of routing are used in the bot:
1. `FAQ | Product` — which data source.
2. `creative | technical` — which generation parameters (see [02](02-llm-params-by-task.md)).

## Pitfalls
- **Normalize the output.** Models return `"FAQ."`, `"Label: Product"`, `"faq"`. `LLMRouter.normalize` handles that; also return a `default` for garbage.
- **Tricky cases come from overlap.** "What sizes do T-shirts come in?" mentions a product but is an FAQ. Put exactly those borderline cases in the examples.
- **Reasoning/thinking models** spend tokens "thinking" before the label. With `max_tokens=5` you get an empty/cut answer. Disable reasoning (Together: `{"reasoning": {"enabled": false}}`) or use a non-reasoning model for routing.
- Always have a **fallback handler** for labels you didn't expect.
- Evaluate the router on a small labelled set (10–30 queries) whenever you change the prompt.

## Alternatives
- Embedding classifier (nearest labelled example) — 0 LLM tokens, needs labelled data.
- Keyword rules as a first pass, LLM only when unsure.
- Structured output / function calling to force one of N labels.
