"""Prompt builders for grounded answers.

Keep the structure fixed (instructions -> tagged context -> question) so the
model always knows where the evidence ends and the question begins.
"""

from __future__ import annotations


def grounded_answer_prompt(
    question: str,
    context: str,
    role: str,
    rules: list[str] | None = None,
    context_tag: str = "CONTEXT",
) -> str:
    rules = rules or []
    rule_text = "\n".join(f"- {r}" for r in rules)
    return (
        f"{role}\n"
        f"Answer using only the {context_tag} below.\n"
        f"{rule_text}\n"
        f"<{context_tag}>\n{context}\n</{context_tag}>\n"
        f"Question: {question}"
    )
