"""Few-shot LLM classifier for routing queries (FAQ vs Product, creative vs technical, ...).

Token-saving recipe (notes/03-prompt-token-optimization.md):
1. One-line instruction that *defines* each label instead of many examples.
2. Only the hardest 2-4 examples (the ones the model gets wrong without them).
3. temperature=0 and max_tokens≈5 — the output is a single label.
4. Normalize the output, because models sometimes add punctuation or words.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_kit.llm import LLMClient, LLMResponse, user
from rag_kit.params import PRESETS, GenParams
from rag_kit.tracing import span


@dataclass
class RouteResult:
    label: str | None
    tokens: int
    raw: str
    response: LLMResponse | None = None


class LLMRouter:
    def __init__(
        self,
        llm: LLMClient,
        labels: list[str],
        instruction: str,
        examples: list[tuple[str, str]] | None = None,
        params: GenParams = PRESETS["classification"],
        model: str | None = None,
        default: str | None = None,
        name: str = "router",
    ):
        self.llm = llm
        self.labels = labels
        self.instruction = instruction.strip()
        self.examples = examples or []
        self.params = params
        self.model = model
        self.default = default
        self.name = name

    def build_prompt(self, query: str) -> str:
        lines = [self.instruction]
        lines += [f"{q} -> {label}" for q, label in self.examples]
        lines.append(f"{query} ->")
        return "\n".join(lines)

    def normalize(self, raw: str) -> str | None:
        text = raw.strip().lower()
        # Prefer an exact match, then the first label mentioned anywhere.
        for label in self.labels:
            if text.strip(" .:'\"") == label.lower():
                return label
        hits = [(text.find(label.lower()), label) for label in self.labels if label.lower() in text]
        return min(hits)[1] if hits else self.default

    def classify(self, query: str) -> RouteResult:
        with span(self.name, kind="chain", input={"query": query}) as s:
            resp = self.llm.complete(
                user(self.build_prompt(query)), self.params, self.model, span_name=f"{self.name}_llm"
            )
            label = self.normalize(resp.content)
            s.set_attribute("output.value", str({"label": label, "tokens": resp.total_tokens}))
            return RouteResult(label=label, tokens=resp.total_tokens, raw=resp.content, response=resp)
