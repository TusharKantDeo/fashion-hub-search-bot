"""Token ledger: record tokens per pipeline step, then estimate cost.

A query's real cost is the sum of *every* LLM call it triggers (router,
metadata extraction, final answer) — not just the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Fill with your provider's current prices.
EXAMPLE_PRICES: dict[str, tuple[float, float]] = {
    "Qwen/Qwen3.5-9B": (0.10, 0.10),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
}


@dataclass
class LedgerEntry:
    step: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenLedger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def add(self, step: str, resp) -> None:
        self.entries.append(LedgerEntry(step, resp.model, resp.prompt_tokens, resp.completion_tokens))

    @property
    def total_tokens(self) -> int:
        return sum(e.total for e in self.entries)

    def by_step(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.step] = out.get(e.step, 0) + e.total
        return out

    def cost_usd(self, prices: dict[str, tuple[float, float]] = EXAMPLE_PRICES) -> float:
        total = 0.0
        for e in self.entries:
            p_in, p_out = prices.get(e.model, (0.0, 0.0))
            total += e.prompt_tokens * p_in / 1e6 + e.completion_tokens * p_out / 1e6
        return total

    def summary(self) -> str:
        steps = ", ".join(f"{k}={v}" for k, v in self.by_step().items())
        return f"{self.total_tokens} tokens ({steps})"
