"""Generic RAG orchestration: route -> handler builds a prompt -> generate.

Handlers don't call the final LLM themselves; they return a PreparedCall.
That keeps "how we build context" separate from "which model answers", so
the cheap model can do routing/extraction and a stronger model can answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rag_kit.cost import TokenLedger
from rag_kit.llm import LLMClient, LLMResponse, Message
from rag_kit.params import GenParams, params_for
from rag_kit.router import LLMRouter
from rag_kit.tracing import set_output, span


@dataclass
class PreparedCall:
    prompt: str
    params: GenParams = field(default_factory=lambda: params_for("default"))
    model: str | None = None
    sources: list = field(default_factory=list)


Handler = Callable[[str, TokenLedger], PreparedCall]


@dataclass
class Answer:
    content: str
    route: str | None
    ledger: TokenLedger
    sources: list


class RAGApp:
    def __init__(
        self,
        llm: LLMClient,
        router: LLMRouter,
        handlers: dict[str, Handler],
        fallback: Handler,
        answer_model: str | None = None,
        history_window: int = 6,
        system_prompt: str = "",
    ):
        self.llm = llm
        self.router = router
        self.handlers = handlers
        self.fallback = fallback
        self.answer_model = answer_model
        self.history_window = history_window
        self.system_prompt = system_prompt
        self.history: list[Message] = []

    def answer(self, query: str) -> Answer:
        ledger = TokenLedger()
        with span("rag_answer", kind="agent", input={"query": query}) as s:
            route = self.router.classify(query)
            if route.response is not None:
                ledger.add("route", route.response)
            handler = self.handlers.get(route.label or "", self.fallback)
            call = handler(query, ledger)

            messages: list[Message] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages += self.history[-self.history_window :]
            messages.append({"role": "user", "content": call.prompt})

            resp: LLMResponse = self.llm.complete(
                messages, call.params, call.model or self.answer_model, span_name="answer_llm"
            )
            ledger.add("answer", resp)
            # Store the raw question (not the context-stuffed prompt) to keep history cheap.
            self.history += [{"role": "user", "content": query}, {"role": "assistant", "content": resp.content}]
            s.set_attribute("rag.total_tokens", ledger.total_tokens)
            set_output(s, {"content": resp.content, "tokens": ledger.by_step()})
            return Answer(resp.content, route.label, ledger, call.sources)

    def reset(self) -> None:
        self.history.clear()

