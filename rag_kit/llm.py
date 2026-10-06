"""LLM client that always returns content *and* token usage.

Token usage is the unit of cost, so every call returns an LLMResponse with
prompt/completion/total tokens. Any OpenAI-compatible endpoint works
(Together, OpenAI, Groq, Ollama, vLLM, LM Studio) by changing base_url.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from rag_kit.params import GenParams
from rag_kit.tracing import record_llm_usage, span

Message = dict[str, str]


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    raw: Any = None


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        params: GenParams | None = None,
        model: str | None = None,
        span_name: str = "llm_call",
    ) -> LLMResponse: ...


def user(prompt: str) -> list[Message]:
    """Shortcut for a single-turn message list."""
    return [{"role": "user", "content": prompt}]


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        extra_body: dict | None = None,
        provider: str = "openai-compatible",
    ):
        from openai import OpenAI  # imported lazily so tests don't need network deps

        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.default_model = default_model
        self.extra_body = extra_body or {}
        self.provider = provider

    def complete(
        self,
        messages: list[Message],
        params: GenParams | None = None,
        model: str | None = None,
        span_name: str = "llm_call",
    ) -> LLMResponse:
        params = params or GenParams()
        model = model or self.default_model
        with span(span_name, kind="llm", input={"messages": messages, "params": params.as_kwargs()}) as s:
            raw = self.client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=self.extra_body or None,
                **params.as_kwargs(),
            )
            usage = raw.usage
            resp = LLMResponse(
                content=(raw.choices[0].message.content or "").strip(),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                model=raw.model or model,
                raw=raw,
            )
            record_llm_usage(s, resp, provider=self.provider)
            return resp


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests and offline demos.

    rules: list of (regex, reply). The first regex that matches the last
    message content wins. Tokens are approximated as whitespace-split words.
    """

    rules: list[tuple[str, str | Callable[[str], str]]] = field(default_factory=list)
    default: str = ""
    calls: list[dict] = field(default_factory=list)

    def complete(self, messages, params=None, model=None, span_name="llm_call") -> LLMResponse:
        prompt = messages[-1]["content"]
        self.calls.append({"messages": messages, "params": params, "model": model})
        reply = self.default
        for pattern, out in self.rules:
            if re.search(pattern, prompt, re.IGNORECASE | re.DOTALL):
                reply = out(prompt) if callable(out) else out
                break
        p = sum(len(m["content"].split()) for m in messages)
        c = len(reply.split())
        return LLMResponse(reply, p, c, p + c, model or "fake")
