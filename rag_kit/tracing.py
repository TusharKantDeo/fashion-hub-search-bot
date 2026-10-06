"""Tracing helpers built on plain OpenTelemetry + OpenInference attribute names.

Why plain OTel: the same code runs with Phoenix (spans show up in the UI with
token counts and costs) and without it (spans are no-ops). Nothing in the app
needs to know whether tracing is on.

Span kinds Phoenix understands: chain, llm, retriever, tool, agent, embedding,
reranker, guardrail, evaluator.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

_TRACER_NAME = "rag_kit"


def setup_phoenix(project_name: str, endpoint: str = "http://localhost:6006/v1/traces") -> bool:
    """Register a tracer provider that exports to Phoenix. Returns False if unavailable."""
    try:
        from phoenix.otel import register
    except ImportError:
        print("[tracing] arize-phoenix-otel not installed; tracing disabled.")
        return False
    # auto_instrument=False on purpose: we trace exactly what we want, nothing more.
    register(project_name=project_name, endpoint=endpoint, auto_instrument=False, batch=True)
    return True


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


@contextmanager
def span(name: str, kind: str = "chain", input: Any = None) -> Iterator[Span]:
    """Start a span with an OpenInference kind; records errors and OK status."""
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind.upper())
        if input is not None:
            s.set_attribute("input.value", _to_str(input))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            s.set_status(Status(StatusCode.OK))


def set_output(s: Span, value: Any) -> None:
    s.set_attribute("output.value", _to_str(value))


def record_llm_usage(s: Span, resp: Any, provider: str = "") -> None:
    """OpenInference conventions Phoenix uses to compute token counts and cost."""
    s.set_attribute("llm.token_count.prompt", resp.prompt_tokens)
    s.set_attribute("llm.token_count.completion", resp.completion_tokens)
    s.set_attribute("llm.token_count.total", resp.total_tokens)
    s.set_attribute("llm.model_name", resp.model)
    if provider:
        s.set_attribute("llm.provider", provider)
    set_output(s, resp.content)


def record_retrieval(s: Span, docs: Iterable[Any]) -> None:
    """Attach retrieved documents so Phoenix shows them on retriever spans."""
    for i, d in enumerate(docs):
        s.set_attribute(f"retrieval.documents.{i}.document.id", str(getattr(d, "id", i)))
        s.set_attribute(f"retrieval.documents.{i}.document.content", _to_str(getattr(d, "properties", d)))
        score = getattr(d, "distance", None)
        if score is not None:
            s.set_attribute(f"retrieval.documents.{i}.document.score", float(score))
