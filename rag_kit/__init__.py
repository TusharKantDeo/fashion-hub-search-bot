"""rag_kit — a small, copy-pasteable toolkit for building RAG systems.

Every module is independent enough to copy on its own:

    config.py           Settings loaded from environment variables
    llm.py              OpenAI-compatible LLM client that always returns token usage
    params.py           Generation presets per task type (classification, creative, ...)
    router.py           Few-shot LLM classifier used to route queries
    retrieval.py        Weaviate connection, embeddings, ingestion and semantic search
    metadata_filter.py  LLM -> JSON metadata -> Weaviate filters, with relaxation
    context.py          Turn retrieved records into compact prompt context
    prompts.py          Grounded-answer prompt builders
    tracing.py          OpenTelemetry/Phoenix spans that work with or without Phoenix
    cost.py             Token ledger and cost estimation
    pipeline.py         Route -> handler -> generate orchestration

See notes/ for the reasoning behind each piece.
"""

from rag_kit.config import Settings
from rag_kit.context import format_records, order_for_prompt
from rag_kit.cost import TokenLedger
from rag_kit.llm import FakeLLM, LLMResponse, OpenAICompatibleLLM
from rag_kit.params import PRESETS, GenParams, params_for
from rag_kit.pipeline import PreparedCall, RAGApp
from rag_kit.router import LLMRouter, RouteResult

__all__ = [
    "Settings",
    "format_records",
    "order_for_prompt",
    "TokenLedger",
    "FakeLLM",
    "LLMResponse",
    "OpenAICompatibleLLM",
    "PRESETS",
    "GenParams",
    "params_for",
    "PreparedCall",
    "RAGApp",
    "LLMRouter",
    "RouteResult",
]
