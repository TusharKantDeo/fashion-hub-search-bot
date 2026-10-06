"""Build real clients (LLM, Weaviate stores, tracing) from Settings."""

from __future__ import annotations

from rag_kit.config import Settings
from rag_kit.llm import OpenAICompatibleLLM
from rag_kit.retrieval import Embedder, WeaviateStore, connect
from rag_kit.tracing import setup_phoenix

from fashion_hub import schema
from fashion_hub.bot import FashionHubBot

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def build_stores(settings: Settings):
    client = connect(settings)
    embedder = Embedder(settings.embed_model, query_instruction=BGE_QUERY_INSTRUCTION)
    products = WeaviateStore(client, schema.PRODUCTS, embedder, schema.product_text)
    faq = WeaviateStore(client, schema.FAQ, embedder, schema.faq_text)
    return client, products, faq


def build_bot(settings: Settings, strategy: str = "metadata"):
    if settings.phoenix_enabled:
        setup_phoenix(settings.phoenix_project, settings.phoenix_endpoint)
    llm = OpenAICompatibleLLM(
        settings.llm_base_url, settings.llm_api_key, settings.router_model, settings.llm_extra_body
    )
    client, products, faq = build_stores(settings)
    bot = FashionHubBot(
        llm,
        products,
        faq,
        router_model=settings.router_model,
        answer_model=settings.answer_model,
        strategy=strategy,
    )
    return client, bot
