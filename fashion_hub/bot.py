"""Fashion Hub bot: wires rag_kit pieces into a FAQ + product assistant.

Flow for every message:
    route (FAQ | Product)                           cheap model, ~100 tokens
    FAQ     -> top-k FAQ retrieval -> grounded prompt
    Product -> task nature (creative | technical)   cheap model, sets temperature/top_p
            -> retrieve products:
                 strategy="metadata": LLM extracts filters -> filtered search (+ relaxation)
                 strategy="direct":   semantic search on the raw query
            -> grounded prompt
    answer                                           stronger model
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from rag_kit.context import format_records, order_for_prompt
from rag_kit.cost import TokenLedger
from rag_kit.llm import LLMClient
from rag_kit.metadata_filter import MetadataFilterExtractor, search_with_relaxation, to_weaviate_filters
from rag_kit.params import params_for
from rag_kit.pipeline import PreparedCall, RAGApp
from rag_kit.prompts import grounded_answer_prompt
from rag_kit.retrieval import Retriever
from rag_kit.router import LLMRouter

from fashion_hub import schema

Strategy = Literal["metadata", "direct"]

SYSTEM_PROMPT = (
    "You are a friendly assistant for Fashion Hub, a clothing store. "
    "Answer questions about the store (FAQ) and its products."
)


class FashionHubBot:
    def __init__(
        self,
        llm: LLMClient,
        products: Retriever,
        faq: Retriever,
        router_model: str | None = None,
        answer_model: str | None = None,
        strategy: Strategy = "metadata",
        product_limit: int = 20,
        faq_limit: int = 5,
        filter_schema=None,
    ):
        self.llm = llm
        self.products = products
        self.faq = faq
        self.strategy = strategy
        self.product_limit = product_limit
        self.faq_limit = faq_limit

        self.task_router = LLMRouter(
            llm,
            labels=["creative", "technical"],
            instruction=schema.TASK_NATURE_INSTRUCTION,
            examples=schema.TASK_NATURE_EXAMPLES,
            model=router_model,
            default="technical",
            name="decide_task_nature",
        )
        self.extractor = MetadataFilterExtractor(
            llm,
            filter_schema or schema.filter_schema(schema.load_products()),
            domain="clothing products",
            model=router_model,
        )
        self.app = RAGApp(
            llm=llm,
            router=LLMRouter(
                llm,
                labels=["FAQ", "Product"],
                instruction=schema.FAQ_OR_PRODUCT_INSTRUCTION,
                examples=schema.FAQ_OR_PRODUCT_EXAMPLES,
                model=router_model,
                name="route_faq_or_product",
            ),
            handlers={"FAQ": self.handle_faq, "Product": self.handle_product},
            fallback=self.handle_other,
            answer_model=answer_model,
            system_prompt=SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------ handlers
    def handle_faq(self, query: str, ledger: TokenLedger) -> PreparedCall:
        docs = self.faq.search(query, limit=self.faq_limit)
        context = format_records(order_for_prompt(docs), schema.FAQ_CONTEXT_FIELDS)
        prompt = grounded_answer_prompt(
            query,
            context,
            role="You answer customer questions for a clothing store.",
            rules=[
                "FAQ items are ordered least to most relevant; the last is the best match.",
                "Combine items if needed. Do not mention that you were given an FAQ.",
            ],
            context_tag="FAQ",
        )
        return PreparedCall(prompt, params_for("technical"), sources=docs)

    def handle_product(self, query: str, ledger: TokenLedger) -> PreparedCall:
        nature = self.task_router.classify(query)
        if nature.response is not None:
            ledger.add("task_nature", nature.response)

        docs = self.retrieve_products(query, ledger)
        context = format_records(order_for_prompt(docs), schema.PRODUCT_CONTEXT_FIELDS)
        prompt = grounded_answer_prompt(
            query,
            context,
            role="You recommend clothing products from the catalog below.",
            rules=[
                "Use at most 5 products and always include each product's ID as 'ID: <number>'.",
                "Use only the attributes relevant to the question.",
                "If nothing fits, say so and suggest the closest options.",
            ],
            context_tag="PRODUCTS",
        )
        return PreparedCall(prompt, params_for(nature.label), sources=docs)

    def handle_other(self, query: str, ledger: TokenLedger) -> PreparedCall:
        prompt = (
            "The user's message is not a store FAQ or product question. Reply briefly and "
            f"steer them back to what you can help with.\nUser: {query}"
        )
        return PreparedCall(prompt, replace(params_for("default"), max_tokens=150))

    # ------------------------------------------------------------ retrieval
    def retrieve_products(self, query: str, ledger: TokenLedger | None = None):
        if self.strategy == "direct":
            return self.products.search(query, limit=self.product_limit)

        meta, resp = self.extractor.extract(query)
        if ledger is not None:
            ledger.add("metadata", resp)
        filters = to_weaviate_filters(meta)
        docs, _ = search_with_relaxation(
            self.products,
            query,
            filters,
            relax_order=self.extractor.schema.relax_order,
            limit=self.product_limit,
            min_results=5,
        )
        return docs

    # ------------------------------------------------------------ public API
    def chat(self, message: str):
        return self.app.answer(message)

    def reset(self) -> None:
        self.app.reset()

