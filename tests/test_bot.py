"""End-to-end bot flow with a fake LLM and fake stores (no network)."""

from rag_kit.llm import FakeLLM
from fashion_hub import schema
from fashion_hub.bot import FashionHubBot
from tests.conftest import FakeStore


def fake_llm():
    return FakeLLM(
        rules=[
            (r"as FAQ .* or Product.*Where is my refund", "FAQ"),
            (r"as FAQ .* or Product", "Product"),
            (r"creative .* or technical.*look", "creative"),
            (r"creative .* or technical", "technical"),
            (r"Extract search filters", '{"gender": ["Men"], "articleType": ["Tshirts"], "baseColour": ["Blue"]}'),
        ],
        default="Here you go. ID: 10001",
    )


def make_bot(product_docs, faq_docs, strategy):
    llm = fake_llm()
    products, faq = FakeStore(docs=product_docs), FakeStore(docs=faq_docs)
    bot = FashionHubBot(llm, products, faq, strategy=strategy,
                        filter_schema=schema.filter_schema(schema.load_products()))
    return bot, llm, products, faq


def test_faq_route_uses_retrieved_faq_only(product_docs, faq_docs):
    bot, llm, products, faq = make_bot(product_docs, faq_docs, "metadata")
    ans = bot.chat("Where is my refund?")
    assert ans.route == "FAQ"
    assert len(faq.calls) == 1 and faq.calls[0]["limit"] == 5
    assert not products.calls
    assert set(ans.ledger.by_step()) == {"route", "answer"}


def test_product_metadata_strategy_applies_filters(product_docs, faq_docs):
    bot, llm, products, _ = make_bot(product_docs, faq_docs, "metadata")
    ans = bot.chat("Blue t-shirts for men")
    assert ans.route == "Product"
    assert products.calls[0]["filters"] is not None
    assert set(ans.ledger.by_step()) == {"route", "task_nature", "metadata", "answer"}
    final_prompt = llm.calls[-1]["messages"][-1]["content"]
    assert "<PRODUCTS>" in final_prompt and "ID: " in final_prompt


def test_product_direct_strategy_skips_metadata_llm(product_docs, faq_docs):
    bot, llm, products, _ = make_bot(product_docs, faq_docs, "direct")
    ans = bot.chat("Blue t-shirts for men")
    assert products.calls[0]["filters"] is None
    assert "metadata" not in ans.ledger.by_step()


def test_creative_query_gets_creative_params(product_docs, faq_docs):
    bot, llm, *_ = make_bot(product_docs, faq_docs, "direct")
    bot.chat("Create a look for a beach party")
    assert llm.calls[-1]["params"].temperature == 1.0


def test_history_stores_raw_question_not_context(product_docs, faq_docs):
    bot, *_ = make_bot(product_docs, faq_docs, "direct")
    bot.chat("Blue t-shirts for men")
    assert bot.app.history[0] == {"role": "user", "content": "Blue t-shirts for men"}
