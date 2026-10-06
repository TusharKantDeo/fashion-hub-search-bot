from rag_kit.llm import FakeLLM
from rag_kit.router import LLMRouter


def make_router(reply):
    return LLMRouter(FakeLLM(default=reply), ["FAQ", "Product"], "Classify.", [("q1", "FAQ")])


def test_prompt_contains_examples_and_query():
    p = make_router("FAQ").build_prompt("Where is my order?")
    assert "q1 -> FAQ" in p and p.endswith("Where is my order? ->")


def test_normalizes_noisy_output():
    assert make_router(" faq.").classify("x").label == "FAQ"
    assert make_router("Label: Product").classify("x").label == "Product"


def test_unknown_label_returns_default():
    r = LLMRouter(FakeLLM(default="banana"), ["a", "b"], "x", default="b")
    assert r.classify("q").label == "b"


def test_uses_classification_params():
    llm = FakeLLM(default="FAQ")
    LLMRouter(llm, ["FAQ"], "x").classify("q")
    assert llm.calls[0]["params"].temperature == 0
