from rag_kit.context import format_records, order_for_prompt
from rag_kit.cost import TokenLedger
from rag_kit.llm import LLMResponse


def test_format_records_skips_empty_and_respects_budget():
    recs = [{"id": 1, "name": "A", "x": ""}, {"id": 2, "name": "B", "x": None}]
    assert format_records(recs, {"ID": "id", "Name": "name", "X": "x"}) == "ID: 1 | Name: A\nID: 2 | Name: B"
    assert format_records(recs, {"ID": "id"}, max_chars=6) == "ID: 1"


def test_most_relevant_goes_last():
    assert order_for_prompt([1, 2, 3]) == [3, 2, 1]


def test_ledger_sums_steps_and_costs():
    led = TokenLedger()
    led.add("route", LLMResponse("x", 100, 1, 101, "m"))
    led.add("answer", LLMResponse("y", 1000, 200, 1200, "m"))
    assert led.total_tokens == 1301
    assert led.by_step() == {"route": 101, "answer": 1200}
    assert abs(led.cost_usd({"m": (1.0, 2.0)}) - (1100 / 1e6 + 201 * 2 / 1e6)) < 1e-12
