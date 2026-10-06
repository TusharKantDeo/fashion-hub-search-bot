from rag_kit.metadata_filter import (
    FilterSchema,
    MetadataFilterExtractor,
    allowed_values,
    parse_json_loose,
    sanitize,
    search_with_relaxation,
    to_weaviate_filters,
)
from rag_kit.llm import FakeLLM

SCHEMA = FilterSchema(
    categorical={"gender": ["Men", "Women"], "baseColour": ["Blue", "Navy Blue"]},
    numeric=["price"],
    relax_order=["baseColour", "gender"],
)


def test_parse_handles_fences_prose_and_inf():
    text = 'Sure!\n```json\n{"gender": ["Men"], "price": {"min": 0, "max": inf}}\n```'
    assert parse_json_loose(text) == {"gender": ["Men"], "price": {"min": 0, "max": "inf"}}


def test_parse_returns_none_on_garbage():
    assert parse_json_loose("no json here") is None


def test_sanitize_drops_unknown_values_and_any():
    meta = {"gender": ["men", "Any"], "baseColour": "Purple", "foo": ["x"], "price": {"min": 0, "max": "inf"}}
    assert sanitize(meta, SCHEMA) == {"gender": ["Men"]}


def test_sanitize_keeps_price_bounds():
    assert sanitize({"price": {"min": 10, "max": 50}}, SCHEMA) == {"price": {"min": 10.0, "max": 50.0}}
    assert sanitize({"price": {"min": 0, "max": 50}}, SCHEMA) == {"price": {"min": None, "max": 50.0}}


def test_to_weaviate_filters_keyed_by_property():
    f = to_weaviate_filters({"gender": ["Men"], "price": {"min": None, "max": 50.0}})
    assert set(f) == {"gender", "price"}


def test_allowed_values_from_records():
    recs = [{"g": "A"}, {"g": "B"}, {"g": "A"}, {"g": None}]
    assert allowed_values(recs, ["g"]) == {"g": ["A", "B"]}


def test_extractor_prompt_is_compact_and_returns_response():
    llm = FakeLLM(default='{"gender": ["Women"]}')
    ex = MetadataFilterExtractor(llm, SCHEMA)
    meta, resp = ex.extract("dress for women")
    assert meta == {"gender": ["Women"]}
    assert resp.total_tokens > 0
    assert len(ex.build_prompt("q").split()) < 120


def test_relaxation_drops_least_important_first():
    from tests.conftest import FakeStore

    store = FakeStore(script=[[], [1, 2, 3, 4, 5]])
    filters = to_weaviate_filters({"gender": ["Men"], "baseColour": ["Blue"]})
    docs, active = search_with_relaxation(store, "q", filters, SCHEMA.relax_order, min_results=5)
    assert len(docs) == 5 and active == ["gender"]
    assert len(store.calls) == 2


def test_relaxation_stops_when_no_filters_left():
    from tests.conftest import FakeStore

    store = FakeStore(docs=[1])
    docs, active = search_with_relaxation(store, "q", to_weaviate_filters({"gender": ["Men"]}), [], min_results=5)
    assert active == [] and store.calls[-1]["filters"] is None
