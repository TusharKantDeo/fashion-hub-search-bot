from dataclasses import dataclass, field

import pytest

from rag_kit.retrieval import Doc
from fashion_hub import schema


@dataclass
class FakeStore:
    """Retriever stand-in. Returns `docs`, or pops from `script` per call; records filters."""

    docs: list = field(default_factory=list)
    script: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def search(self, query, limit=5, filters=None):
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        out = self.script.pop(0) if self.script else self.docs
        return out[:limit]


@pytest.fixture
def product_docs():
    return [Doc(str(p["product_id"]), p, 0.1) for p in schema.load_products()]


@pytest.fixture
def faq_docs():
    return [Doc(f["question"], f, 0.1) for f in schema.load_faq()]
