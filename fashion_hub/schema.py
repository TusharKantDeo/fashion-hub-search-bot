"""Everything specific to the Fashion Hub data: collections, fields, prompts.

To adapt the bot to another domain, this is the main file to rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_kit.metadata_filter import FilterSchema, allowed_values

DATA_DIR = Path(__file__).parent / "data"

PRODUCTS = "FashionProducts"
FAQ = "FashionFaq"

# Fields the LLM may filter on. Least important first in RELAX_ORDER:
# colour is the first constraint we give up, gender the last.
FILTER_KEYS = ["gender", "masterCategory", "articleType", "baseColour", "season", "usage"]
RELAX_ORDER = ["baseColour", "season", "usage", "articleType", "masterCategory", "gender"]

# Only the fields the answer model needs — each one costs tokens on every query.
PRODUCT_CONTEXT_FIELDS = {
    "ID": "product_id",
    "Name": "productDisplayName",
    "Type": "articleType",
    "Colour": "baseColour",
    "Gender": "gender",
    "Usage": "usage",
    "Season": "season",
    "Price": "price",
}
FAQ_CONTEXT_FIELDS = {"Q": "question", "A": "answer"}

# ---- routing prompts (short instruction + only the hardest examples) ----
FAQ_OR_PRODUCT_INSTRUCTION = (
    "Classify a clothing-store query as FAQ (policies, returns, shipping, sizes, contact, promotions) "
    "or Product (needs catalog items: find, list, suggest, style). Reply with one word."
)
FAQ_OR_PRODUCT_EXAMPLES = [
    ("What sizes do your t-shirts come in?", "FAQ"),
    ("Do you have blue T-shirts under $100?", "Product"),
    ("Give me ideas for a sunny look.", "Product"),
]

TASK_NATURE_INSTRUCTION = (
    "Classify a product query as creative (compose a look, style, match items) "
    "or technical (find/list products, prices, attributes). Reply with one word."
)
TASK_NATURE_EXAMPLES = [
    ("Give me three T-shirts for summer.", "technical"),
    ("Suggest an accessory to match my green dress.", "creative"),
]


def load_products() -> list[dict]:
    return json.loads((DATA_DIR / "products.json").read_text())


def load_faq() -> list[dict]:
    return json.loads((DATA_DIR / "faq.json").read_text())


def product_text(p: dict) -> str:
    """Text that gets embedded for each product."""
    return (
        f"{p['productDisplayName']}. {p['gender']} {p['articleType']} in {p['baseColour']}, "
        f"{p['usage']} wear for {p['season']}."
    )


def faq_text(f: dict) -> str:
    return f"{f['question']} {f['answer']}"


def filter_schema(products: list[dict]) -> FilterSchema:
    return FilterSchema(
        categorical=allowed_values(products, FILTER_KEYS),
        numeric=["price"],
        relax_order=RELAX_ORDER,
    )


def weaviate_properties():
    """Categorical fields use FIELD tokenization so filters match whole values ("Navy Blue")."""
    from weaviate.classes.config import DataType, Property, Tokenization

    cat = dict(data_type=DataType.TEXT, tokenization=Tokenization.FIELD)
    products = [
        Property(name="product_id", data_type=DataType.INT),
        Property(name="productDisplayName", data_type=DataType.TEXT),
        Property(name="subCategory", **cat),
        Property(name="year", data_type=DataType.INT),
        Property(name="price", data_type=DataType.NUMBER),
        *[Property(name=k, **cat) for k in FILTER_KEYS],
    ]
    faq = [
        Property(name="question", data_type=DataType.TEXT),
        Property(name="answer", data_type=DataType.TEXT),
        Property(name="type", **cat),
    ]
    return products, faq
