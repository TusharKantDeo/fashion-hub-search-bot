"""Compare metadata-filter retrieval vs direct semantic search.

For each query prints: tokens spent on retrieval, and how many of the top-5
results satisfy the query's hard constraints (a cheap quality proxy).

    python scripts/compare_strategies.py
"""

from __future__ import annotations

from rag_kit.config import Settings
from rag_kit.cost import TokenLedger

from fashion_hub.factory import build_bot

# (query, constraint check on a product dict)
CASES = [
    ("Blue T-shirts for men", lambda p: p["gender"] == "Men" and p["articleType"] == "Tshirts" and "Blue" in p["baseColour"]),
    ("Black party dresses for women under $100", lambda p: p["articleType"] == "Dresses" and p["baseColour"] == "Black" and p["price"] <= 100),
    ("Formal shoes for men", lambda p: p["gender"] == "Men" and p["articleType"] == "Formal Shoes"),
    ("Something gold to wear to a party, for women", lambda p: p["gender"] == "Women" and p["baseColour"] == "Gold"),
    ("Summer casual outfit for a woman", lambda p: p["gender"] == "Women" and p["season"] == "Summer" and p["usage"] == "Casual"),
]


def main() -> None:
    settings = Settings.from_env()
    client, bot = build_bot(settings, strategy="metadata")
    try:
        print(f"{'query':48} {'strategy':9} {'tokens':>6} {'top5 precision':>15}")
        for query, ok in CASES:
            for strategy in ("direct", "metadata"):
                bot.strategy = strategy
                ledger = TokenLedger()
                docs = bot.retrieve_products(query, ledger)[:5]
                precision = sum(ok(d.properties) for d in docs) / max(len(docs), 1)
                print(f"{query[:48]:48} {strategy:9} {ledger.total_tokens:>6} {precision:>15.0%}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
