"""Create the Weaviate collections and load the sample data.

    python -m fashion_hub.ingest            # create if missing, upsert data
    python -m fashion_hub.ingest --recreate # drop and rebuild
"""

from __future__ import annotations

import argparse

from rag_kit.config import Settings

from fashion_hub import schema
from fashion_hub.factory import build_stores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client, products, faq = build_stores(Settings.from_env())
    try:
        product_props, faq_props = schema.weaviate_properties()
        products.create(product_props, recreate=args.recreate)
        faq.create(faq_props, recreate=args.recreate)
        n_p = products.ingest(schema.load_products(), id_key="product_id")
        n_f = faq.ingest(schema.load_faq(), id_key="question")
        print(f"Loaded {n_p} products and {n_f} FAQ entries.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
