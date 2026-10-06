# 06 — Weaviate setup

## Three ways to run it
| Mode | Command | When |
|------|---------|------|
| Docker (local) | `docker compose up -d weaviate` | default for dev |
| Embedded | `WEAVIATE_MODE=embedded` (no Docker; client downloads a binary, Linux/macOS) | laptops without Docker, CI, notebooks |
| Weaviate Cloud | `WEAVIATE_MODE=cloud` + `WEAVIATE_URL` + `WEAVIATE_API_KEY` | shared/prod |

`rag_kit.retrieval.connect(settings)` picks the right one.

## docker-compose essentials
```yaml
weaviate:
  image: cr.weaviate.io/semitechnologies/weaviate:1.30.0
  ports: ["8080:8080", "50051:50051"]   # REST + gRPC (v4 client needs both)
  environment:
    AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"   # dev only
    PERSISTENCE_DATA_PATH: /var/lib/weaviate
    DEFAULT_VECTORIZER_MODULE: none                   # we bring our own vectors
```

## Bring your own vectors (recommended default)
```python
from weaviate.classes.config import Configure, Property, DataType, Tokenization

client.collections.create(
    "FashionProducts",
    vector_config=Configure.Vectors.self_provided(),
    properties=[
        Property(name="productDisplayName", data_type=DataType.TEXT),
        Property(name="gender", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
        Property(name="price", data_type=DataType.NUMBER),
    ],
)
```
- Embeddings are computed with `sentence-transformers` (`BAAI/bge-base-en-v1.5`) in `rag_kit.retrieval.Embedder`.
- BGE models work better with an instruction prefix on **queries only**: `"Represent this sentence for searching relevant passages: "`.
- Normalize embeddings (cosine distance).
- The course used a vectorizer module (`text2vec-transformers`) behind a Flask server. That works too, but self-provided vectors mean one less service and the embedding model is explicit in code.

## Tokenization for filter fields
Categorical fields that you filter on (`gender`, `baseColour`, ...) should use `Tokenization.FIELD` so `"Navy Blue"` is one value. With the default word tokenization, a filter for `Blue` would also match `Navy Blue`.

## Ingestion
```python
from weaviate.util import generate_uuid5
with collection.batch.fixed_size(batch_size=100) as batch:
    for rec, vec in zip(records, vectors):
        batch.add_object(properties=rec, vector=vec, uuid=generate_uuid5(str(rec["product_id"])))
if collection.batch.failed_objects: ...   # always check
```
Deterministic UUIDs (`generate_uuid5`) make re-ingestion idempotent (upsert instead of duplicates).

## Query
```python
from weaviate.classes.query import Filter, MetadataQuery
collection.query.near_vector(
    near_vector=embedder.embed_query(q), limit=20,
    filters=Filter.all_of([Filter.by_property("gender").contains_any(["Men"]),
                           Filter.by_property("price").less_or_equal(50)]),
    return_metadata=MetadataQuery(distance=True),
)
```
Other useful queries: `near_text` (needs a vectorizer module), `hybrid` (BM25 + vector, `alpha` balances them), `bm25`.

## Pitfalls
- v4 client talks gRPC: expose port **50051**.
- Always `client.close()` (use `try/finally`).
- Changing the embedding model ⇒ re-embed everything (`python -m fashion_hub.ingest --recreate`).
