---
name: weaviate-setup
description: Set up Weaviate (Docker, embedded, or Cloud) with the v4 Python client, create collections with self-provided vectors, ingest with batching, and query with near_vector/hybrid and filters. Use when the user needs a vector database for RAG or semantic search with Weaviate.
---

# Weaviate setup (v4 client)

Reference: `notes/06-weaviate-setup.md`, code in `rag_kit/retrieval.py`.

1. **Run it**
   - Docker: copy the `weaviate` service from `docker-compose.yml`; `docker compose up -d weaviate`. Expose 8080 and 50051 (gRPC is required by the v4 client).
   - No Docker: `WEAVIATE_MODE=embedded` → `weaviate.connect_to_embedded(persistence_data_path=...)`.
   - Cloud: `weaviate.connect_to_weaviate_cloud(cluster_url=..., auth_credentials=Auth.api_key(...))`.
   - Verify: `client.is_ready()`.
2. **Collection:** `client.collections.create(name, vector_config=Configure.Vectors.self_provided(), properties=[...])`. Filterable categorical text → `tokenization=Tokenization.FIELD`; numbers → `DataType.NUMBER`/`INT`.
3. **Embeddings:** sentence-transformers (`rag_kit.retrieval.Embedder`), normalized; BGE query instruction prefix on queries only.
4. **Ingest:** `collection.batch.fixed_size(batch_size=100)`, `uuid=generate_uuid5(stable_key)` for idempotent upserts, then check `collection.batch.failed_objects`.
5. **Query:** `collection.query.near_vector(near_vector=vec, limit=k, filters=Filter.all_of([...]), return_metadata=MetadataQuery(distance=True))`. Use `hybrid(query, vector=vec, alpha=0.5)` when exact terms (brands, SKUs) matter.
6. Always `client.close()` in `finally`.

Before writing code against a newer client version, check current docs (context7 `/weaviate/weaviate-python-client`) — the v4 API renamed `vectorizer_config` → `vector_config`.
