"""Weaviate (v4 client) helpers: connect, embed, create, ingest, search.

Design choice: vectors are computed in Python with sentence-transformers and
sent to Weaviate ("self-provided" vectors). That keeps Weaviate config trivial
(no vectorizer module/container) and makes the embedding model explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from rag_kit.tracing import record_retrieval, span


@dataclass
class Doc:
    id: str
    properties: dict[str, Any]
    distance: float | None = None


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5, filters: Any = None) -> list[Doc]: ...


# ---------------------------------------------------------------- connection
def connect(settings) -> Any:
    """Return a connected weaviate client for settings.weaviate_mode."""
    import weaviate
    from weaviate.classes.init import Auth

    mode = settings.weaviate_mode
    if mode == "local":
        return weaviate.connect_to_local(
            host=settings.weaviate_host, port=settings.weaviate_port, grpc_port=settings.weaviate_grpc_port
        )
    if mode == "embedded":
        # Downloads and runs a Weaviate binary in-process (Linux/macOS). No Docker needed.
        return weaviate.connect_to_embedded(persistence_data_path=settings.weaviate_data_path)
    if mode == "cloud":
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url, auth_credentials=Auth.api_key(settings.weaviate_api_key)
        )
    raise ValueError(f"Unknown WEAVIATE_MODE: {mode!r}")


# ---------------------------------------------------------------- embeddings
class Embedder:
    """sentence-transformers wrapper. BGE models like a short instruction on queries."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_instruction = query_instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_instruction + text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------- store
@dataclass
class WeaviateStore:
    """One collection + the function that turns a record into embeddable text."""

    client: Any
    name: str
    embedder: Embedder
    text_fn: Callable[[dict], str]
    return_properties: list[str] | None = None
    _collection: Any = field(init=False, default=None)

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.collections.get(self.name)
        return self._collection

    def create(self, properties: list[Any], recreate: bool = False) -> None:
        """properties: list of weaviate.classes.config.Property."""
        from weaviate.classes.config import Configure

        if self.client.collections.exists(self.name):
            if not recreate:
                return
            self.client.collections.delete(self.name)
        self.client.collections.create(
            self.name,
            vector_config=Configure.Vectors.self_provided(),
            properties=properties,
        )
        self._collection = None

    def ingest(self, records: Iterable[dict], id_key: str | None = None, batch_size: int = 100) -> int:
        """Embed and upsert records. Deterministic UUIDs make re-running idempotent."""
        from weaviate.util import generate_uuid5

        records = list(records)
        vectors = self.embedder.embed_documents([self.text_fn(r) for r in records])
        with self.collection.batch.fixed_size(batch_size=batch_size) as batch:
            for r, v in zip(records, vectors):
                key = r[id_key] if id_key else self.text_fn(r)
                batch.add_object(properties=r, vector=v, uuid=generate_uuid5(str(key)))
        failed = self.collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed to import, first: {failed[0].message}")
        return len(records)

    def search(self, query: str, limit: int = 5, filters: Any = None) -> list[Doc]:
        from weaviate.classes.query import MetadataQuery

        with span(f"search_{self.name}", kind="retriever", input={"query": query, "limit": limit}) as s:
            s.set_attribute("retrieval.filters", str(filters) if filters is not None else "")
            res = self.collection.query.near_vector(
                near_vector=self.embedder.embed_query(query),
                limit=limit,
                filters=filters,
                return_metadata=MetadataQuery(distance=True),
                return_properties=self.return_properties,
            )
            docs = [Doc(str(o.uuid), dict(o.properties), o.metadata.distance) for o in res.objects]
            record_retrieval(s, docs)
            return docs
