# References

## Course
- DeepLearning.AI — *Retrieval Augmented Generation (RAG)* course (Coursera). This repo is my own re-implementation of the ideas from its final assignment; no course code, data or graded solutions are included.

## Tools
- Weaviate Python client v4 — https://weaviate.io/developers/weaviate/client-libraries/python
- Weaviate filters — https://weaviate.io/developers/weaviate/search/filters
- Weaviate hybrid search — https://weaviate.io/developers/weaviate/search/hybrid
- Weaviate Docker install — https://weaviate.io/developers/weaviate/installation/docker-compose
- Arize Phoenix docs — https://arize.com/docs/phoenix
- phoenix.otel `register` — https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel
- OpenInference semantic conventions — https://github.com/Arize-ai/openinference/tree/main/spec
- OpenTelemetry Python — https://opentelemetry.io/docs/languages/python/
- sentence-transformers — https://sbert.net
- BGE embedding models — https://huggingface.co/BAAI/bge-base-en-v1.5
- Together AI (OpenAI-compatible API) — https://docs.together.ai
- OpenAI Python SDK — https://github.com/openai/openai-python

## Papers / concepts
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — https://arxiv.org/abs/2005.11401
- Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts* — https://arxiv.org/abs/2307.03172
- Holtzman et al., 2019 — *The Curious Case of Neural Text Degeneration* (nucleus / top-p sampling) — https://arxiv.org/abs/1904.09751
- LangChain "self-query retriever" — the same LLM→filters pattern as `metadata_filter.py`.
