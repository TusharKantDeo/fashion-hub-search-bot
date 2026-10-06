"""Settings loaded from environment variables (and an optional .env file).

Keep every endpoint, model name and key here so the rest of the code never
reads os.environ directly. Swap providers by changing env vars, not code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # LLM (any OpenAI-compatible endpoint: Together, OpenAI, Groq, Ollama, vLLM, ...)
    llm_base_url: str = "https://api.together.xyz/v1"
    llm_api_key: str = ""
    # Cheap/fast model for routing + metadata extraction; stronger model for final answers.
    router_model: str = "Qwen/Qwen3.5-9B"
    answer_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # Provider-specific extra request body, e.g. {"reasoning": {"enabled": false}} on Together.
    llm_extra_body: dict = field(default_factory=dict)

    # Embeddings (computed locally with sentence-transformers, vectors sent to Weaviate)
    embed_model: str = "BAAI/bge-base-en-v1.5"

    # Weaviate: "local" (docker), "embedded" (no docker), or "cloud"
    weaviate_mode: str = "local"
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_grpc_port: int = 50051
    weaviate_url: str = ""
    weaviate_api_key: str = ""
    weaviate_data_path: str = ".weaviate-data"

    # Phoenix tracing
    phoenix_enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project: str = "fashion-hub"

    @classmethod
    def from_env(cls, dotenv: str | Path | None = ".env") -> "Settings":
        if dotenv:
            load_dotenv(dotenv)
        extra = _env("LLM_EXTRA_BODY", "")
        return cls(
            llm_base_url=_env("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=_env("LLM_API_KEY", ""),
            router_model=_env("ROUTER_MODEL", cls.router_model),
            answer_model=_env("ANSWER_MODEL", cls.answer_model),
            llm_extra_body=json.loads(extra) if extra else {},
            embed_model=_env("EMBED_MODEL", cls.embed_model),
            weaviate_mode=_env("WEAVIATE_MODE", cls.weaviate_mode),
            weaviate_host=_env("WEAVIATE_HOST", cls.weaviate_host),
            weaviate_port=int(_env("WEAVIATE_PORT", str(cls.weaviate_port))),
            weaviate_grpc_port=int(_env("WEAVIATE_GRPC_PORT", str(cls.weaviate_grpc_port))),
            weaviate_url=_env("WEAVIATE_URL", ""),
            weaviate_api_key=_env("WEAVIATE_API_KEY", ""),
            weaviate_data_path=_env("WEAVIATE_DATA_PATH", cls.weaviate_data_path),
            phoenix_enabled=_env("PHOENIX_ENABLED", "true").lower() == "true",
            phoenix_endpoint=_env("PHOENIX_ENDPOINT", cls.phoenix_endpoint),
            phoenix_project=_env("PHOENIX_PROJECT", cls.phoenix_project),
        )
