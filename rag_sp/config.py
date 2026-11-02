"""Settings read from environment variables (see .env.example). Nothing secret is hard-coded."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    # Fiches pratiques Service-Public.gouv.fr (DILA), Licence Ouverte 2.0, published on data.gouv.fr
    source_url: str = field(default_factory=lambda: _env(
        "SOURCE_URL", "https://www.data.gouv.fr/api/1/datasets/r/0ed10f28-d197-4324-97b3-037f625095ac"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))          # empty -> local file storage
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))  # Qdrant Cloud
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", "data/qdrant"))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "service_public"))
    # any OpenAI-compatible chat API: Groq (default, free tier), Mistral, OpenAI, a local vLLM/Ollama...
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-oss-120b"))
    # cross-encoder that re-scores the first `rerank_candidates` passages; empty = no reranking
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    rerank_candidates: int = field(default_factory=lambda: int(_env("RERANK_CANDIDATES", "30")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "300")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))


def get_settings() -> Settings:
    return Settings()
