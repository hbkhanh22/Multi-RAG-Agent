"""
Configuration settings for Multi-RAG Agent System.

LLM provider: NVIDIA NIM. Embeddings: local HuggingFace model.
All relative paths are resolved against the project root, so the app works
regardless of the directory uvicorn is started from.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

_PATH_FIELDS = (
    "data_dir",
    "finance_data_path", "tech_data_path", "general_data_path",
    "finance_store_path", "tech_store_path", "general_store_path",
    "log_dir", "memory_db_path",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============= API Keys =============
    nvidia_api_key: str = ""
    google_api_key: Optional[str] = None  # unused (kept so old .env files still load)

    # ============= LLM =============
    default_llm_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    fast_llm_model: str = "nvidia/nemotron-3.5-bolt-e4b"
    powerful_llm_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    # Router only emits a tiny JSON: use a small non-reasoning model + short output.
    # Empty string => fall back to default_llm_model.
    # Reasoning-capable models (nemotron 3.5) spend the token budget "thinking" and can return
    # EMPTY answers; keep thinking off for latency and reliability.
    llm_enable_thinking: bool = False
    router_llm_model: str = "nvidia/nemotron-3.5-bolt-e4b"
    router_max_tokens: int = 200
    # Small talk (direct chat): fast non-reasoning model, streamed.
    chat_llm_model: str = "nvidia/nemotron-3.5-bolt-e4b"
    chat_max_tokens: int = 800
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    llm_timeout: int = 300

    # ============= Embedding =============
    # Multilingual (vi/en/...). Old English-only model: "all-MiniLM-L6-v2" (384 dims).
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    # CPU threads for torch (0 = all cores). torch defaulted to 1 thread here.
    torch_threads: int = 0

    # ============= Chunking / Retrieval =============
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 4
    # Cross-encoder cost grows linearly with candidates (CPU: ~0.7s each) => cap them.
    rerank_max_candidates: int = 10

    # Hybrid retrieval fusion weights (vector vs BM25)
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # Reranking
    # Multilingual cross-encoder. Scores are sigmoid-normalised to 0..1.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 512
    reranker_batch_size: int = 16
    use_reranker: bool = True
    # Rerank score below this => chunk considered not relevant (fallback kicks in)
    relevance_threshold: float = 0.05

    # ============= Paths =============
    data_dir: Path = Path("data")
    finance_data_path: Path = Path("data/raw/finance")
    tech_data_path: Path = Path("data/raw/tech")
    general_data_path: Path = Path("data/raw/general")
    finance_store_path: Path = Path("data/vectorstores/finance_store")
    tech_store_path: Path = Path("data/vectorstores/tech_store")
    general_store_path: Path = Path("data/vectorstores/general_store")
    log_dir: Path = Path("logs")

    # ============= Conversation memory =============
    memory_db_path: Path = Path("data/memory.db")
    memory_max_turns: int = 6        # most recent user+assistant pairs sent to the model
    memory_max_chars: int = 4000     # character budget for that history

    # ============= Documents =============
    supported_file_types: List[str] = [".pdf", ".txt", ".md", ".docx"]
    max_file_size_mb: int = 100

    # ============= API =============
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000"
    max_batch_queries: int = 10

    # ============= Logging =============
    log_level: str = "INFO"

    @field_validator(*_PATH_FIELDS, mode="after")
    @classmethod
    def _anchor_to_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else BASE_DIR / value

    @staticmethod
    def _local_or_hub(name: str) -> str:
        """Use a pre-downloaded copy in data/models/<name> if present, else the hub id."""
        local = BASE_DIR / "data" / "models" / name.split("/")[-1]
        return str(local) if (local / "config.json").exists() else name

    @property
    def embedding_model_path(self) -> str:
        return self._local_or_hub(self.embedding_model)

    @property
    def reranker_model_path(self) -> str:
        return self._local_or_hub(self.reranker_model)

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

# Ensure required directories exist
for _p in (
    settings.data_dir,
    settings.finance_data_path, settings.tech_data_path, settings.general_data_path,
    settings.finance_store_path, settings.tech_store_path, settings.general_store_path,
    settings.log_dir,
):
    _p.mkdir(parents=True, exist_ok=True)
