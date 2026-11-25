"""
Configuration settings for Multi-RAG Agent System.

LLM provider: NVIDIA NIM. Embeddings: local HuggingFace model.
All relative paths are resolved against the project root, so the app works
regardless of the directory uvicorn is started from.
"""

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

_PATH_FIELDS = (
    "data_dir",
    "sport_data_path", "tech_data_path", "general_data_path",
    "sport_store_path", "tech_store_path", "general_store_path",
    "log_dir", "memory_db_path", "mcp_config_path", "mcp_workspace_dir", "onnx_model_dir",
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

    # ============= LLM =============
    default_llm_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    fast_llm_model: str = "nvidia/nemotron-3.5-bolt-e4b"
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
    # Cross-encoder cost grows linearly with candidates (CPU ~0.28s each) => cap them.
    # Eval (24 q): 10, 8 and 6 all give hit 0.96 / MRR 0.90; rerank 2.75s (10) vs 1.89s (6).
    # 8 keeps ~3 slots per expanded query for harder questions.
    rerank_max_candidates: int = 8

    # Index layout. Changing chunking/contextualisation needs a re-index: each version
    # lives in its own store sub-directory (<model>-<version>) with its own manifest, so
    # an old index stays usable while a new one is built. "" = legacy (512/64, no context).
    index_version: str = ""
    # Contextual retrieval: one LLM call per PAGE yields its section + a one-line summary,
    # prepended to every chunk of that page before embedding / BM25 (cached on disk).
    contextual_chunks: bool = False
    contextual_llm_model: str = ""     # empty => fast_llm_model
    contextual_max_page_chars: int = 6000
    # Multi-query expansion: make one rewrite a translation (vi <-> en), so a Vietnamese
    # question also searches the English documents (and vice versa).
    query_expansion_crosslingual: bool = True
    # Run multi-query expansion concurrently with the router LLM (it does not depend
    # on the route): saves one LLM round-trip before the first token.
    parallel_expansion: bool = True

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
    sport_data_path: Path = Path("data/raw/sport")
    tech_data_path: Path = Path("data/raw/tech")
    general_data_path: Path = Path("data/raw/general")
    sport_store_path: Path = Path("data/vectorstores/sport_store")
    tech_store_path: Path = Path("data/vectorstores/tech_store")
    general_store_path: Path = Path("data/vectorstores/general_store")
    log_dir: Path = Path("logs")

    # ============= MCP (Model Context Protocol) client =============
    mcp_enabled: bool = True
    mcp_config_path: Path = Path("config/mcp_servers.json")
    mcp_workspace_dir: Path = Path("data/mcp_workspace")   # root exposed to the filesystem server
    mcp_connect_timeout: int = 90      # first run may download the server package (npx/uvx)
    mcp_tool_timeout: int = 60
    mcp_max_iterations: int = 5        # tool-call rounds per question
    mcp_max_output_chars: int = 8000   # cap on one tool result sent back to the model

    # ============= Web search =============
    # Tavily (https://tavily.com) when a key is set; ddgs (scraping, often blocked) otherwise
    # and as fallback. Questions routed to the web are sent to the provider.
    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com"
    tavily_search_depth: str = "basic"   # basic (1 credit) | advanced (2 credits)
    tavily_timeout: float = 15.0
    web_search_attempts: int = 3         # ddgs attempts before giving up
    web_read_pages: int = 3              # result pages read when the provider gave no content (0 = off)
    web_page_timeout: float = 5.0
    web_page_chars: int = 3000           # relevant text kept per page
    local_utc_offset: float = 7.0        # hours; "today" for web questions (Vietnam = +7)

    # ============= Conversation memory =============
    memory_db_path: Path = Path("data/memory.db")
    memory_max_turns: int = 6        # most recent user+assistant pairs sent to the model
    memory_max_chars: int = 4000     # character budget for that history

    # ============= Evaluation / self-correction =============
    # After synthesis an LLM judge scores the answer against the retrieved context
    # (faithfulness, relevance, completeness: 1-5). If any score is below the
    # threshold, the judge's feedback is sent back to the LLM to revise the answer.
    self_eval_enabled: bool = True
    judge_llm_model: str = ""          # empty => default_llm_model
    judge_max_tokens: int = 600
    self_eval_min_score: int = 4       # a score below this fails the answer (3 let hallucinated numbers pass)
    self_eval_max_rounds: int = 1      # revise-and-recheck rounds (0 = judge only)
    judge_max_context_chars: int = 12000   # 3 web pages of web_page_chars

    # Offline RAGAS metrics (python -m src.evaluation.run --ragas). Called through
    # NIM's OpenAI-compatible endpoint.
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    ragas_llm_model: str = ""          # empty => judge_llm_model or default_llm_model
    ragas_max_tokens: int = 4096       # claim lists can be long; truncated JSON = failed metric
    ragas_concurrency: int = 2         # parallel LLM calls (NIM free tier ~40 requests/min; 4 hit 429s)

    # ============= Image generation =============
    # Text-to-image models on NIM; the request is first checked and rewritten by default_llm_model.
    image_enabled: bool = True
    image_base_url: str = "https://ai.api.nvidia.com/v1/genai"
    image_model: str = "black-forest-labs/flux.2-klein-4b"
    image_fallback_model: str = "black-forest-labs/flux.1-schnell"    # "" = none
    image_timeout: float = 60.0
    image_limit_per_hour: int = 20           # per user

    # ============= Chat attachments =============
    # Images, and PDF pages without a text layer, are read by a vision model (sent to NIM).
    vision_llm_model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    vision_fallback_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"   # "" = none
    vision_max_tokens: int = 3000
    vision_timeout: float = 60.0
    attachment_max_mb: int = 20
    attachment_max_files: int = 5            # per message
    attachment_max_pages: int = 300          # PDF
    attachment_max_scan_pages: int = 10      # one vision call per page
    attachment_max_chars: int = 200_000      # text kept per file
    attachment_context_chars: int = 24_000   # file text sent to the model per question
    attachment_unused_hours: int = 24        # purge uploads never used in a message

    # ============= Documents =============
    supported_file_types: List[str] = [".pdf", ".txt", ".md", ".docx"]
    max_file_size_mb: int = 100

    # ============= Infrastructure (Phase 4) =============
    # Relational DB for conversations, document registry and ingest jobs.
    # Empty => SQLite file at memory_db_path (dev). Production: postgresql+psycopg://...
    database_url: str = ""
    # Vector store backend: "chroma" (embedded, files under data/vectorstores) or
    # "pgvector" (chunks + embeddings in the same Postgres as database_url).
    vector_backend: str = "chroma"
    # Embedding / reranker runtime: "torch" (fp32 sentence-transformers), "onnx" (int8
    # ONNX Runtime models exported by scripts/export_onnx.py) or "auto" (onnx when the
    # exported models exist, else torch). Eval: identical hit/MRR, 1.6-3.3x faster.
    model_backend: str = "auto"
    onnx_model_dir: Path = Path("data/models/onnx")
    onnx_threads: int = 0              # 0 = onnxruntime default (all physical cores)

    # Where uploaded / raw documents live: "local" (data/raw/<domain>) or "s3"
    # (any S3-compatible store: AWS S3, MinIO, Cloudflare R2, GCS interop).
    document_storage: str = "local"
    s3_endpoint_url: str = ""          # e.g. http://minio:9000 ; empty => AWS default
    s3_bucket: str = "rag-documents"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"

    # Ingestion worker: "inline" = a thread inside the API process (single-node dev),
    # "external" = run `python -m src.worker` separately (API only enqueues jobs).
    ingest_worker: str = "inline"
    ingest_poll_interval: float = 2.0
    ingest_max_attempts: int = 3
    # Watch data/raw for changes and re-sync automatically (local storage only).
    watch_documents: bool = True
    watch_debounce_ms: int = 2000

    # Rate limiting (per client IP / API key, per process): requests per minute on
    # the query endpoints. 0 disables it.
    rate_limit_per_minute: int = 30
    max_upload_mb: int = 50
    # Admin token for document management endpoints. Empty => endpoints open
    # (dev only; set it in any shared deployment).
    admin_token: str = ""

    # Observability
    log_json: bool = False             # one JSON object per log line (for Loki/ELK/Cloud Logging)
    metrics_enabled: bool = True       # Prometheus metrics at /metrics
    # Langfuse tracing: enabled when the keys are set (Cloud or self-hosted host).
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ============= Security (Phase 5) =============
    # Authentication. Off => single anonymous user (dev, old frontend); every
    # conversation is visible. On => each user only sees their own conversations and
    # only admins manage documents.
    auth_enabled: bool = False
    jwt_secret: str = ""               # HS256 key; empty => random per process (dev only)
    access_token_minutes: int = 15
    refresh_token_days: int = 14
    refresh_reuse_grace_seconds: int = 10  # a just-rotated refresh token still works (tabs, reloads); -1 = strict
    allow_registration: bool = True    # self sign-up with email + password
    admin_emails: str = ""             # comma-separated: admin when signed in with Google; password sign-up refused
    min_password_length: int = 8
    cookie_secure: bool = False        # True behind HTTPS (refresh cookie only sent over TLS)
    login_attempts_per_minute: int = 5 # per IP+email, against password guessing
    # Google sign-in (OpenID Connect). Enabled when both are set.
    google_client_id: str = ""
    google_client_secret: str = ""
    public_base_url: str = "http://localhost:8000"   # where /api/auth/google/callback is reachable
    frontend_url: str = "http://localhost:5173"      # where to send the browser after Google login

    # Guardrails: an input safety classifier (NemoGuard) runs in parallel with the
    # router; unsafe requests get a refusal. Retrieved web/tool/document text is
    # scanned for prompt-injection phrases before it reaches the LLM.
    guardrails_enabled: bool = True
    guard_model: str = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
    guard_timeout: float = 5.0
    # Unsafe categories that do NOT block (a user sharing their own contact details
    # is flagged "PII/Privacy" by the classifier but is not an attack).
    guard_allowed_categories: str = "PII/Privacy"
    # Mask e-mails, phone numbers, ID / card numbers in logs and traces.
    pii_redaction: bool = True

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
    def effective_model_backend(self) -> str:
        """"torch" or "onnx" ("auto" resolved by checking the exported model files)."""
        backend = self.model_backend.lower()
        if backend != "auto":
            return backend
        ready = all(
            (self.onnx_model_dir / m.split("/")[-1] / "model_int8.onnx").exists()
            for m in (self.embedding_model, self.reranker_model)
        )
        return "onnx" if ready else "torch"

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url or f"sqlite:///{self.memory_db_path.as_posix()}"

    @property
    def is_postgres(self) -> bool:
        return self.sqlalchemy_url.startswith("postgresql")

    @property
    def domain_data_paths(self) -> dict:
        return {"sport": self.sport_data_path, "tech": self.tech_data_path, "general": self.general_data_path}

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

# Ensure required directories exist
for _p in (
    settings.data_dir,
    settings.sport_data_path, settings.tech_data_path, settings.general_data_path,
    settings.sport_store_path, settings.tech_store_path, settings.general_store_path,
    settings.log_dir,
):
    _p.mkdir(parents=True, exist_ok=True)
