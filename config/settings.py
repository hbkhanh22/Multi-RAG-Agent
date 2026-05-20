"""
Configuration settings for Multi-RAG Agent System
Using Google LLM Models (Gemini) and Embeddings
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # ============= API Keys & Credentials =============
    google_api_key: str  # Google API Key (required)
    
    # ============= LLM Configuration =============
    # Default: gemini-2.0-flash (fastest, best for agents)
    # Alternative: gemini-1.5-pro (more powerful), gemini-1.5-flash (budget)
    default_llm_model: str = "gemini-2.0-flash"
    fast_llm_model: str = "gemini-1.5-flash"
    powerful_llm_model: str = "gemini-2.0-flash"
    
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    llm_timeout: int = 300
    
    # ============= Embedding Configuration =============
    # Google's embedding model
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768
    embedding_batch_size: int = 100
    
    # ============= Retrieval Configuration =============
    chunk_size: int = 512
    chunk_overlap: int = 64
    
    top_k_retrieval: int = 10
    top_k_rerank: int = 4
    
    # Hybrid retrieval weights (vector + BM25)
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    
    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    use_reranker: bool = True
    
    # ============= Vector Database Paths =============
    # Base data directory
    data_dir: Path = Path("data")
    
    # Raw document paths
    finance_data_path: Path = Path("data/raw/finance")
    tech_data_path: Path = Path("data/raw/tech")
    general_data_path: Path = Path("data/raw/general")
    
    # Vector store paths
    finance_store_path: Path = Path("data/vectorstores/finance_store")
    tech_store_path: Path = Path("data/vectorstores/tech_store")
    general_store_path: Path = Path("data/vectorstores/general_store")
    
    # Processed data path
    processed_data_path: Path = Path("data/processed")
    
    # ============= Agent Configuration =============
    router_temperature: float = 0.0
    router_max_tokens: int = 500
    
    supervisor_max_iterations: int = 3
    supervisor_timeout: int = 600
    
    synthesizer_temperature: float = 0.0
    synthesizer_max_tokens: int = 2000
    
    # ============= Document Processing =============
    supported_file_types: list[str] = [".pdf", ".txt", ".md", ".docx"]
    max_file_size_mb: int = 100
    
    # ============= Output Configuration =============
    include_source_attribution: bool = True
    include_confidence_scores: bool = True
    include_processing_time: bool = True
    
    # ============= Logging =============
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    
    # ============= Cache =============
    use_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # ============= Performance =============
    max_concurrent_requests: int = 5
    request_timeout: int = 300
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Ensure required directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.finance_data_path.mkdir(parents=True, exist_ok=True)
settings.tech_data_path.mkdir(parents=True, exist_ok=True)
settings.general_data_path.mkdir(parents=True, exist_ok=True)
settings.finance_store_path.mkdir(parents=True, exist_ok=True)
settings.tech_store_path.mkdir(parents=True, exist_ok=True)
settings.general_store_path.mkdir(parents=True, exist_ok=True)
settings.processed_data_path.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
