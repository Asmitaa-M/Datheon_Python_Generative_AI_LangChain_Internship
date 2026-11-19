"""Application configuration.

All settings come from environment variables (optionally loaded from a local
``.env`` file) and have safe defaults. Secrets such as ``GOOGLE_API_KEY`` are
only ever read from the environment - never hard-coded and never sent to the
browser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env once at import time. Real environment variables always win.
load_dotenv(BASE_DIR / ".env", override=False)

# Chroma telemetry is not needed for a local app.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

SUPPORTED_LLM_PROVIDERS = ("gemini", "mock")
SUPPORTED_EMBEDDING_BACKENDS = ("huggingface", "hash")


class ConfigurationError(ValueError):
    """Raised when settings are missing or invalid."""


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- LLM -----------------------------------------------------------
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.0
    google_api_key: str = ""

    # --- Embeddings ----------------------------------------------------
    embedding_backend: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval -----------------------------------------------------
    top_k: int = 4
    # Minimum cosine similarity (-1..1) for a chunk to be used as context.
    # Scores are NOT comparable between embedding models: re-calibrate this
    # value whenever EMBEDDING_MODEL / EMBEDDING_BACKEND is changed.
    relevance_threshold: float = 0.25

    # --- Storage -------------------------------------------------------
    chroma_dir: Path = BASE_DIR / "data" / "chroma_db"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    collection_name: str = "documents"

    # --- Uploads / display ---------------------------------------------
    max_upload_mb: int = 50
    snippet_chars: int = 400

    def __post_init__(self) -> None:
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            raise ConfigurationError(
                f"LLM_PROVIDER must be one of {SUPPORTED_LLM_PROVIDERS}, got {self.llm_provider!r}."
            )
        if self.embedding_backend not in SUPPORTED_EMBEDDING_BACKENDS:
            raise ConfigurationError(
                f"EMBEDDING_BACKEND must be one of {SUPPORTED_EMBEDDING_BACKENDS}, "
                f"got {self.embedding_backend!r}."
            )
        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
        if self.top_k <= 0:
            raise ConfigurationError("TOP_K must be greater than 0.")
        if not -1.0 <= self.relevance_threshold <= 1.0:
            raise ConfigurationError("RELEVANCE_THRESHOLD must be between -1 and 1.")
        if self.max_upload_mb <= 0:
            raise ConfigurationError("MAX_UPLOAD_MB must be greater than 0.")

    @property
    def embedding_id(self) -> str:
        """Identifier of the embedding configuration used to build an index."""
        return f"{self.embedding_backend}:{self.embedding_model}"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        defaults = cls()
        return cls(
            llm_provider=_env_str("LLM_PROVIDER", defaults.llm_provider).lower(),
            llm_model=_env_str("LLM_MODEL", defaults.llm_model),
            llm_temperature=_env_float("LLM_TEMPERATURE", defaults.llm_temperature),
            google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            embedding_backend=_env_str("EMBEDDING_BACKEND", defaults.embedding_backend).lower(),
            embedding_model=_env_str("EMBEDDING_MODEL", defaults.embedding_model),
            chunk_size=_env_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            top_k=_env_int("TOP_K", defaults.top_k),
            relevance_threshold=_env_float("RELEVANCE_THRESHOLD", defaults.relevance_threshold),
            chroma_dir=Path(_env_str("CHROMA_DIR", str(defaults.chroma_dir))),
            upload_dir=Path(_env_str("UPLOAD_DIR", str(defaults.upload_dir))),
            collection_name=_env_str("COLLECTION_NAME", defaults.collection_name),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", defaults.max_upload_mb),
            snippet_chars=_env_int("SNIPPET_CHARS", defaults.snippet_chars),
        )


def get_settings() -> Settings:
    """Return settings freshly read from the environment."""
    return Settings.from_env()
