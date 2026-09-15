"""
Central configuration for the RAG pipeline.

Notebook source: cells 2, 4, 6, 8, 12, 18, 20 (paths and hyperparameters scattered there).
Why here: one place for paths, model names, and secrets so other modules stay testable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root and/or project/ (supports project/.env)
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
for _env_file in (_REPO_ROOT / ".env", _PKG_DIR / ".env"):
    if _env_file.is_file():
        load_dotenv(_env_file)

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _repo_root() -> Path:
    """Project lives under repo root; config.py is in project/."""
    return Path(__file__).resolve().parent.parent


def resolve_logs_dir() -> Path:
    """Resolve and create the logs/ directory at repo root."""
    path = _repo_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(
    name: str = "baseline_rag",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure console + optional file logging.

    File logs land in ``logs/<name>.log`` under the repo root.
    Safe to call more than once (handlers are not duplicated).
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

    if log_to_file:
        log_path = resolve_logs_dir() / f"{name}.log"
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_path)
            for h in root.handlers
        )
        if not already:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    return logging.getLogger(name)


def resolve_resume_dir() -> Path:
    """
    Resolve resume document folder.

    Works when cwd is repo root, project/, or workspace/.
    """
    candidates = (
        _repo_root() / "data" / "resumes",
        Path("data/resumes"),
        Path("../data/resumes"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "No resume directory found. Expected data/resumes at repo root "
        "(or data/resumes relative to current working directory)."
    )


def resolve_vector_store_dir() -> Path:
    """Resolve Chroma persist path."""
    if (_repo_root() / "data" / "resumes").is_dir():
        path = _repo_root() / "data" / "vector_store"
    elif Path("data/resumes").is_dir():
        path = Path("data/vector_store")
    else:
        path = Path("../data/vector_store")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable settings used across the RAG pipeline."""

    # Document ingestion (cell 4)
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Embeddings (cell 6)
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector store (cell 8)
    collection_name: str = "resume_baseline"

    # Retrieval defaults (cell 12)
    default_top_k: int = 4
    default_score_threshold: float = 0.0

    # LLM (cell 20)
    # llama-3.1-8b-instant was decommissioned by Groq on 2026-08-16
    groq_model_name: str = "openai/gpt-oss-20b"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024

    @property
    def groq_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY")

    def require_groq_api_key(self) -> str:
        key = self.groq_api_key
        if not key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY in .env or the environment."
            )
        return key


settings = Settings()
