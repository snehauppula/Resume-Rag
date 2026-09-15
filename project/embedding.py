"""
Embedding generation for the RAG pipeline.

Notebook source: cell 6 (EmbeddingManager).
Pipeline role: text chunks -> dense vectors for vector search.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Wraps SentenceTransformer; model name from config.settings."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = SentenceTransformer(self.model_name)
            dim_fn = getattr(
                self.model, "get_embedding_dimension", None
            ) or getattr(self.model, "get_sentence_embedding_dimension")
            dim = dim_fn()
            logger.info(
                "Model %s loaded successfully with dimension %s",
                self.model_name,
                dim,
            )
        except Exception as exc:
            logger.error("Error loading model %s: %s", self.model_name, exc)
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to a numpy array (notebook: generate_embeddings).

        Inputs: list of chunk strings.
        Outputs: ndarray shape (n_texts, embedding_dim).
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        if not texts:
            raise ValueError("Cannot embed an empty text list.")

        logger.info("Generating embeddings for %s texts", len(texts))
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info("Embeddings generated successfully with shape %s", embeddings.shape)
        return embeddings
