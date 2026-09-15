"""
ChromaDB vector store: persistence only, no embedding logic.

Notebook source: cell 8 (VectorStoreManager, vector_store_dir).
Pipeline role: store chunk text + embeddings + metadata for later retrieval.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from langchain_core.documents import Document

from .config import resolve_vector_store_dir, settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Persistent Chroma collection for document chunks (notebook: VectorStoreManager)."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = str(
            persist_directory or resolve_vector_store_dir()
        )
        self.client: chromadb.PersistentClient | None = None
        self.collection: Any = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> None:
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Vector store for PDF documents",
                    "hnsw:space": "cosine",
                },
            )
            logger.info(
                "Collection initialized: %s (%s documents)",
                self.collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            logger.error("Error initializing vector store: %s", exc)
            raise

    def get_document_count(self) -> int:
        """Return number of stored chunks."""
        if self.collection is None:
            return 0
        return int(self.collection.count())

    def reset_collection(self) -> int:
        """
        Delete and recreate the collection (clean re-ingest, no duplicates).

        Returns the number of chunks removed.
        """
        if self.client is None:
            raise ValueError("Vector store not initialized.")
        removed = self.get_document_count()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._initialize_vectorstore()
        return removed

    @staticmethod
    def _stable_chunk_id(doc: Document, index: int) -> str:
        """Same source+page+text -> same id (safe upsert on re-ingest)."""
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", doc.metadata.get("page_label", "")))
        text_key = doc.page_content[:500]
        payload = f"{source}|{page}|{text_key}|{index}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add chunked documents and precomputed embeddings (notebook: add_documents).

        Inputs: LangChain Documents + matching embedding matrix.
        Outputs: number of documents added.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length.")
        if self.collection is None:
            raise ValueError("Vector store not initialized.")

        logger.info("Adding %s documents to vector store", len(documents))

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents_list: list[str] = []
        embeddings_list: list[list[float]] = []

        for i, doc in enumerate(documents):
            ids.append(self._stable_chunk_id(doc, i))
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            documents_list.append(doc.page_content)
            embeddings_list.append(embeddings[i].tolist())

        self.collection.add(
            ids=ids,
            documents=documents_list,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        count = self.get_document_count()
        logger.info(
            "Successfully added %s documents. Total in store: %s",
            len(documents),
            count,
        )
        return len(documents)

    def similarity_search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Perform pure vector similarity search using ChromaDB.

        Inputs:
          query_embedding: the vector to search for (as a list or numpy array).
          top_k: number of nearest neighbors to return.

        Returns:
          A list of dictionaries, each containing:
            - "document": the chunk text.
            - "metadata": the chunk metadata (including candidate_id, filename, page, chunk_index, etc.).
            - "distance": the cosine distance.
            - "id": the chunk stable ID.
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized.")

        # Convert numpy array to list of floats if necessary
        if isinstance(query_embedding, np.ndarray):
            query_emb_list = query_embedding.tolist()
        else:
            query_emb_list = list(query_embedding)

        # If it's a 2D array/list (e.g. from generate_embeddings), make it 1D
        if len(query_emb_list) > 0 and isinstance(query_emb_list[0], list):
            query_emb_list = query_emb_list[0]

        results = self.collection.query(
            query_embeddings=[query_emb_list],
            n_results=top_k,
        )

        search_results = []
        if results and "documents" in results and results["documents"]:
            # Get list of results for the first query
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc_text, meta, dist, chunk_id in zip(docs, metas, distances, ids):
                search_results.append({
                    "document": doc_text,
                    "metadata": meta,
                    "distance": dist,
                    "id": chunk_id,
                })
        return search_results
