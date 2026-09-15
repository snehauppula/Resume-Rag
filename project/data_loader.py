"""
Document loading and chunking for the RAG ingestion stage.

Notebook source: cells 2 (PDF load), 4 (split_documents).
Pipeline role: raw files -> LangChain Documents -> chunked Documents for embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import resolve_resume_dir, settings

logger = logging.getLogger(__name__)


def load_pdf_files(pdf_directory: Path) -> list[Document]:
    """
    Load all PDF files from a directory (notebook cell 2: load_pdf_files).

    Each page becomes a Document with source metadata.
    """
    all_documents: list[Document] = []
    pdf_dir = Path(pdf_directory)

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return all_documents

    logger.info("Loading PDF files from %s", pdf_dir)
    for file_path in pdf_files:
        try:
            logger.info("Loading %s", file_path)
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = str(file_path.resolve())
                doc.metadata["filename"] = file_path.name
                doc.metadata["candidate_id"] = file_path.stem
                doc.metadata["page"] = doc.metadata.get("page", 0)
                doc.metadata["type"] = "resume"
            all_documents.extend(documents)
            logger.info("Loaded %s page(s) from %s", len(documents), file_path.name)
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", file_path, exc)
            raise

    return all_documents


def load_txt_files(txt_directory: Path) -> list[Document]:
    """
    Load all .txt files from a directory (extension of notebook; same metadata pattern).

    Uses LangChain TextLoader for plain-text corpora.
    """
    all_documents: list[Document] = []
    txt_dir = Path(txt_directory)

    if not txt_dir.is_dir():
        raise FileNotFoundError(f"TXT directory does not exist: {txt_dir}")

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No TXT files found in %s", txt_dir)
        return all_documents

    logger.info("Loading TXT files from %s", txt_dir)
    for file_path in txt_files:
        try:
            logger.info("Loading %s", file_path)
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = str(file_path.resolve())
                doc.metadata["type"] = "txt"
            all_documents.extend(documents)
            logger.info("Loaded %s document(s) from %s", len(documents), file_path.name)
        except Exception as exc:
            logger.error("Failed to load TXT %s: %s", file_path, exc)
            raise

    return all_documents


def load_documents(data_directory: Path | None = None) -> list[Document]:
    """
    Load PDF resumes from the data directory.

    Inputs: optional path (defaults to config.resolve_resume_dir()).
    Outputs: list of page-level LangChain Documents.
    """
    data_dir = data_directory or resolve_resume_dir()
    data_dir = Path(data_dir)

    documents = load_pdf_files(data_dir)

    if not documents:
        raise ValueError(
            f"No documents loaded from {data_dir}. Add PDFs under data/resumes."
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into chunks (notebook cell 4: split_documents).

    Why: embeddings and retrieval work on smaller, focused passages.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list.")

    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)

    # Assign deterministic chunk index per candidate_id
    chunk_counters: dict[str, int] = {}
    for doc in split_docs:
        cid = doc.metadata.get("candidate_id", "unknown")
        idx = chunk_counters.get(cid, 0)
        doc.metadata["chunk_index"] = idx
        chunk_counters[cid] = idx + 1

    logger.info(
        "Split %s chunks from %s source document(s)",
        len(split_docs),
        len(documents),
    )
    return split_docs
