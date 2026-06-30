"""FAISS vector store abstractions for candidate retrieval."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from loguru import logger


class FaissVectorStore:
    """Manage a local FAISS index for semantic candidate search."""

    def __init__(self, index_path: Path | str) -> None:
        self.index_path = Path(index_path)
        self.index: faiss.Index | None = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build an in-memory FAISS index from embeddings."""
        logger.debug("FAISS index build is not implemented for shape {}", embeddings.shape)
        raise NotImplementedError("Vector index building will be implemented later.")

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        logger.debug("FAISS index save is not implemented for {}", self.index_path)
        raise NotImplementedError("Vector index saving will be implemented later.")

    def load(self) -> None:
        """Load a FAISS index from disk."""
        logger.debug("FAISS index load is not implemented for {}", self.index_path)
        raise NotImplementedError("Vector index loading will be implemented later.")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search the vector index for nearest candidates."""
        logger.debug("FAISS search is not implemented with top_k={}", top_k)
        raise NotImplementedError("Vector search will be implemented later.")

