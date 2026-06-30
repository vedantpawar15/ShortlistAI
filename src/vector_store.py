"""FAISS vector store abstractions for candidate retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.logging_utils import logger

try:
    import faiss
except ImportError:
    faiss = None


class FaissVectorStore:
    """Manage a local FAISS index for semantic candidate search."""

    def __init__(self, index_path: Path | str) -> None:
        self.index_path = Path(index_path)
        self.index: object | None = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build an in-memory FAISS index from embeddings."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build a FAISS index with no embeddings")
        if faiss is None:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self.index = vectors / np.maximum(norms, 1e-12)
        else:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        logger.info("Built FAISS index with {} vectors and dimension {}", vectors.shape[0], vectors.shape[1])

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("Cannot save before building or loading a FAISS index")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if faiss is None:
            with self.index_path.open("wb") as handle:
                np.save(handle, self.index)
        else:
            faiss.write_index(self.index, str(self.index_path))
        logger.info("Saved FAISS index to {}", self.index_path)

    def load(self) -> None:
        """Load a FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if faiss is None:
            with self.index_path.open("rb") as handle:
                self.index = np.load(handle)
        else:
            self.index = faiss.read_index(str(self.index_path))
        logger.info("Loaded FAISS index from {}", self.index_path)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search the vector index for nearest candidates."""
        if self.index is None:
            raise RuntimeError("Cannot search before building or loading a FAISS index")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2:
            raise ValueError("Query embedding must be a 1D or 2D array")
        if faiss is None:
            matrix = np.asarray(self.index, dtype=np.float32)
            query_norms = np.linalg.norm(query, axis=1, keepdims=True)
            normalized_query = query / np.maximum(query_norms, 1e-12)
            scores = normalized_query @ matrix.T
            k = max(1, min(top_k, matrix.shape[0]))
            indices = np.argsort(-scores, axis=1)[:, :k]
            distances = np.take_along_axis(scores, indices, axis=1)
            return distances, indices

        k = max(1, min(top_k, self.index.ntotal))
        distances, indices = self.index.search(query, k)
        return distances, indices

