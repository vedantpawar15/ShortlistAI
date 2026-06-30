"""FAISS vector store abstractions for candidate retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_utils import logger

try:
    import faiss
except ImportError:
    faiss = None


@dataclass(frozen=True)
class RetrievalMatch:
    """One ranked retrieval match returned from the vector store."""

    item_id: str
    score: float
    index: int
    metadata: dict[str, Any]


class FaissVectorStore:
    """Manage a local FAISS index for semantic candidate search."""

    def __init__(self, index_path: Path | str) -> None:
        self.index_path = Path(index_path)
        self.index: object | None = None
        self.item_ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []

    def build(
        self,
        embeddings: np.ndarray,
        item_ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build an in-memory FAISS index from embeddings."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build a FAISS index with no embeddings")
        self.item_ids = item_ids or [str(index) for index in range(vectors.shape[0])]
        if len(self.item_ids) != vectors.shape[0]:
            raise ValueError("item_ids length must match number of embeddings")
        self.metadata = metadata or [{} for _ in range(vectors.shape[0])]
        if len(self.metadata) != vectors.shape[0]:
            raise ValueError("metadata length must match number of embeddings")
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
        self._metadata_path().write_text(
            json.dumps({"item_ids": self.item_ids, "metadata": self.metadata}, indent=2),
            encoding="utf-8",
        )
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
        metadata_path = self._metadata_path()
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.item_ids = [str(item_id) for item_id in payload.get("item_ids", [])]
            self.metadata = [dict(item) for item in payload.get("metadata", [])]
        else:
            total = self._index_size()
            self.item_ids = [str(index) for index in range(total)]
            self.metadata = [{} for _ in range(total)]
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

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> list[RetrievalMatch]:
        """Search and return ranked retrieval matches with IDs and metadata."""
        distances, indices = self.search(query_embedding, top_k=top_k)
        matches: list[RetrievalMatch] = []
        for score, index in zip(distances[0], indices[0], strict=False):
            if index < 0:
                continue
            item_id = self.item_ids[index] if index < len(self.item_ids) else str(index)
            metadata = self.metadata[index] if index < len(self.metadata) else {}
            matches.append(
                RetrievalMatch(
                    item_id=item_id,
                    score=float(score),
                    index=int(index),
                    metadata=metadata,
                )
            )
        return matches

    def _metadata_path(self) -> Path:
        return self.index_path.with_suffix(f"{self.index_path.suffix}.meta.json")

    def _index_size(self) -> int:
        if self.index is None:
            return 0
        if faiss is None:
            return int(np.asarray(self.index).shape[0])
        return int(self.index.ntotal)

