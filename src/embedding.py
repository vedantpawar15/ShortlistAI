"""Offline-first embedding generation for semantic candidate ranking."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_utils import logger

PRIMARY_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_SUBDIR = "embeddings"
DEFAULT_MODEL_CACHE_SUBDIR = "sentence_transformers"


@dataclass(frozen=True)
class EmbeddingResult:
    """Embeddings and metadata produced by an embedding run."""

    embeddings: np.ndarray
    documents: list[str]
    model_name: str
    device: str
    cache_key: str
    embedding_path: Path | None = None
    metadata_path: Path | None = None


class EmbeddingModel:
    """Sentence Transformers wrapper for cached BGE-small embeddings."""

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL_NAME,
        fallback_model_name: str = FALLBACK_MODEL_NAME,
        models_dir: Path | str = "models",
        outputs_dir: Path | str = "outputs",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.device = device or detect_device()
        self.model: Any | None = None
        self.loaded_model_name: str | None = None
        self.model_cache_dir = self.models_dir / DEFAULT_MODEL_CACHE_SUBDIR
        self.embedding_store_dir = self.outputs_dir / DEFAULT_EMBEDDING_SUBDIR

    def load(self, local_files_only: bool = False) -> None:
        """Load cached weights first, then download BGE-small when cache is missing."""
        if self.model is not None:
            return

        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        original_device = self.device
        candidate_devices = [original_device] if original_device == "cpu" else [original_device, "cpu"]
        local_modes = [True] if local_files_only else [True, False]
        for local_only in local_modes:
            for candidate_model in (self.model_name, self.fallback_model_name):
                for candidate_device in candidate_devices:
                    self.device = candidate_device
                    try:
                        self.model = self._load_sentence_transformer(candidate_model, local_files_only=local_only)
                        self.loaded_model_name = candidate_model
                        logger.info("Loaded embedding model {} on {}", candidate_model, self.device)
                        return
                    except Exception as exc:
                        mode = "local cache" if local_only else "huggingface download"
                        errors.append(f"{candidate_model} on {candidate_device} via {mode}: {exc}")
                        logger.warning(
                            "Failed to load embedding model {} on {} via {}: {}",
                            candidate_model,
                            candidate_device,
                            mode,
                            exc,
                        )

        self.device = original_device
        message = "Unable to load any embedding model. " + " | ".join(errors)
        raise RuntimeError(message)

    def encode(
        self,
        texts: Sequence[str],
        cache_namespace: str = "documents",
        use_cache: bool = True,
        persist: bool = True,
    ) -> np.ndarray:
        """Encode texts in batches with optional disk caching."""
        result = self.embed_documents(texts, cache_namespace=cache_namespace, use_cache=use_cache, persist=persist)
        return result.embeddings

    def embed_candidates(
        self,
        candidates: Sequence[str] | Iterable[dict[str, Any]] | Any,
        use_cache: bool = True,
        persist: bool = True,
    ) -> EmbeddingResult:
        """Embed candidate semantic profile documents."""
        documents = coerce_documents(candidates)
        logger.info("Embedding {} candidate documents", len(documents))
        return self.embed_documents(documents, cache_namespace="candidates", use_cache=use_cache, persist=persist)

    def embed_jobs(
        self,
        jobs: Sequence[str] | Iterable[dict[str, Any]] | Any,
        use_cache: bool = True,
        persist: bool = True,
    ) -> EmbeddingResult:
        """Embed job description documents."""
        documents = coerce_documents(jobs)
        logger.info("Embedding {} job documents", len(documents))
        return self.embed_documents(documents, cache_namespace="jobs", use_cache=use_cache, persist=persist)

    def embed_documents(
        self,
        documents: Sequence[str],
        cache_namespace: str = "documents",
        use_cache: bool = True,
        persist: bool = True,
    ) -> EmbeddingResult:
        """Embed cleaned documents and persist embeddings plus metadata."""
        cleaned_documents = [str(document or "") for document in documents]
        self.load(local_files_only=False)
        model_name = self.loaded_model_name or self.model_name
        cache_key = build_cache_key(
            cleaned_documents,
            model_name,
            self.batch_size,
            self.normalize_embeddings,
            cache_namespace,
        )
        embedding_path, metadata_path = self._cache_paths(cache_namespace, cache_key)

        if use_cache and embedding_path.exists() and metadata_path.exists():
            logger.info("Loading cached embeddings from {}", embedding_path)
            embeddings = np.load(embedding_path)
            return EmbeddingResult(
                embeddings=embeddings,
                documents=cleaned_documents,
                model_name=model_name,
                device=self.device,
                cache_key=cache_key,
                embedding_path=embedding_path,
                metadata_path=metadata_path,
            )

        if not cleaned_documents:
            embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            embeddings = self._encode_with_model(cleaned_documents)

        result = EmbeddingResult(
            embeddings=embeddings,
            documents=cleaned_documents,
            model_name=model_name,
            device=self.device,
            cache_key=cache_key,
            embedding_path=embedding_path if persist else None,
            metadata_path=metadata_path if persist else None,
        )
        if persist:
            self.save_embeddings(result)
        return result

    def save_embeddings(self, result: EmbeddingResult) -> None:
        """Persist embeddings and metadata to disk."""
        if result.embedding_path is None or result.metadata_path is None:
            raise ValueError("EmbeddingResult does not include storage paths")
        result.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(result.embedding_path, result.embeddings)
        metadata = {
            "model_name": result.model_name,
            "device": result.device,
            "cache_key": result.cache_key,
            "document_count": len(result.documents),
            "embedding_shape": list(result.embeddings.shape),
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
        }
        result.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved embeddings to {}", result.embedding_path)

    def load_embeddings(self, cache_namespace: str, cache_key: str) -> EmbeddingResult:
        """Load persisted embeddings by namespace and cache key."""
        embedding_path, metadata_path = self._cache_paths(cache_namespace, cache_key)
        if not embedding_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"No stored embeddings for {cache_namespace}/{cache_key}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_path)
        return EmbeddingResult(
            embeddings=embeddings,
            documents=[],
            model_name=metadata["model_name"],
            device=metadata["device"],
            cache_key=cache_key,
            embedding_path=embedding_path,
            metadata_path=metadata_path,
        )

    def _load_sentence_transformer(self, model_name: str, local_files_only: bool) -> Any:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading SentenceTransformer model={} cache={} device={} local_files_only={}",
            model_name,
            self.model_cache_dir,
            self.device,
            local_files_only,
        )
        return SentenceTransformer(
            model_name,
            cache_folder=str(self.model_cache_dir),
            device=self.device,
            local_files_only=local_files_only,
        )

    def _encode_with_model(self, documents: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _cache_paths(self, namespace: str, cache_key: str) -> tuple[Path, Path]:
        safe_namespace = safe_path_name(namespace)
        directory = self.embedding_store_dir / safe_namespace
        return directory / f"{cache_key}.npy", directory / f"{cache_key}.json"


def detect_device() -> str:
    """Detect GPU acceleration and fall back to CPU when unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA GPU detected for embeddings")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS GPU detected for embeddings")
            return "mps"
    except Exception as exc:
        logger.debug("Torch device detection unavailable: {}", exc)
    logger.info("Using CPU for embeddings")
    return "cpu"


def coerce_documents(items: Sequence[str] | Iterable[dict[str, Any]] | Any) -> list[str]:
    """Convert strings, records, or DataFrame-like objects into documents."""
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    if isinstance(items, Mapping):
        return [document_from_record(dict(items))]
    if hasattr(items, "to_dict") and hasattr(items, "columns"):
        records = items.to_dict(orient="records")
        return [document_from_record(record) for record in records]
    if isinstance(items, Sequence) and all(isinstance(item, str) for item in items):
        return [str(item) for item in items]
    return [document_from_record(item) if isinstance(item, dict) else str(item) for item in items]


def document_from_record(record: dict[str, Any]) -> str:
    """Select the best text field from a candidate or job record."""
    preferred_fields = (
        "semantic_document",
        "document",
        "cleaned_document",
        "profile_document",
        "description",
        "job_description",
        "summary",
        "profile_summary",
        "text",
    )
    for field_name in preferred_fields:
        value = record.get(field_name)
        if value:
            return str(value)
    return " ".join(str(value) for value in record.values() if value is not None)


def build_cache_key(
    documents: Sequence[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    namespace: str,
) -> str:
    """Build a stable fingerprint for embedding cache reuse."""
    digest = hashlib.sha256()
    payload = {
        "namespace": namespace,
        "model_name": model_name,
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "documents": list(documents),
    }
    digest.update(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:24]


def safe_path_name(value: str) -> str:
    """Return a filesystem-safe path segment."""
    cleaned = re.sub(r"[^0-9a-zA-Z._-]+", "_", value).strip("._-")
    return cleaned or "embeddings"
