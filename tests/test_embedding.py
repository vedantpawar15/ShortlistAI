"""Tests for offline embedding generation and caching."""

from __future__ import annotations

import json

import numpy as np

from src.embedding import (
    PRIMARY_MODEL_NAME,
    EmbeddingModel,
    build_cache_key,
    coerce_documents,
    document_from_record,
    safe_path_name,
)


class FakeSentenceTransformer:
    """Deterministic fake embedding model for unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, documents, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        self.calls.append(
            {
                "documents": list(documents),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        rows = []
        for index, document in enumerate(documents):
            rows.append([float(index + 1), float(len(document)), 1.0])
        return np.asarray(rows, dtype=np.float32)


class FakeFrame:
    """Small DataFrame-like object for document coercion tests."""

    columns = ["semantic_document"]

    def __init__(self, records):
        self.records = records

    def to_dict(self, orient):
        assert orient == "records"
        return self.records


def make_loaded_embedding_model(tmp_path) -> tuple[EmbeddingModel, FakeSentenceTransformer]:
    model = EmbeddingModel(
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        batch_size=2,
        show_progress_bar=True,
        device="cpu",
    )
    fake = FakeSentenceTransformer()
    model.model = fake
    model.loaded_model_name = "fake-model"
    return model, fake


def test_default_model_uses_bge_small() -> None:
    model = EmbeddingModel()

    assert PRIMARY_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert model.model_name == "BAAI/bge-small-en-v1.5"
    assert "large" not in model.model_name.lower()


def test_embed_candidates_batches_and_persists_embeddings(tmp_path) -> None:
    model, fake = make_loaded_embedding_model(tmp_path)

    result = model.embed_candidates(["first profile", "second profile"])

    assert result.embeddings.shape == (2, 3)
    assert result.model_name == "fake-model"
    assert result.device == "cpu"
    assert result.embedding_path is not None and result.embedding_path.exists()
    assert result.metadata_path is not None and result.metadata_path.exists()
    assert fake.calls[0]["batch_size"] == 2
    assert fake.calls[0]["show_progress_bar"] is True
    assert fake.calls[0]["normalize_embeddings"] is True

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["document_count"] == 2
    assert metadata["embedding_shape"] == [2, 3]


def test_embed_jobs_uses_disk_cache_without_reencoding(tmp_path) -> None:
    model, fake = make_loaded_embedding_model(tmp_path)

    first = model.embed_jobs(["job one", "job two"])
    second = model.embed_jobs(["job one", "job two"])

    assert len(fake.calls) == 1
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    assert first.cache_key == second.cache_key


def test_load_embeddings_reads_persisted_vectors(tmp_path) -> None:
    model, _ = make_loaded_embedding_model(tmp_path)
    result = model.embed_candidates(["candidate profile"])

    loaded = model.load_embeddings("candidates", result.cache_key)

    np.testing.assert_array_equal(result.embeddings, loaded.embeddings)
    assert loaded.model_name == "fake-model"
    assert loaded.device == "cpu"


def test_coerce_documents_from_records_and_dataframe_like() -> None:
    records = [{"semantic_document": "clean candidate"}, {"description": "job description"}]
    frame = FakeFrame(records)

    assert coerce_documents(frame) == ["clean candidate", "job description"]
    assert coerce_documents(records) == ["clean candidate", "job description"]
    assert coerce_documents({"description": "one record"}) == ["one record"]
    assert coerce_documents("single document") == ["single document"]


def test_document_from_record_prefers_semantic_fields() -> None:
    record = {"name": "Asha", "semantic_document": "best text", "description": "fallback"}

    assert document_from_record(record) == "best text"


def test_cache_key_and_safe_path_name_are_stable() -> None:
    first = build_cache_key(["a", "b"], "model", 32, True, "candidates")
    second = build_cache_key(["a", "b"], "model", 32, True, "candidates")
    different = build_cache_key(["a", "c"], "model", 32, True, "candidates")

    assert first == second
    assert first != different
    assert safe_path_name("jobs/main model") == "jobs_main_model"
